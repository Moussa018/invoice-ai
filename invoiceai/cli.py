"""Command line entry point: ``python -m invoiceai.cli <command>``."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import extract_rules
from .anomalies import analyze, frame_from_predictions, frame_from_truth
from .data import load_labels, load_ledger, load_ocr_text, load_truth, read_jsonl, write_jsonl
from .evaluate import (evaluate_anomalies, evaluate_extraction, format_anomaly_table,
                       format_extraction_table, save_json)
from .extract_llm import ClaudeExtractor, ExtractionResult, LocalHFExtractor
from .generate import generate_dataset
from .ocr import ocr_all
from .schema import ALL_KEYS


def cmd_generate(a):
    recs = generate_dataset(a.data, a.n, a.seed, a.noise)
    print(f"wrote {len(recs)} invoices to {a.data}")


def cmd_ocr(a):
    n = ocr_all(a.data, a.workers, a.force)
    print(f"OCR'd {n} images")


def _rules_result(text: str) -> ExtractionResult:
    import time

    t0 = time.perf_counter()
    pred = extract_rules.extract(text)
    return ExtractionResult(pred, time.perf_counter() - t0)


def cmd_extract(a):
    rows = load_truth(a.data, a.split)
    if a.limit:
        rows = rows[: a.limit]
    if a.backend == "rules":
        fn = lambda text: _rules_result(text)  # noqa: E731
        workers = 1
    elif a.backend == "claude":
        ex = ClaudeExtractor(model=a.model, prompt=a.prompt)
        fn, workers = ex.extract, a.workers
    else:
        ex = LocalHFExtractor(a.base_model, a.adapter, a.prompt)
        fn, workers = ex.extract, 1

    def run(r):
        res = fn(load_ocr_text(a.data, r["id"]))
        return {"id": r["id"], "pred": res.pred, "latency_s": round(res.latency_s, 5),
                "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
                "raw": res.raw, "error": res.error}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        out = list(pool.map(run, rows))
    write_jsonl(a.out, out)
    bad = sum(1 for o in out if o["pred"] is None)
    print(f"wrote {len(out)} predictions to {a.out} ({bad} unparseable)")


def cmd_evaluate(a):
    truth = load_truth(a.data, a.split)
    preds = read_jsonl(a.preds)
    name = a.name or Path(a.preds).stem
    m = evaluate_extraction(truth, preds, name, a.price_in, a.price_out, a.gpu_hour_cost)
    m["split"] = a.split
    save_json(Path(a.results) / f"{name}_metrics.json", m)
    print(format_extraction_table([m]))


def cmd_analyze(a):
    ledger = load_ledger(a.data)
    labels = load_labels(a.data)
    if a.oracle:
        df, name = frame_from_truth(load_truth(a.data), ledger), "oracle"
    else:
        df = frame_from_predictions(read_jsonl(a.preds), ledger)
        name = a.name or Path(a.preds).stem
    res = analyze(df)
    Path(a.results).mkdir(parents=True, exist_ok=True)
    cols = ["id", "vendor", "invoice_number", "date", "total_amount", "booked_category",
            "flag_duplicate", "duplicate_of", "flag_amount", "amount_z", "amount_reason",
            "flag_category", "predicted_category", "category_conf"]
    res[cols].to_csv(Path(a.results) / f"anomalies_{name}.csv", index=False)
    m = evaluate_anomalies(res, labels)
    save_json(Path(a.results) / f"anomaly_metrics_{name}.json", m)
    print(format_anomaly_table({name: m}))


def cmd_compare(a):
    ms = [json.loads(p.read_text()) for p in sorted(Path(a.results).glob("*_metrics.json"))
          if not p.name.startswith("anomaly_")]
    if not ms:
        sys.exit("no *_metrics.json files found; run `evaluate` first")
    table = format_extraction_table(ms)
    (Path(a.results) / "comparison.md").write_text(table + "\n", encoding="utf-8")
    print(table)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="invoiceai")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("generate", help="create synthetic invoices + ground truth")
    s.add_argument("--data", default="data")
    s.add_argument("--n", type=int, default=400)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--noise", type=float, default=1.0)
    s.set_defaults(fn=cmd_generate)

    s = sub.add_parser("ocr", help="run Tesseract over all images (cached)")
    s.add_argument("--data", default="data")
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_ocr)

    s = sub.add_parser("extract", help="extract fields with a chosen backend")
    s.add_argument("--data", default="data")
    s.add_argument("--backend", choices=["rules", "claude", "local"], required=True)
    s.add_argument("--split", default="test", choices=["all", "train", "test"])
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--out", required=True)
    s.add_argument("--model", default="claude-haiku-5-5", help="Claude model id (backend=claude)")
    s.add_argument("--prompt", default=None, choices=["full", "compact"])
    s.add_argument("--workers", type=int, default=4, help="parallel API calls (backend=claude)")
    s.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct", help="backend=local")
    s.add_argument("--adapter", default=None, help="LoRA adapter dir (backend=local)")
    s.set_defaults(fn=cmd_extract)

    s = sub.add_parser("evaluate", help="field-level accuracy, latency and cost")
    s.add_argument("--data", default="data")
    s.add_argument("--preds", required=True)
    s.add_argument("--name", default=None)
    s.add_argument("--split", default="test", choices=["all", "train", "test"])
    s.add_argument("--results", default="results")
    s.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens")
    s.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens")
    s.add_argument("--gpu-hour-cost", type=float, default=None, help="USD per GPU hour (local models)")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("analyze", help="flag duplicates, odd amounts, misclassified costs")
    s.add_argument("--data", default="data")
    s.add_argument("--preds", default=None)
    s.add_argument("--oracle", action="store_true", help="use ground-truth fields instead of predictions")
    s.add_argument("--name", default=None)
    s.add_argument("--results", default="results")
    s.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("compare", help="merge all *_metrics.json into one table")
    s.add_argument("--results", default="results")
    s.set_defaults(fn=cmd_compare)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.cmd == "extract" and a.prompt is None:
        a.prompt = "compact" if a.backend == "local" else "full"
    if a.cmd == "analyze" and not a.oracle and not a.preds:
        sys.exit("analyze needs --preds FILE or --oracle")
    a.fn(a)


if __name__ == "__main__":
    main()
