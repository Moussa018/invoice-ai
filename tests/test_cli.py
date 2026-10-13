import json
from pathlib import Path

from invoiceai import cli
from invoiceai.data import load_truth, ocr_path, read_jsonl
from invoiceai.extract_llm import ExtractionResult


class OracleExtractor:
    """Stands in for ClaudeExtractor: returns the ground truth for whatever invoice it is shown."""

    by_text: dict = {}

    def __init__(self, model="m", prompt="full", **kw):
        self.model, self.prompt = model, prompt

    def extract(self, text):
        t = self.by_text[text]
        pred = {k: t[k] for k in ("invoice_number", "date", "vendor", "currency", "tax_amount",
                                  "total_amount", "description")}
        return ExtractionResult(pred, 0.25, input_tokens=500, output_tokens=60, raw="{}")


def test_extract_evaluate_roundtrip_with_a_stub_model(fresh_data, tmp_path, monkeypatch):
    out, _ = fresh_data
    rows = load_truth(out, "test")[:5]
    for r in rows:
        ocr_path(out, r["id"]).parent.mkdir(exist_ok=True)
        text = f"ocr text of {r['id']}"
        ocr_path(out, r["id"]).write_text(text, encoding="utf-8")
        OracleExtractor.by_text[text] = r
    monkeypatch.setattr(cli, "ClaudeExtractor", OracleExtractor)

    preds = tmp_path / "preds.jsonl"
    cli.main(["extract", "--data", str(out), "--backend", "claude", "--split", "test",
              "--limit", "5", "--workers", "2", "--out", str(preds)])
    rec = read_jsonl(preds)
    assert [r["id"] for r in rec] == [r["id"] for r in rows]
    assert rec[0]["input_tokens"] == 500 and rec[0]["latency_s"] == 0.25

    # evaluate against exactly those five invoices
    truth_subset = tmp_path / "subset"
    truth_subset.mkdir()
    (truth_subset / "truth.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    cli.main(["evaluate", "--data", str(truth_subset), "--preds", str(preds), "--name", "stub",
              "--split", "all", "--results", str(tmp_path), "--price-in", "1", "--price-out", "5"])
    m = json.loads((tmp_path / "stub_metrics.json").read_text())
    assert m["macro_field_accuracy"] == 1.0 and m["doc_exact_match"] == 1.0
    assert m["tokens"]["input"] == 2500 and m["cost_usd"] == round(2500 / 1e6 + 300 * 5 / 1e6, 4)

    cli.main(["compare", "--results", str(tmp_path)])
    assert "stub" in (tmp_path / "comparison.md").read_text()


def test_analyze_oracle_writes_report_and_metrics(small_data, tmp_path):
    out, _ = small_data
    cli.main(["analyze", "--oracle", "--data", str(out), "--results", str(tmp_path)])
    assert (Path(tmp_path) / "anomalies_oracle.csv").exists()
    m = json.loads((Path(tmp_path) / "anomaly_metrics_oracle.json").read_text())
    assert m["duplicate"]["recall"] == 1.0 and m["n_docs"] == 42
