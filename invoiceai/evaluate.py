"""Evaluation: field-level extraction accuracy, latency/cost, and anomaly metrics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .schema import AMOUNT_FIELDS, DESCRIPTION, FIELDS, norm_key, parse_amount, tokens


# --------------------------------------------------------------- extraction
def field_correct(field: str, pred, truth) -> bool:
    if pred is None:
        return False
    if field in AMOUNT_FIELDS:
        p = parse_amount(pred)
        return p is not None and abs(p - float(truth)) < 0.005
    if field == "currency":
        return str(pred).strip().upper() == str(truth).strip().upper()
    if field in ("vendor", "invoice_number"):
        return norm_key(pred) == norm_key(truth)
    return str(pred).strip() == str(truth).strip()


def token_f1(pred, truth) -> float:
    a, b = tokens(pred), tokens(truth)
    if not a or not b:
        return 0.0
    common = 0
    pool = list(b)
    for t in a:
        if t in pool:
            pool.remove(t)
            common += 1
    if common == 0:
        return 0.0
    p, r = common / len(a), common / len(b)
    return 2 * p * r / (p + r)


def evaluate_extraction(truth_rows: list[dict], pred_rows: list[dict], name: str = "run",
                        price_in: float | None = None, price_out: float | None = None,
                        gpu_hour_cost: float | None = None) -> dict:
    """Score predictions against truth. Missing predictions count as wrong."""
    preds = {r["id"]: r for r in pred_rows}
    n = len(truth_rows)
    correct = {f: 0 for f in FIELDS}
    doc_exact = 0
    f1s = []
    failures = 0
    lat, tin, tout = [], 0, 0
    for t in truth_rows:
        row = preds.get(t["id"])
        p = (row or {}).get("pred")
        if row is None or p is None:
            failures += 1
            f1s.append(0.0)
            if row is not None:
                lat.append(row.get("latency_s", 0.0))
                tin += row.get("input_tokens", 0)
                tout += row.get("output_tokens", 0)
            continue
        ok = {f: field_correct(f, p.get(f), t[f]) for f in FIELDS}
        for f, v in ok.items():
            correct[f] += int(v)
        doc_exact += int(all(ok.values()))
        f1s.append(token_f1(p.get(DESCRIPTION), t.get(DESCRIPTION)))
        lat.append(row.get("latency_s", 0.0))
        tin += row.get("input_tokens", 0)
        tout += row.get("output_tokens", 0)

    lat_arr = np.array(lat) if lat else np.array([0.0])
    metrics = {
        "name": name, "n_docs": n, "parse_failures": failures,
        "field_accuracy": {f: round(correct[f] / n, 4) for f in FIELDS},
        "macro_field_accuracy": round(sum(correct.values()) / (n * len(FIELDS)), 4),
        "doc_exact_match": round(doc_exact / n, 4),
        "description_token_f1": round(float(np.mean(f1s)), 4),
        "latency_s": {"mean": round(float(lat_arr.mean()), 4),
                      "p50": round(float(np.percentile(lat_arr, 50)), 4),
                      "p95": round(float(np.percentile(lat_arr, 95)), 4),
                      "total": round(float(lat_arr.sum()), 2)},
        "tokens": {"input": tin, "output": tout,
                   "input_per_doc": round(tin / n, 1), "output_per_doc": round(tout / n, 1)},
        "cost_usd": None, "cost_per_1k_docs_usd": None,
    }
    cost = None
    if price_in is not None and price_out is not None:
        cost = tin / 1e6 * price_in + tout / 1e6 * price_out
    elif gpu_hour_cost is not None:
        cost = float(lat_arr.sum()) / 3600 * gpu_hour_cost
    if cost is not None:
        metrics["cost_usd"] = round(cost, 4)
        metrics["cost_per_1k_docs_usd"] = round(cost / n * 1000, 4)
    return metrics


# ----------------------------------------------------------------- anomalies
def _prf(pred: np.ndarray, gold: np.ndarray) -> dict:
    tp = int((pred & gold).sum())
    fp = int((pred & ~gold).sum())
    fn = int((~pred & gold).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 3), "recall": round(r, 3),
            "f1": round(f, 3), "n_positive": int(gold.sum())}


def evaluate_anomalies(result, labels: dict[str, dict]) -> dict:
    """``result`` is the DataFrame returned by anomalies.analyze."""
    ids = result["id"].tolist()
    gold = {
        "duplicate": np.array([labels[i]["is_duplicate"] == "1" for i in ids]),
        "amount": np.array([labels[i]["is_amount_outlier"] == "1" for i in ids]),
        "category": np.array([labels[i]["is_misclassified"] == "1" for i in ids]),
    }
    pred = {
        "duplicate": result["flag_duplicate"].to_numpy(dtype=bool),
        "amount": result["flag_amount"].to_numpy(dtype=bool),
        "category": result["flag_category"].to_numpy(dtype=bool),
    }
    out = {k: _prf(pred[k], gold[k]) for k in gold}
    any_gold = gold["duplicate"] | gold["amount"] | gold["category"]
    out["any"] = _prf(result["flag_any"].to_numpy(dtype=bool), any_gold)
    out["n_docs"] = len(ids)
    return out


# ------------------------------------------------------------------ reports
def format_extraction_table(metrics_list: list[dict]) -> str:
    cols = FIELDS
    head = ["run", "docs"] + cols + ["macro", "exact doc", "desc F1", "p50 s", "p95 s", "in tok/doc",
                                      "out tok/doc", "$/1k docs"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for m in metrics_list:
        cost = m.get("cost_per_1k_docs_usd")
        row = [m["name"], str(m["n_docs"])] + [f"{m['field_accuracy'][c]:.1%}" for c in cols] + [
            f"{m['macro_field_accuracy']:.1%}", f"{m['doc_exact_match']:.1%}",
            f"{m['description_token_f1']:.2f}", f"{m['latency_s']['p50']:.2f}",
            f"{m['latency_s']['p95']:.2f}", f"{m['tokens']['input_per_doc']:.0f}",
            f"{m['tokens']['output_per_doc']:.0f}", "n/a" if cost is None else f"{cost:.2f}"]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def format_anomaly_table(named: dict[str, dict]) -> str:
    head = ["input", "detector", "positives", "precision", "recall", "F1", "TP", "FP", "FN"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for name, res in named.items():
        for k in ("duplicate", "amount", "category", "any"):
            r = res[k]
            lines.append(f"| {name} | {k} | {r['n_positive']} | {r['precision']:.2f} | {r['recall']:.2f} | "
                         f"{r['f1']:.2f} | {r['tp']} | {r['fp']} | {r['fn']} |")
    return "\n".join(lines)


def save_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
