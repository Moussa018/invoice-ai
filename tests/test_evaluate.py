import pandas as pd

from invoiceai.evaluate import (evaluate_anomalies, evaluate_extraction, field_correct, token_f1)

TRUTH = [
    {"id": "a", "vendor": "Atlas Cloud Hosting", "invoice_number": "ACH-2025-00001", "date": "2025-10-03",
     "currency": "MAD", "tax_amount": 320.36, "total_amount": 1922.16, "description": "Object storage 5 TB"},
    {"id": "b", "vendor": "Riad Hotels Group", "invoice_number": "RHG-2026-00002", "date": "2026-01-02",
     "currency": "MAD", "tax_amount": 10.0, "total_amount": 110.0, "description": "Breakfast package"},
]


def _pred(row, **override):
    p = {k: row[k] for k in ("vendor", "invoice_number", "date", "currency", "tax_amount",
                             "total_amount", "description")}
    p.update(override)
    return p


def test_field_correct_normalisation():
    assert field_correct("vendor", "ATLAS  cloud hosting.", "Atlas Cloud Hosting")
    assert field_correct("total_amount", "1,922.16", 1922.16)
    assert not field_correct("total_amount", 1922.17, 1922.16)
    assert not field_correct("date", None, "2025-10-03")
    assert field_correct("currency", "mad", "MAD")


def test_token_f1():
    assert token_f1("a b c", "a b c") == 1.0
    assert token_f1("a b", "c d") == 0.0
    assert 0 < token_f1("a b c", "a b d") < 1


def test_evaluate_extraction_counts_errors_missing_and_failures():
    preds = [
        {"id": "a", "pred": _pred(TRUTH[0], total_amount=1.0), "latency_s": 1.0,
         "input_tokens": 1000, "output_tokens": 100},
        {"id": "b", "pred": None, "latency_s": 3.0, "input_tokens": 1000, "output_tokens": 50},
    ]
    m = evaluate_extraction(TRUTH, preds, "t", price_in=1.0, price_out=5.0)
    assert m["field_accuracy"]["total_amount"] == 0.0
    assert m["field_accuracy"]["vendor"] == 0.5
    assert m["doc_exact_match"] == 0.0 and m["parse_failures"] == 1
    assert m["latency_s"]["mean"] == 2.0
    # 2000 in * $1/M + 150 out * $5/M
    assert m["cost_usd"] == round(2000 / 1e6 + 150 * 5 / 1e6, 4)
    only_a = evaluate_extraction(TRUTH, preds[:1], "t")      # "b" missing entirely
    assert only_a["parse_failures"] == 1 and only_a["cost_usd"] is None


def test_perfect_predictions():
    preds = [{"id": r["id"], "pred": _pred(r), "latency_s": 0.1} for r in TRUTH]
    m = evaluate_extraction(TRUTH, preds, "perfect", gpu_hour_cost=3600.0)
    assert m["macro_field_accuracy"] == 1.0 and m["doc_exact_match"] == 1.0
    assert m["description_token_f1"] == 1.0
    assert m["cost_usd"] == 0.2          # 0.2 s of GPU time at $3600/h


def test_evaluate_anomalies_precision_recall():
    res = pd.DataFrame({
        "id": ["1", "2", "3", "4"],
        "flag_duplicate": [True, False, False, False],
        "flag_amount": [False, True, True, False],
        "flag_category": [False, False, False, False],
        "flag_any": [True, True, True, False]})
    labels = {
        "1": {"is_duplicate": "1", "is_amount_outlier": "0", "is_misclassified": "0"},
        "2": {"is_duplicate": "0", "is_amount_outlier": "1", "is_misclassified": "0"},
        "3": {"is_duplicate": "0", "is_amount_outlier": "0", "is_misclassified": "0"},
        "4": {"is_duplicate": "0", "is_amount_outlier": "0", "is_misclassified": "1"}}
    m = evaluate_anomalies(res, labels)
    assert m["duplicate"]["precision"] == 1.0 and m["duplicate"]["recall"] == 1.0
    assert m["amount"]["precision"] == 0.5 and m["amount"]["recall"] == 1.0
    assert m["category"]["recall"] == 0.0
    assert m["any"]["tp"] == 2 and m["any"]["fp"] == 1 and m["any"]["fn"] == 1
