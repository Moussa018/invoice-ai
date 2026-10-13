import csv
from pathlib import Path

from invoiceai.data import load_labels, load_ledger, load_truth
from invoiceai.generate import generate_dataset


def test_files_and_counts(small_data):
    out, records = small_data
    assert len(records) == 40 + round(40 * 0.04)
    for r in records:
        assert (Path(out) / r["file"]).exists()
    assert len(load_truth(out)) == len(records)


def test_amounts_are_arithmetically_consistent(small_data):
    _, records = small_data
    for r in records:
        sub = round(sum(li["amount"] for li in r["line_items"]), 2)
        assert r["subtotal"] == sub
        assert abs(r["total_amount"] - (sub + r["tax_amount"])) < 0.011
        assert abs(r["tax_amount"] - round(sub * r["vat_rate"], 2)) < 0.011


def test_duplicates_copy_their_source_and_split(small_data):
    _, records = small_data
    by_id = {r["id"]: r for r in records}
    dups = [r for r in records if r["is_duplicate"]]
    assert dups
    for d in dups:
        src = by_id[d["duplicate_of"]]
        assert d["vendor"] == src["vendor"]
        assert d["total_amount"] == src["total_amount"]
        assert d["date"] == src["date"]
        assert d["split"] == src["split"]
        if d["dup_kind"] == "exact":
            assert d["invoice_number"] == src["invoice_number"]
        else:
            assert d["invoice_number"] != src["invoice_number"]
        assert d["id"] > src["id"]


def test_ledger_differs_from_truth_only_for_misclassified(small_data):
    out, records = small_data
    ledger, labels = load_ledger(out), load_labels(out)
    for r in records:
        mismatch = ledger[r["id"]] != r["true_category"]
        assert mismatch == (labels[r["id"]]["is_misclassified"] == "1")


def test_anomaly_types_are_disjoint(small_data):
    _, records = small_data
    for r in records:
        flags = [r["is_duplicate"], r["is_amount_outlier"], r["is_misclassified"]]
        assert sum(map(bool, flags)) <= 1


def test_generation_is_deterministic(tmp_path):
    a = generate_dataset(tmp_path / "a", n=10, seed=5, render=False)
    b = generate_dataset(tmp_path / "b", n=10, seed=5, render=False)
    assert [r["invoice_number"] for r in a] == [r["invoice_number"] for r in b]
    assert (tmp_path / "a" / "truth.jsonl").read_text() == (tmp_path / "b" / "truth.jsonl").read_text()
    with open(tmp_path / "a" / "labels.csv") as f:
        assert len(list(csv.DictReader(f))) == len(a)
