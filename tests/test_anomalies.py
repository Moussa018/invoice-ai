import numpy as np
import pandas as pd

from invoiceai.anomalies import analyze, canonical_vendors, detect_duplicates

VENDORS = {
    "Alpha Cloud Hosting": ("Software & Cloud", "cloud server subscription; managed database plan", 2800),
    "Beta Travel Agency": ("Travel", "flight tickets; hotel accommodation", 3600),
    "Gamma Paper Supplies": ("Office Supplies", "a4 paper reams; toner cartridges", 450),
}


def _frame(seed=0, per_vendor=20):
    rng = np.random.default_rng(seed)
    rows = []
    for vendor, (cat, desc, typical) in VENDORS.items():
        for k in range(per_vendor):
            total = round(typical * float(np.exp(rng.normal(0, 0.25))), 2)
            rows.append({
                "id": f"{vendor[:3].lower()}_{k:02d}", "vendor": vendor,
                "invoice_number": f"{vendor[:3].upper()}-{1000 + k}",
                "date": f"2026-{1 + k % 9:02d}-{1 + (k * 3) % 27:02d}", "currency": "MAD",
                "total_amount": total, "tax_amount": round(total / 6, 2),  # 20% VAT on subtotal
                "description": desc, "booked_category": cat})
    return pd.DataFrame(rows)


def test_canonical_vendors_merges_ocr_noise():
    keys = canonical_vendors(["Atlas Cloud Hosting", "Atlas Cloud Hosfing", "Riad Hotels Group", ""])
    assert keys[0] == keys[1] and keys[0] != keys[2] and keys[3] == ""


def test_clean_data_has_no_flags():
    res = analyze(_frame())
    assert not res["flag_duplicate"].any()
    assert not res["flag_amount"].any()
    assert not res["flag_category"].any()


def test_exact_and_near_duplicates_are_flagged_once():
    df = _frame()
    exact = df.iloc[3].copy(); exact["id"] = "dup_exact"
    near = df.iloc[25].copy(); near["id"] = "dup_near"; near["invoice_number"] = "BET-9999"
    df = pd.concat([df, pd.DataFrame([exact, near])], ignore_index=True)
    df["vendor_key"] = canonical_vendors(df["vendor"].tolist())
    dup_of = detect_duplicates(df)
    flagged = dict(zip(df["id"], dup_of))
    assert flagged["dup_exact"] == df.iloc[3]["id"]
    assert flagged["dup_near"] == df.iloc[25]["id"]
    assert sum(v != "" for v in flagged.values()) == 2


def test_amount_outlier_is_flagged():
    df = _frame()
    df.loc[5, "total_amount"] *= 9
    df.loc[5, "tax_amount"] *= 9
    res = analyze(df)
    assert bool(res.loc[5, "flag_amount"]) and res.loc[5, "amount_reason"] == "amount"
    assert res["flag_amount"].sum() == 1


def test_tax_mismatch_is_flagged():
    df = _frame()
    df.loc[7, "tax_amount"] = df.loc[7, "total_amount"] * 0.5
    res = analyze(df)
    assert bool(res.loc[7, "flag_amount"]) and res.loc[7, "amount_reason"] == "tax_mismatch"


def test_misclassified_category_is_flagged():
    df = _frame()
    df.loc[2, "booked_category"] = "Travel"
    res = analyze(df)
    assert bool(res.loc[2, "flag_category"])
    assert res.loc[2, "predicted_category"] == "Software & Cloud"
    assert res["flag_category"].sum() == 1


def test_missing_values_do_not_crash():
    df = _frame()
    df.loc[0, ["vendor", "total_amount", "date", "description"]] = None
    res = analyze(df)
    assert len(res) == len(df)
