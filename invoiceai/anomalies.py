"""Anomaly detection on extracted invoices.

Three detectors, each flagging a different problem:

* duplicates          - same vendor and invoice number, or same vendor, amount and
                        (nearly) the same date. The later invoice is flagged.
* odd amounts         - IsolationForest on a vendor-relative amount score and a
                        tax-ratio check, confirmed by a robust z-score threshold.
* misclassified costs - a text classifier (TF-IDF + logistic regression) predicts the
                        category out-of-fold; a confident disagreement with the
                        booked category is flagged.
"""
from __future__ import annotations

from datetime import date
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, make_pipeline

from .schema import norm_key

COLUMNS = ["id", "vendor", "invoice_number", "date", "currency", "total_amount", "tax_amount",
           "description", "booked_category"]


# ------------------------------------------------------------------- inputs
def frame_from_predictions(pred_rows: list[dict], ledger: dict[str, str]) -> pd.DataFrame:
    rows = []
    for r in pred_rows:
        p = r.get("pred") or {}
        rows.append({"id": r["id"], "vendor": p.get("vendor"), "invoice_number": p.get("invoice_number"),
                     "date": p.get("date"), "currency": p.get("currency"),
                     "total_amount": p.get("total_amount"), "tax_amount": p.get("tax_amount"),
                     "description": p.get("description"), "booked_category": ledger.get(r["id"])})
    return pd.DataFrame(rows, columns=COLUMNS)


def frame_from_truth(truth_rows: list[dict], ledger: dict[str, str]) -> pd.DataFrame:
    rows = [{**{k: t.get(k) for k in COLUMNS if k != "booked_category"},
             "booked_category": ledger.get(t["id"])} for t in truth_rows]
    return pd.DataFrame(rows, columns=COLUMNS)


# ------------------------------------------------------------------- vendors
def canonical_vendors(names, threshold: float = 0.8) -> list[str]:
    """Map OCR-noisy vendor names onto a shared key (greedy fuzzy clustering)."""
    reps: list[str] = []
    out = []
    for n in names:
        k = norm_key(n) if isinstance(n, str) else ""
        if not k:
            out.append("")
            continue
        match = next((r for r in reps if SequenceMatcher(None, k, r).ratio() >= threshold), None)
        if match is None:
            reps.append(k)
            match = k
        out.append(match)
    return out


# ---------------------------------------------------------------- duplicates
def _to_date(s):
    try:
        return date.fromisoformat(s) if isinstance(s, str) else None
    except ValueError:
        return None


def detect_duplicates(df: pd.DataFrame, date_window: int = 3, amount_tol: float = 0.01) -> pd.Series:
    """Return a Series (index = df.index) holding the id of the earlier twin, or ''."""
    keys = df["vendor_key"].tolist()
    nums = [norm_key(x) for x in df["invoice_number"]]
    amts = df["total_amount"].astype(float).tolist()
    dts = [_to_date(x) for x in df["date"]]
    ids = df["id"].tolist()
    kept: list[int] = []
    dup_of = [""] * len(df)
    for i in range(len(df)):
        for j in kept:
            if not keys[i] or keys[i] != keys[j]:
                continue
            same_num = bool(nums[i]) and nums[i] == nums[j]
            same_amt_date = (
                not np.isnan(amts[i]) and not np.isnan(amts[j]) and abs(amts[i] - amts[j]) <= amount_tol
                and dts[i] is not None and dts[j] is not None
                and abs((dts[i] - dts[j]).days) <= date_window)
            if same_num or same_amt_date:
                dup_of[i] = ids[j]
                break
        else:
            kept.append(i)
    return pd.Series(dup_of, index=df.index)


# ------------------------------------------------------------------- amounts
def detect_amount_outliers(df: pd.DataFrame, contamination: float = 0.06, z_thresh: float = 3.0,
                           tax_tol: float = 0.05, min_history: int = 4, seed: int = 0) -> pd.DataFrame:
    total = df["total_amount"].astype(float)
    tax = df["tax_amount"].astype(float)
    valid = total.notna() & (total > 0)
    logv = np.log(total.where(valid))
    key = df["vendor_key"]

    med = logv.groupby(key).transform("median")
    mad = (logv - med).abs().groupby(key).transform("median")
    scale = np.maximum(1.4826 * mad, 0.15)
    z = ((logv - med) / scale).where(key.groupby(key).transform("size") >= min_history, 0.0)

    ratio = (tax / (total - tax)).replace([np.inf, -np.inf], np.nan)
    ratio_med = ratio.groupby(key).transform("median")
    tax_dev = (ratio - ratio_med).abs()

    # Hinge the tax feature so it is exactly 0 for consistent invoices. Otherwise
    # IsolationForest spends half its random splits on rounding noise in this column.
    feats = pd.DataFrame({"z": z.fillna(0.0),
                          "tax_dev": (tax_dev.fillna(0.0) - 0.01).clip(lower=0.0, upper=1.0)})
    flag = pd.Series(False, index=df.index)
    if valid.sum() >= 10:
        iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
        outlier = pd.Series(iso.fit_predict(feats.to_numpy()) == -1, index=df.index)
        flag = outlier & ((feats["z"].abs() >= z_thresh) | (feats["tax_dev"] >= tax_tol)) & valid
    reason = np.where(~flag, "", np.where(feats["z"].abs() >= z_thresh, "amount", "tax_mismatch"))
    return pd.DataFrame({"flag_amount": flag, "amount_z": feats["z"].round(2), "amount_reason": reason},
                        index=df.index)


# ---------------------------------------------------------------- categories
def _category_model():
    vec = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
    ])
    return make_pipeline(vec, LogisticRegression(max_iter=3000, C=5.0))


def detect_misclassified(df: pd.DataFrame, min_conf: float = 0.6, seed: int = 0) -> pd.DataFrame:
    text = (df["vendor"].fillna("") + " " + df["description"].fillna("")).str.strip()
    y = df["booked_category"].fillna("")
    counts = y.value_counts()
    empty = pd.DataFrame({"flag_category": False, "predicted_category": y, "category_conf": 0.0},
                         index=df.index)
    if len(df) < 10 or counts.size < 2 or counts.min() < 2:
        return empty
    k = int(min(5, counts.min()))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    proba = cross_val_predict(_category_model(), text, y, cv=cv, method="predict_proba")
    classes = np.array(sorted(counts.index))
    pred = classes[proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    flag = (pred != y.to_numpy()) & (conf >= min_conf)
    return pd.DataFrame({"flag_category": flag, "predicted_category": pred,
                         "category_conf": conf.round(3)}, index=df.index)


# ------------------------------------------------------------------- driver
def analyze(df: pd.DataFrame, **kw) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    df["vendor_key"] = canonical_vendors(df["vendor"].tolist())
    dup_of = detect_duplicates(df)
    amounts = detect_amount_outliers(df, **{k: v for k, v in kw.items()
                                            if k in ("contamination", "z_thresh", "tax_tol")})
    cats = detect_misclassified(df, **{k: v for k, v in kw.items() if k == "min_conf"})
    out = pd.concat([df, amounts, cats], axis=1)
    out["flag_duplicate"] = dup_of != ""
    out["duplicate_of"] = dup_of
    out["flag_any"] = out[["flag_duplicate", "flag_amount", "flag_category"]].any(axis=1)
    return out
