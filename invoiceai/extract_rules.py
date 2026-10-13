"""Rule-based baseline: regex heuristics over OCR text.

This is the "no model" reference point. It is deliberately simple (label-driven
regexes, first line = vendor) so the LLM approaches have something honest to beat.
"""
from __future__ import annotations

import re

from .schema import parse_amount, parse_currency, parse_date

_AMOUNT = re.compile(r"\d[\d,  ]*[.,]\d{2}(?!\d)")
_NUMBER_LABEL = re.compile(
    r"(?:invoice\s*(?:no|number|#)|facture\s*(?:n[°o]|no|num[ée]ro)|invoice\s*#)\s*[.:;#°]?\s*"
    r"([A-Z0-9][A-Z0-9\-/]{3,})", re.I)
_TITLE_WORDS = re.compile(r"\b(invoice|facture)\b", re.I)
_CUR = re.compile(r"\b(MAD|EUR|DHS?)\b|€")


def _last_amount(line: str) -> float | None:
    found = _AMOUNT.findall(line)
    return parse_amount(found[-1]) if found else None


def _find_vendor(lines: list[str]) -> str | None:
    for ln in lines:
        cleaned = _TITLE_WORDS.sub("", ln).strip(" -:|")
        if len(cleaned) >= 3:
            return cleaned
    return None


def _find_number(text: str) -> str | None:
    m = _NUMBER_LABEL.search(text)
    return m.group(1).strip() if m else None


def _find_date(lines: list[str]) -> str | None:
    for ln in lines:
        if re.search(r"\b(date|issued)\b", ln, re.I):
            d = parse_date(ln)
            if d:
                return d
    for ln in lines:
        d = parse_date(ln)
        if d:
            return d
    return None


def _find_total(lines: list[str]) -> tuple[float | None, str | None]:
    best = None
    for ln in lines:
        low = ln.lower()
        if re.search(r"sub\s*-?total|total\s*ht|\bht\b", low):
            continue
        if re.search(r"total\s*ttc|amount\s*due|\btotal\b", low):
            amt = _last_amount(ln)
            if amt is not None:
                best = (amt, ln)
    return best if best else (None, None)


def _find_tax(lines: list[str]) -> float | None:
    """Last VAT/TVA/Tax line with an amount (summary lines sit below the line items,
    so a first-match rule would be fooled by an item such as "Tax advisory session")."""
    found = None
    for ln in lines:
        if re.search(r"\b(vat|tva|tax)\b", ln, re.I):
            amt = _last_amount(ln)
            if amt is not None:
                found = amt
    return found


def _find_currency(total_line: str | None, text: str) -> str | None:
    for src in (total_line or "", text):
        m = _CUR.search(src)
        if m:
            return parse_currency(m.group(0))
    return None


_ROW_STOP = re.compile(r"^\s*(sub\s*-?total|total\b|amount\s*due|(?:tax|tva|vat)\s*[(\d:])", re.I)
_ROW_DESC = re.compile(r"^(.*?)\s+\d{1,3}\s*(?:x\s*)?\d[\d  .,]*\d")


def _find_description(lines: list[str]) -> str | None:
    start = None
    for i, ln in enumerate(lines):
        if re.search(r"\b(description|d[ée]signation)\b", ln, re.I):
            start = i + 1
            break
    if start is None:
        # compact layout: rows follow the customer line
        for i, ln in enumerate(lines):
            if re.search(r"\b(customer|client)\b", ln, re.I):
                start = i + 1
                break
    if start is None:
        return None
    descs = []
    for ln in lines[start:]:
        if _ROW_STOP.match(ln):
            break
        m = _ROW_DESC.match(ln)
        if m and len(m.group(1)) >= 3:
            descs.append(m.group(1).strip())
    return "; ".join(descs) or None


def extract(ocr_text: str) -> dict:
    lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]
    total, total_line = _find_total(lines)
    return {
        "invoice_number": _find_number(ocr_text),
        "date": _find_date(lines),
        "vendor": _find_vendor(lines),
        "currency": _find_currency(total_line, ocr_text),
        "tax_amount": _find_tax(lines),
        "total_amount": total,
        "description": _find_description(lines),
    }
