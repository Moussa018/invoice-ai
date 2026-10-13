"""Shared field definitions and normalisation helpers."""
from __future__ import annotations

import re
import unicodedata
from datetime import date

# Header fields that are scored in the extraction evaluation.
FIELDS = ["invoice_number", "date", "vendor", "currency", "tax_amount", "total_amount"]
AMOUNT_FIELDS = {"tax_amount", "total_amount"}
# Extra free-text field used downstream (category check); scored with token F1.
DESCRIPTION = "description"
ALL_KEYS = FIELDS + [DESCRIPTION]

CURRENCY_ALIASES = {
    "MAD": "MAD", "DH": "MAD", "DHS": "MAD", "MAD.": "MAD",
    "EUR": "EUR", "€": "EUR", "EURO": "EUR", "EUROS": "EUR",
}

_MONTHS = {
    "jan": 1, "january": 1, "janvier": 1,
    "feb": 2, "february": 2, "fev": 2, "fév": 2, "fevrier": 2, "février": 2,
    "mar": 3, "march": 3, "mars": 3,
    "apr": 4, "april": 4, "avr": 4, "avril": 4,
    "may": 5, "mai": 5,
    "jun": 6, "june": 6, "juin": 6,
    "jul": 7, "july": 7, "juil": 7, "juillet": 7,
    "aug": 8, "august": 8, "aou": 8, "aoû": 8, "aout": 8, "août": 8,
    "sep": 9, "sept": 9, "september": 9, "septembre": 9,
    "oct": 10, "october": 10, "octobre": 10,
    "nov": 11, "november": 11, "novembre": 11,
    "dec": 12, "december": 12, "déc": 12, "decembre": 12, "décembre": 12,
}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm_key(s: str | None) -> str:
    """Lower-case alphanumeric key used to compare names and identifiers."""
    if not s:
        return ""
    return re.sub(r"[^a-z0-9]+", "", strip_accents(str(s)).lower())


def tokens(s: str | None) -> list[str]:
    if not s:
        return []
    return re.findall(r"[a-z0-9]+", strip_accents(str(s)).lower())


def parse_amount(value) -> float | None:
    """Parse '1,234.56', '1 234,56', '1234.56 MAD' or a number into a float."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"\d[\d   .,]*", str(value))
    if not m:
        return None
    t = re.sub(r"[   ]", "", m.group(0)).rstrip(".,")
    if "," in t and "." in t:
        if t.rfind(",") > t.rfind("."):
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(",", "")
    elif "," in t:
        head, _, tail = t.rpartition(",")
        if len(tail) in (1, 2):
            t = head.replace(",", "") + "." + tail
        else:
            t = t.replace(",", "")
    elif t.count(".") > 1:
        head, _, tail = t.rpartition(".")
        t = head.replace(".", "") + ("." + tail if len(tail) in (1, 2) else tail)
    try:
        return float(t)
    except ValueError:
        return None


def _safe_date(y: int, m: int, d: int) -> str | None:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def parse_date(value) -> str | None:
    """Return an ISO date (YYYY-MM-DD) from the formats used on the invoices."""
    if value is None:
        return None
    s = str(value).strip()
    m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if m:
        return _safe_date(int(m[1]), int(m[2]), int(m[3]))
    m = re.search(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})", s)
    if m:
        return _safe_date(int(m[3]), int(m[2]), int(m[1]))
    m = re.search(r"(\d{1,2})\s+([A-Za-zéèûôÀ-ÿ]{3,9})\.?,?\s+(\d{4})", s)
    if m:
        mon = _MONTHS.get(strip_accents(m[2]).lower()) or _MONTHS.get(m[2].lower())
        if mon:
            return _safe_date(int(m[3]), mon, int(m[1]))
    m = re.search(r"([A-Za-zéèûôÀ-ÿ]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})", s)
    if m:
        mon = _MONTHS.get(strip_accents(m[1]).lower()) or _MONTHS.get(m[1].lower())
        if mon:
            return _safe_date(int(m[3]), mon, int(m[2]))
    return None


def parse_currency(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip().upper()
    if s in CURRENCY_ALIASES:
        return CURRENCY_ALIASES[s]
    if "€" in s:
        return "EUR"
    return s or None
