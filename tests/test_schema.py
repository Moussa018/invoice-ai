import pytest

from invoiceai.schema import norm_key, parse_amount, parse_currency, parse_date


@pytest.mark.parametrize("raw,expected", [
    ("1,234.56", 1234.56),
    ("1 234,56", 1234.56),
    ("1234.56", 1234.56),
    ("3 041,83 DH", 3041.83),
    ("25,200.00 MAD", 25200.0),
    ("1.234.567,89", 1234567.89),
    ("12,50", 12.5),
    ("1,234", 1234.0),
    (900, 900.0),
    ("no digits", None),
    (None, None),
])
def test_parse_amount(raw, expected):
    assert parse_amount(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("2026-03-14", "2026-03-14"),
    ("14/03/2026", "2026-03-14"),
    ("14 Mar 2026", "2026-03-14"),
    ("Date: 03 Oct 2025", "2025-10-03"),
    ("March 14, 2026", "2026-03-14"),
    ("14 février 2026", "2026-02-14"),
    ("31/02/2026", None),
    ("nonsense", None),
])
def test_parse_date(raw, expected):
    assert parse_date(raw) == expected


def test_parse_currency_aliases():
    assert parse_currency("DH") == "MAD"
    assert parse_currency("€") == "EUR"
    assert parse_currency("eur") == "EUR"
    assert parse_currency(None) is None


def test_norm_key_ignores_case_accents_punctuation():
    assert norm_key("Eau et Lumière SA") == norm_key("eau et lumiere  S.A.")
