import pytest

from invoiceai.data import load_truth
from invoiceai.evaluate import evaluate_extraction
from invoiceai.extract_rules import extract
from tests.conftest import needs_tesseract

FR = """Eau et Lumiere SA FACTURE

27 Rue Ibn Sina, Casablanca

Facture N° ELS-2025-72193

Date de facture : 23/09/2025

Client : Northwind Labs SARL

Désignation Qté P.U. Montant

Network connection fee 2 503,53 1007,06

Water consumption 2 763,90 1527,80
Total HT 2 534,86 DH
TVA 20 % 506,97 DH
Total TTC 3 041,83 DH
"""

EN = """Atlas Cloud Hosting INVOICE
12 Avenue Hassan II, Casablanca
Invoice No: ACH-2025-61965
Date: 03 Oct 2025
Bill to: Northwind Labs SARL
Description Qty Unit price Amount
Object storage 5 TB 4 163.21 652.84
Managed database plan 2 474.48 948.96
Subtotal 1,601.80 MAD
VAT (20%) 320.36 MAD
Total 1,922.16 MAD
"""

COMPACT = """Nimbus Data Services
8 Boulevard Zerktouni, Lyon
Invoice # NDS-2026-00412     Issued 2026-02-11
Customer: Northwind Labs SARL
API platform license  2 x 300.00 = 600.00
Premium support plan  1 x 150.50 = 150.50
Tax (20%): 150.10 EUR
Amount due: 900.60 EUR
"""


def test_french_layout():
    p = extract(FR)
    assert p["vendor"] == "Eau et Lumiere SA"
    assert p["invoice_number"] == "ELS-2025-72193"
    assert p["date"] == "2025-09-23"
    assert p["total_amount"] == 3041.83
    assert p["tax_amount"] == 506.97
    assert p["currency"] == "MAD"
    assert p["description"] == "Network connection fee; Water consumption"


def test_english_layout_keeps_digits_inside_descriptions():
    p = extract(EN)
    assert p["total_amount"] == 1922.16 and p["tax_amount"] == 320.36
    assert p["date"] == "2025-10-03"
    assert p["description"] == "Object storage 5 TB; Managed database plan"


def test_compact_layout():
    p = extract(COMPACT)
    assert p["invoice_number"] == "NDS-2026-00412"
    assert p["date"] == "2026-02-11"
    assert p["total_amount"] == 900.6
    assert p["currency"] == "EUR"


def test_item_named_tax_advisory_is_not_mistaken_for_the_vat_line():
    text = COMPACT.replace("API platform license  2 x 300.00 = 600.00",
                           "Tax advisory session  2 x 300.00 = 600.00")
    p = extract(text)
    assert p["tax_amount"] == 150.10
    assert p["description"].startswith("Tax advisory session")


def test_ocr_misreading_colon_as_semicolon_in_the_label():
    assert extract("Atlas Cloud Hosting\nInvoice No; ACH-2025-61965\n")["invoice_number"] == "ACH-2025-61965"


def test_empty_text_gives_all_none():
    assert all(v is None for v in extract("").values())


@pytest.mark.ocr
@needs_tesseract
def test_end_to_end_on_generated_images(small_data):
    """Real OCR on a handful of generated images; rules should clearly beat chance."""
    from invoiceai.ocr import ocr_image

    out, _ = small_data
    rows = load_truth(out)[:6]
    preds = [{"id": r["id"], "pred": extract(ocr_image(out / r["file"]))} for r in rows]
    m = evaluate_extraction(rows, preds)
    assert m["macro_field_accuracy"] >= 0.6
