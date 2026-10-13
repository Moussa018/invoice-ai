"""Synthetic invoice generator.

Creates noisy PNG invoices in three layouts (EN/FR labels, three number formats,
three date formats) plus:

* ``truth.jsonl``  - ground truth for extraction
* ``ledger.csv``   - the category each invoice was booked under (model input)
* ``labels.csv``   - injected anomaly labels (evaluation only)

Anomalies injected: duplicates (exact and near), amount outliers and
misclassified cost categories.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

CATEGORIES = [
    "Software & Cloud", "Travel", "Office Supplies", "Consulting",
    "Equipment", "Utilities", "Marketing",
]
BILL_TO = "Northwind Labs SARL"
W, H = 1000, 1300


@dataclass(frozen=True)
class Vendor:
    name: str
    prefix: str
    category: str
    template: int      # 1 = English table, 2 = French table, 3 = compact
    currency: str      # MAD or EUR
    vat_rate: float
    typical: float     # typical subtotal in the vendor's currency
    items: tuple
    city: str


VENDORS = [
    Vendor("Atlas Cloud Hosting", "ACH", "Software & Cloud", 1, "MAD", 0.20, 2800,
           ("Cloud server subscription", "Managed database plan", "Object storage 5 TB"), "Casablanca"),
    Vendor("Nimbus Data Services", "NDS", "Software & Cloud", 3, "EUR", 0.20, 900,
           ("API platform license", "Monitoring seats (annual)", "Premium support plan"), "Lyon"),
    Vendor("CodeForge Licenses", "CFL", "Software & Cloud", 2, "MAD", 0.20, 1900,
           ("IDE team licenses", "CI minutes bundle", "Security scanner seats"), "Rabat"),
    Vendor("Maghreb Air Voyages", "MAV", "Travel", 2, "MAD", 0.10, 3600,
           ("Flight Casablanca-Paris", "Flight Rabat-Lyon", "Baggage supplement"), "Casablanca"),
    Vendor("Riad Hotels Group", "RHG", "Travel", 1, "MAD", 0.10, 2400,
           ("Hotel accommodation 2 nights", "Conference room rental", "Breakfast package"), "Marrakech"),
    Vendor("Rapide Rail Tickets", "RRT", "Travel", 3, "EUR", 0.10, 120,
           ("Train ticket Paris-Lyon", "Train ticket Lyon-Marseille", "Seat reservation fee"), "Paris"),
    Vendor("Papeterie Centrale", "PCE", "Office Supplies", 2, "MAD", 0.20, 450,
           ("A4 paper reams", "Toner cartridges", "Notebooks and pens set"), "Fes"),
    Vendor("OfficeMart Maroc", "OMM", "Office Supplies", 1, "MAD", 0.20, 1500,
           ("Ergonomic office chairs", "Whiteboard and markers", "Desk organizers"), "Tangier"),
    Vendor("PrintPoint", "PPT", "Office Supplies", 3, "EUR", 0.20, 80,
           ("Business cards 500 pcs", "Poster printing A1", "Binding and lamination"), "Nantes"),
    Vendor("Sahara Advisory", "SAH", "Consulting", 1, "MAD", 0.20, 9500,
           ("Strategy consulting days", "Market study report", "Workshop facilitation"), "Casablanca"),
    Vendor("Delta Legal Partners", "DLP", "Consulting", 2, "EUR", 0.20, 2200,
           ("Legal review of contracts", "IP filing assistance", "Regulatory compliance memo"), "Paris"),
    Vendor("Optima Audit Firm", "OAF", "Consulting", 3, "MAD", 0.20, 11000,
           ("Annual audit fee", "Financial statements review", "Tax advisory session"), "Rabat"),
    Vendor("TechParts Casablanca", "TPC", "Equipment", 3, "MAD", 0.20, 7500,
           ("Laptop 14 inch", "Network switch 24 ports", "External SSD 2 TB"), "Casablanca"),
    Vendor("Hardware Hub", "HWH", "Equipment", 1, "EUR", 0.20, 3800,
           ("Workstation GPU server", "Server rack 42U", "UPS battery unit"), "Toulouse"),
    Vendor("LabTools Maroc", "LTM", "Equipment", 2, "MAD", 0.20, 2900,
           ("Oscilloscope rental", "Calibration kit", "Soldering station"), "Agadir"),
    Vendor("Eau et Lumiere SA", "ELS", "Utilities", 2, "MAD", 0.20, 2300,
           ("Electricity consumption", "Water consumption", "Network connection fee"), "Casablanca"),
    Vendor("TelecomNet Maroc", "TNM", "Utilities", 1, "MAD", 0.20, 900,
           ("Fiber internet 1 Gbps", "Mobile lines bundle", "Landline subscription"), "Rabat"),
    Vendor("Pixel Studio Agency", "PSA", "Marketing", 2, "MAD", 0.20, 5200,
           ("Brand identity design", "Social media campaign", "Website banner set"), "Casablanca"),
    Vendor("AdWave Media", "AWM", "Marketing", 1, "EUR", 0.20, 2400,
           ("Online ad campaign", "Video ad production", "Sponsored newsletter slot"), "Lille"),
]

_MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_STREETS = ["12 Avenue Hassan II", "45 Rue des Entrepreneurs", "8 Boulevard Zerktouni",
            "27 Rue Ibn Sina", "3 Place de la Gare", "91 Avenue des FAR"]

_FONT_DIRS = ["/usr/share/fonts/truetype/dejavu/", "/usr/share/fonts/dejavu/",
              "/usr/share/fonts/truetype/liberation/"]
_FONT_FILES = {
    "regular": ["DejaVuSans.ttf", "LiberationSans-Regular.ttf"],
    "bold": ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"],
    "mono": ["DejaVuSansMono.ttf", "LiberationMono-Regular.ttf"],
}


def _font(kind: str, size: int):
    for d in _FONT_DIRS:
        for f in _FONT_FILES[kind]:
            p = Path(d) / f
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size)


# ---------------------------------------------------------------- formatting
def fmt_amount(x: float, style: str) -> str:
    if style == "us":
        return f"{x:,.2f}"
    if style == "fr":
        return f"{x:,.2f}".replace(",", " ").replace(".", ",")
    return f"{x:.2f}"


def fmt_date(d: date, template: int) -> str:
    if template == 1:
        return f"{d.day:02d} {_MONTH_ABBR[d.month - 1]} {d.year}"
    if template == 2:
        return f"{d.day:02d}/{d.month:02d}/{d.year}"
    return d.isoformat()


_CUR_LABEL = {
    "MAD": {1: "MAD", 2: "DH", 3: "MAD"},
    "EUR": {1: "EUR", 2: "€", 3: "EUR"},
}
_STYLE = {1: "us", 2: "fr", 3: "plain"}


# ------------------------------------------------------------------- records
def _make_lines(v: Vendor, subtotal_target: float, rng: random.Random) -> list[dict]:
    k = rng.randint(1, min(3, len(v.items)))
    descs = rng.sample(list(v.items), k)
    weights = [rng.uniform(0.5, 1.5) for _ in range(k)]
    tw = sum(weights)
    lines = []
    for d, w in zip(descs, weights):
        qty = rng.randint(1, 4)
        unit = max(round(subtotal_target * w / tw / qty, 2), 1.0)
        lines.append({"description": d, "qty": qty, "unit_price": unit,
                      "amount": round(qty * unit, 2)})
    return lines


def _finish_amounts(rec: dict, v: Vendor) -> None:
    sub = round(sum(li["amount"] for li in rec["line_items"]), 2)
    tax = round(sub * v.vat_rate, 2)
    rec["subtotal"] = sub
    rec["tax_amount"] = tax
    rec["total_amount"] = round(sub + tax, 2)
    rec["description"] = "; ".join(li["description"] for li in rec["line_items"])


def _make_record(idx: int, v: Vendor, rng: random.Random, used: set, scale: float = 1.0) -> dict:
    d = date(2025, 1, 1) + timedelta(days=rng.randint(0, 637))
    while True:
        num = f"{v.prefix}-{d.year}-{rng.randint(1, 99999):05d}"
        if num not in used:
            used.add(num)
            break
    target = v.typical * float(np.exp(rng.gauss(0, 0.30))) * scale
    rec = {
        "id": f"inv_{idx:04d}",
        "file": f"images/inv_{idx:04d}.png",
        "vendor": v.name,
        "invoice_number": num,
        "date": d.isoformat(),
        "currency": v.currency,
        "vat_rate": v.vat_rate,
        "line_items": _make_lines(v, target, rng),
        "true_category": v.category,
        "booked_category": v.category,
        "is_duplicate": False,
        "duplicate_of": "",
        "dup_kind": "",
        "is_amount_outlier": scale != 1.0,
        "is_misclassified": False,
    }
    _finish_amounts(rec, v)
    return rec


# ----------------------------------------------------------------- rendering
def _vendor_by_name(name: str) -> Vendor:
    return next(v for v in VENDORS if v.name == name)


def render_invoice(rec: dict, rng: random.Random, noise: float = 1.0) -> Image.Image:
    v = _vendor_by_name(rec["vendor"])
    t = v.template
    style = _STYLE[t]
    cur = _CUR_LABEL[v.currency][t]
    d = date.fromisoformat(rec["date"])
    img = Image.new("L", (W, H), 255)
    dr = ImageDraw.Draw(img)
    h1, m, s = _font("bold", 34), _font("regular", 24), _font("regular", 20)
    mono = _font("mono", 22)
    street = _STREETS[sum(map(ord, v.name)) % len(_STREETS)]
    pct = int(round(v.vat_rate * 100))

    dr.text((60, 60), v.name, font=h1, fill=0)
    dr.text((60, 108), f"{street}, {v.city}", font=s, fill=70)

    if t in (1, 2):
        if t == 1:
            title, num_l, date_l = "INVOICE", "Invoice No:", "Date:"
            cols = ("Description", "Qty", "Unit price", "Amount")
            bill = f"Bill to: {BILL_TO}"
            sub_l, tax_l, tot_l = "Subtotal", f"VAT ({pct}%)", "Total"
            foot = "Payment terms: 30 days. Thank you for your business."
        else:
            title, num_l, date_l = "FACTURE", "Facture N°", "Date de facture :"
            cols = ("Désignation", "Qté", "P.U.", "Montant")
            bill = f"Client : {BILL_TO}"
            sub_l, tax_l, tot_l = "Total HT", f"TVA {pct} %", "Total TTC"
            foot = "Conditions de paiement : 30 jours."
        dr.text((940, 60), title, font=h1, fill=0, anchor="ra")
        dr.text((60, 200), f"{num_l} {rec['invoice_number']}", font=m, fill=0)
        dr.text((60, 240), f"{date_l} {fmt_date(d, t)}", font=m, fill=0)
        dr.text((60, 280), bill, font=m, fill=0)
        y = 380
        dr.line((60, y - 8, 940, y - 8), fill=0, width=2)
        dr.text((60, y), cols[0], font=_font("bold", 22), fill=0)
        for x, c in ((640, cols[1]), (790, cols[2]), (940, cols[3])):
            dr.text((x, y), c, font=_font("bold", 22), fill=0, anchor="ra")
        y += 40
        dr.line((60, y - 4, 940, y - 4), fill=0, width=1)
        for li in rec["line_items"]:
            dr.text((60, y), li["description"], font=m, fill=0)
            dr.text((640, y), str(li["qty"]), font=m, fill=0, anchor="ra")
            dr.text((790, y), fmt_amount(li["unit_price"], style), font=m, fill=0, anchor="ra")
            dr.text((940, y), fmt_amount(li["amount"], style), font=m, fill=0, anchor="ra")
            y += 46
        dr.line((60, y, 940, y), fill=0, width=1)
        y += 30
        for label, val, bold in ((sub_l, rec["subtotal"], False), (tax_l, rec["tax_amount"], False),
                                 (tot_l, rec["total_amount"], True)):
            f = _font("bold", 26) if bold else m
            dr.text((540, y), label, font=f, fill=0)
            dr.text((940, y), f"{fmt_amount(val, style)} {cur}", font=f, fill=0, anchor="ra")
            y += 44
        dr.text((60, H - 120), foot, font=s, fill=70)
    else:
        dr.text((60, 190), f"Invoice # {rec['invoice_number']}     Issued {fmt_date(d, t)}", font=m, fill=0)
        dr.text((60, 232), f"Customer: {BILL_TO}", font=m, fill=0)
        y = 320
        dr.line((60, y - 10, 940, y - 10), fill=0, width=2)
        for li in rec["line_items"]:
            row = (f"{li['description']}  {li['qty']} x {fmt_amount(li['unit_price'], style)}"
                   f" = {fmt_amount(li['amount'], style)}")
            dr.text((60, y), row, font=mono, fill=0)
            y += 42
        dr.line((60, y, 940, y), fill=0, width=1)
        y += 30
        dr.text((60, y), f"Tax ({pct}%): {fmt_amount(rec['tax_amount'], style)} {cur}", font=m, fill=0)
        y += 44
        dr.text((60, y), f"Amount due: {fmt_amount(rec['total_amount'], style)} {cur}",
                font=_font("bold", 26), fill=0)
        dr.text((60, H - 120), "Net 30 days.", font=s, fill=70)

    return add_scan_noise(img, rng, noise)


def add_scan_noise(img: Image.Image, rng: random.Random, level: float) -> Image.Image:
    """Rotate, blur and add grain to mimic a mediocre scan."""
    if level <= 0:
        return img
    img = img.rotate(rng.uniform(-1.2, 1.2) * level, resample=Image.BICUBIC, fillcolor=255)
    blur = rng.uniform(0.0, 0.9) * level
    if blur > 0.05:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(img, dtype=np.float32)
    nrng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    arr = arr + nrng.normal(0, 4 + 10 * level * rng.random(), arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ------------------------------------------------------------------- dataset
def generate_dataset(out_dir: str | Path, n: int = 400, seed: int = 7, noise: float = 1.0,
                     dup_rate: float = 0.04, outlier_rate: float = 0.04,
                     misclass_rate: float = 0.05, train_frac: float = 0.6,
                     render: bool = True) -> list[dict]:
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    order = list(range(n))
    rng.shuffle(order)
    n_out, n_mis, n_dup = round(n * outlier_rate), round(n * misclass_rate), round(n * dup_rate)
    outlier_idx = set(order[:n_out])
    mis_idx = set(order[n_out:n_out + n_mis])
    dup_src = order[n_out + n_mis:n_out + n_mis + n_dup]

    used: set = set()
    records: list[dict] = []
    for i in range(n):
        v = rng.choice(VENDORS)
        scale = rng.uniform(3.5, 12.0) if i in outlier_idx else 1.0
        rec = _make_record(i + 1, v, rng, used, scale)
        if i in mis_idx:
            rec["booked_category"] = rng.choice([c for c in CATEGORIES if c != v.category])
            rec["is_misclassified"] = True
        rec["split"] = "train" if rng.random() < train_frac else "test"
        records.append(rec)

    for j, src_i in enumerate(dup_src):
        src = records[src_i]
        dup = json.loads(json.dumps(src))
        dup["id"] = f"inv_{n + j + 1:04d}"
        dup["file"] = f"images/{dup['id']}.png"
        dup["is_duplicate"] = True
        dup["duplicate_of"] = src["id"]
        if rng.random() < 0.6:
            dup["dup_kind"] = "exact"
        else:
            dup["dup_kind"] = "near"
            last = int(src["invoice_number"][-1])
            dup["invoice_number"] = src["invoice_number"][:-1] + str((last + rng.randint(1, 9)) % 10)
        records.append(dup)

    if render:
        for rec in records:
            render_invoice(rec, rng, noise).save(out / rec["file"])

    _write_outputs(out, records)
    return records


def _write_outputs(out: Path, records: list[dict]) -> None:
    keep = ["id", "file", "split", "vendor", "invoice_number", "date", "currency", "tax_amount",
            "total_amount", "description", "subtotal", "vat_rate", "line_items", "true_category"]
    with open(out / "truth.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({k: r[k] for k in keep}, ensure_ascii=False) + "\n")
    with open(out / "ledger.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "booked_category"])
        for r in records:
            w.writerow([r["id"], r["booked_category"]])
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "true_category", "booked_category", "is_duplicate", "duplicate_of",
                    "dup_kind", "is_amount_outlier", "is_misclassified"])
        for r in records:
            w.writerow([r["id"], r["split"], r["true_category"], r["booked_category"],
                        int(r["is_duplicate"]), r["duplicate_of"], r["dup_kind"],
                        int(r["is_amount_outlier"]), int(r["is_misclassified"])])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--noise", type=float, default=1.0)
    a = p.parse_args(argv)
    recs = generate_dataset(a.out, a.n, a.seed, a.noise)
    print(f"wrote {len(recs)} invoices to {a.out}")


if __name__ == "__main__":
    main()
