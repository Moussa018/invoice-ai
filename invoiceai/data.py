"""Loaders for the generated dataset layout."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def load_truth(data_dir: str | Path, split: str = "all") -> list[dict]:
    rows = []
    with open(Path(data_dir) / "truth.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if split != "all":
        rows = [r for r in rows if r["split"] == split]
    return rows


def load_ledger(data_dir: str | Path) -> dict[str, str]:
    with open(Path(data_dir) / "ledger.csv", newline="", encoding="utf-8") as f:
        return {r["id"]: r["booked_category"] for r in csv.DictReader(f)}


def load_labels(data_dir: str | Path) -> dict[str, dict]:
    with open(Path(data_dir) / "labels.csv", newline="", encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def ocr_path(data_dir: str | Path, inv_id: str) -> Path:
    return Path(data_dir) / "ocr" / f"{inv_id}.txt"


def load_ocr_text(data_dir: str | Path, inv_id: str) -> str:
    p = ocr_path(data_dir, inv_id)
    return p.read_text(encoding="utf-8") if p.exists() else ""


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
