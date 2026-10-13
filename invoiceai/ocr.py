"""OCR step: invoice image -> text, cached next to the dataset."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from .data import load_truth, ocr_path

# Tesseract spawns OpenMP threads; one per process avoids oversubscription
# when several images are processed in parallel.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")


def ocr_image(path: str | Path, psm: int = 6) -> str:
    import pytesseract

    with Image.open(path) as im:
        return pytesseract.image_to_string(im, config=f"--psm {psm}")


def ocr_all(data_dir: str | Path, workers: int | None = None, force: bool = False,
            psm: int = 6) -> int:
    data_dir = Path(data_dir)
    (data_dir / "ocr").mkdir(exist_ok=True)
    todo = []
    for r in load_truth(data_dir):
        out = ocr_path(data_dir, r["id"])
        if force or not out.exists():
            todo.append((data_dir / r["file"], out))

    def run(pair):
        img, out = pair
        out.write_text(ocr_image(img, psm), encoding="utf-8")

    with ThreadPoolExecutor(max_workers=workers or os.cpu_count() or 1) as ex:
        list(ex.map(run, todo))
    return len(todo)
