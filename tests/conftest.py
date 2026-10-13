import shutil

import pytest

from invoiceai.generate import generate_dataset


@pytest.fixture(scope="session")
def small_data(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    records = generate_dataset(out, n=40, seed=11, noise=1.0)
    return out, records


@pytest.fixture
def fresh_data(tmp_path):
    """Function-scoped, image-free dataset for tests that write OCR files into it."""
    records = generate_dataset(tmp_path / "d", n=30, seed=3, render=False)
    return tmp_path / "d", records


needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
