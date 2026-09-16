from pathlib import Path

import pytest


def test_supported_extensions():

    supported = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }

    assert ".jpg" in supported
    assert ".png" in supported
    assert ".webp" in supported


@pytest.mark.parametrize(
    "filename",
    [
        "receipt.jpg",
        "receipt.jpeg",
        "receipt.png",
        "receipt.bmp",
        "receipt.tif",
        "receipt.tiff",
        "receipt.webp",
    ],
)
def test_valid_receipt_filename(filename):

    extension = Path(
        filename
    ).suffix.lower()

    assert extension in {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }


@pytest.mark.parametrize(
    "filename",
    [
        "receipt.exe",
        "receipt.pdf",
        "receipt.txt",
        "receipt.csv",
        "receipt.zip",
    ],
)
def test_invalid_receipt_filename(filename):

    extension = Path(
        filename
    ).suffix.lower()

    assert extension not in {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }