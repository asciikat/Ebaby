import cv2
import numpy as np
import pytest

from ebaby.stages import barcode_decode as bd

try:
    import barcode as _pybarcode  # python-barcode, used only to synthesize a test fixture
    from barcode.writer import ImageWriter
    _HAVE_PYBARCODE = True
except ImportError:
    _HAVE_PYBARCODE = False


@pytest.mark.skipif(not _HAVE_PYBARCODE, reason="python-barcode not installed; "
                     "install with `pip install python-barcode` to generate a "
                     "synthetic EAN-13 fixture for this test")
def test_decode_barcodes_reads_a_generated_ean13(tmp_path):
    ean = _pybarcode.get("ean13", "400638133393", writer=ImageWriter())
    png_path = ean.save(str(tmp_path / "code"))
    img = cv2.imread(png_path)
    result = bd.decode_barcodes(img)
    assert result and result[0].isdigit()


def test_decode_barcodes_returns_empty_list_on_blank_image():
    blank = np.full((200, 400, 3), 255, dtype=np.uint8)
    assert bd.decode_barcodes(blank) == []


def test_normalize_barcode_strips_leading_zero_for_upca_via_ean13():
    assert bd._normalize_barcode("0883316276402", "EAN13") == "883316276402"


def test_normalize_barcode_leaves_other_symbologies_alone():
    assert bd._normalize_barcode("012345678905", "UPCA") == "012345678905"
