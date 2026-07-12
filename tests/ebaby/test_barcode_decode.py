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


def test_decode_barcodes_treats_none_as_a_miss_not_a_crash():
    """An unreadable/corrupt crop reaches decode_barcodes as None; it must be
    a clean miss so one bad file doesn't abort the whole barcode batch (which
    would lose every result decoded before it)."""
    assert bd.decode_barcodes(None) == []


def test_normalize_barcode_strips_leading_zero_for_upca_via_ean13():
    assert bd._normalize_barcode("0883316276402", "EAN13") == "883316276402"


def test_normalize_barcode_leaves_other_symbologies_alone():
    assert bd._normalize_barcode("012345678905", "UPCA") == "012345678905"


def test_candidates_never_upscale_past_the_view_cap():
    """A 2400px full-frame fallback x3 = 7200px through 7 contrast variants
    is the 'minutes per missed barcode' path — big frames must skip x2/x3."""
    import numpy as np
    from ebaby.stages import barcode_decode
    big = np.full((2400, 1600), 128, dtype=np.uint8)
    assert max(max(v.shape[:2]) for v in barcode_decode._candidates(big)) <= \
        barcode_decode._MAX_VIEW_DIM
    small = np.full((200, 400), 128, dtype=np.uint8)  # tight crop: x3 kept
    assert max(max(v.shape[:2]) for v in barcode_decode._candidates(small)) == 1200
