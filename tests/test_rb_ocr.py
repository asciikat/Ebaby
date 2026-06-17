import numpy as np
import pytest

from redboxflip import ocr


def test_extract_title_returns_none_on_blank_image():
    img = np.full((400, 300, 3), 255, np.uint8)
    result = ocr.extract_title(img)
    assert result is None


def test_extract_title_returns_none_on_dark_image():
    img = np.full((400, 300, 3), 10, np.uint8)
    result = ocr.extract_title(img)
    assert result is None


@pytest.mark.skipif(
    not __import__("pathlib").Path("samples/redbox/front.jpg").exists(),
    reason="front.jpg sample not present",
)
def test_extract_title_does_not_crash_on_real_front():
    """OCR on the full un-cropped photo is best-effort; just ensure no crash."""
    from redboxflip.imaging import load_image_bgr
    bgr = load_image_bgr("samples/redbox/front.jpg")
    result = ocr.extract_title(bgr)
    assert result is None or isinstance(result, str)
