import numpy as np
import pytest

from redboxflip import barcode


def _make_ean13(tmp_path, digits12="590123412345"):
    bc = pytest.importorskip("barcode")
    from barcode.writer import ImageWriter
    ean = bc.get("ean13", digits12, writer=ImageWriter())
    path = tmp_path / "code"
    saved = ean.save(str(path))      # returns full path incl. extension
    return saved, ean.get_fullcode()


def test_available_decoders_lists_something():
    decoders = barcode.available_decoders()
    if not decoders:
        pytest.skip("no barcode decoder installed in this environment")
    assert isinstance(decoders, list)


def test_decode_reads_a_generated_ean13(tmp_path):
    if not barcode.available_decoders():
        pytest.skip("no barcode decoder installed")
    import cv2
    saved, fullcode = _make_ean13(tmp_path)
    bgr = cv2.imread(saved)
    digits, method, rot = barcode.decode(bgr)
    assert digits == fullcode


def test_decode_reads_rotated_barcode(tmp_path):
    if not barcode.available_decoders():
        pytest.skip("no barcode decoder installed")
    import cv2
    saved, fullcode = _make_ean13(tmp_path)
    bgr = cv2.imread(saved)
    rotated = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    digits, method, rot = barcode.decode(rotated)
    assert digits == fullcode


def test_decode_returns_none_on_blank():
    if not barcode.available_decoders():
        pytest.skip("no barcode decoder installed")
    blank = np.full((200, 300, 3), 255, np.uint8)
    digits, method, rot = barcode.decode(blank)
    assert digits is None


def test_decode_deadline_bails_early(monkeypatch):
    # Deterministic: decoders are slow and never succeed. Without the
    # deadline the sweep would grind through 4 rotations x 3 scales x
    # 5 preprocs x 3 decoders x 0.2s = ~36s (measured ~114s worst case on
    # a real photo); the 1s deadline must abort it almost immediately.
    import time
    def slow_never(img):
        time.sleep(0.2)
        return None
    monkeypatch.setattr(barcode, "_try_pyzbar", slow_never)
    monkeypatch.setattr(barcode, "_try_zxing", slow_never)
    monkeypatch.setattr(barcode, "_try_opencv", slow_never)
    img = np.full((400, 300, 3), 128, np.uint8)
    t = time.time()
    digits, method, rot = barcode.decode(img, deadline_s=1.0)
    elapsed = time.time() - t
    assert digits is None
    assert elapsed < 5


def test_decode_deadline_still_finds_fast_barcodes(tmp_path):
    if not barcode.available_decoders():
        pytest.skip("no barcode decoder installed")
    import cv2
    saved, fullcode = _make_ean13(tmp_path)
    bgr = cv2.imread(saved)
    digits, method, rot = barcode.decode(bgr, deadline_s=15.0)
    assert digits == fullcode
