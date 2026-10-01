import cv2
import numpy as np

from ebaby.stages import barcode_locate as bl


def _frame_with_barcode_stripe():
    """A 400x300 frame with a wide, high-contrast horizontal stripe region
    standing in for a barcode (alternating dark/light bars)."""
    frame = np.full((300, 400, 3), 220, dtype=np.uint8)
    stripe = frame[130:170, 80:320]
    stripe[:, ::4] = 20  # vertical dark bars every 4px -> strong gradient
    return frame


def test_locate_barcode_crop_finds_the_stripe_region():
    frame = _frame_with_barcode_stripe()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    box = bl.locate_barcode_crop(gray)
    assert box is not None
    x0, y0, x1, y1 = box
    assert x0 < 100 and x1 > 300   # roughly spans the stripe's x-range
    assert y0 < 140 and y1 > 160   # roughly spans the stripe's y-range


def test_locate_barcode_crop_returns_none_on_blank_frame():
    blank = np.full((100, 100), 220, dtype=np.uint8)
    assert bl.locate_barcode_crop(blank) is None


def test_downscale_if_needed_caps_long_edge():
    frame = np.zeros((4000, 2000, 3), dtype=np.uint8)
    out = bl.downscale_if_needed(frame, max_dim=2400)
    assert max(out.shape[:2]) == 2400


def test_downscale_if_needed_leaves_small_frame_alone():
    frame = np.zeros((500, 400, 3), dtype=np.uint8)
    out = bl.downscale_if_needed(frame, max_dim=2400)
    assert out.shape == frame.shape


def test_load_image_uses_direct_read_for_non_raw(tmp_path):
    p = tmp_path / "back.jpg"
    cv2.imwrite(str(p), _frame_with_barcode_stripe())
    img = bl.load_image(p)
    assert img is not None and img.shape[2] == 3
