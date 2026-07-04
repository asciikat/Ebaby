import cv2
import numpy as np
import pytest

from ebaby.stages import color


def _white_bordered_frame():
    """A 40x40 BGR frame: white border (color cast), grey square in the
    middle standing in for the DVD case content."""
    frame = np.full((40, 40, 3), (200, 150, 150), dtype=np.uint8)  # blue-cast white
    frame[10:30, 10:30] = (80, 80, 80)  # neutral grey "case"
    return frame


def test_get_surrounding_white_mask_ignores_interior_and_keeps_border():
    frame = _white_bordered_frame()
    mask = color.get_surrounding_white_mask(frame)
    assert mask[0, 0] == 255       # border pixel included
    assert mask[20, 20] == 0       # interior "case" pixel excluded


def test_get_white_balance_gains_neutralizes_border_cast():
    frame = _white_bordered_frame()
    gains = color.get_white_balance_gains(frame)
    corrected = color.apply_white_balance(frame, gains)
    b, g, r = corrected[0, 0].astype(int)
    assert abs(int(b) - int(g)) <= 2
    assert abs(int(g) - int(r)) <= 2


def test_get_white_balance_gains_all_black_frame_is_safe_identity():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    gains = color.get_white_balance_gains(frame)
    assert gains == (1.0, 1.0, 1.0)


def test_enhance_image_is_a_noop_at_default_neutral_settings():
    frame = _white_bordered_frame()
    out = color.enhance_image(frame, contrast=0.0, saturation=1.0, sharpen=1.0)
    assert np.array_equal(out, frame)


def test_enhance_image_increases_saturation():
    frame = _white_bordered_frame()
    out = color.enhance_image(frame, contrast=1.0, saturation=1.5, sharpen=1.0)
    hsv_before = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_after = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    assert hsv_after[0, 0, 1] >= hsv_before[0, 0, 1]


def test_color_correct_file_writes_a_png(sample_front, tmp_path):
    out_path = tmp_path / "01_front.png"
    color.color_correct_file(sample_front, out_path)
    assert out_path.exists()
    img = cv2.imread(str(out_path))
    assert img is not None and img.shape[2] == 3
