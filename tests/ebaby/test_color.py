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


def test_lift_whites_brightens_dim_paper_without_blowout():
    img = np.full((80, 80, 3), 60, dtype=np.uint8)   # dark case...
    img[:10, :] = 190                                 # ...dim paper at border
    img[-10:, :] = 190
    img[:, :10] = 190
    img[:, -10:] = 190
    out = color.lift_whites(img)
    assert out[2, 40].mean() > 190          # paper got lifted brighter
    assert out[2, 40].mean() <= 255
    assert out[40, 40].mean() > 60          # cover lifted with it
    assert out[40, 40].mean() < 120         # but nowhere near blown out


def test_color_correct_plain_file_writes_enhanced_png(tmp_path):
    src = tmp_path / "a_front.jpg"
    img = np.full((60, 60, 3), 120, dtype=np.uint8)
    img[:8, :] = 200
    cv2.imwrite(str(src), img)
    out = tmp_path / "a_front.png"
    gains = color.color_correct_plain_file(src, out)
    assert gains and len(gains) == 3          # returns the WB gains used
    assert out.exists()
    assert color.color_correct_plain_file(tmp_path / "missing.jpg", out) is None


def test_vibrance_protects_already_saturated_colours():
    """Vibrance must push a dull pixel more than an already-vivid one — the
    whole point of vibrance over a flat saturation multiply."""
    dull = np.full((1, 1, 3), (150, 130, 120), dtype=np.uint8)   # low sat
    vivid = np.full((1, 1, 3), (240, 20, 20), dtype=np.uint8)     # high sat
    def sat(img):
        return int(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[0, 0, 1])
    dull_gain = sat(color.enhance_image(dull, contrast=0.0, saturation=1.4, sharpen=1.0)) - sat(dull)
    vivid_gain = sat(color.enhance_image(vivid, contrast=0.0, saturation=1.4, sharpen=1.0)) - sat(vivid)
    assert dull_gain > vivid_gain


def test_lift_whites_soft_knee_never_hard_clips_a_bright_frame():
    bright = np.full((40, 40, 3), 250, dtype=np.uint8)  # near-white all over
    bright[:5, :] = 240                                  # paper border
    out = color.lift_whites(bright)
    assert out.max() <= 255 and out[20, 20].mean() > 240  # lifted, not crushed


def test_lift_midtones_brightens_body_pins_endpoints():
    img = np.full((10, 10, 3), 100, dtype=np.uint8)
    img[0, 0] = 0
    img[0, 1] = 255
    out = color.lift_midtones(img, gamma=0.82)
    assert out[5, 5].mean() > 100        # midtones opened up
    assert out[0, 0].mean() == 0         # black pinned
    assert out[0, 1].mean() == 255       # white pinned
