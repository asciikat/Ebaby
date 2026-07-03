import numpy as np

from cropstudio import colorcorrect


def _bordered_image(border_bgr, center_bgr, size=200, border=40):
    """Synthetic photo: a colored 'paper' border around a 'DVD' center block."""
    img = np.full((size, size, 3), border_bgr, np.uint8)
    img[border:-border, border:-border] = center_bgr
    return img


def test_wb_gains_neutralize_a_warm_cast():
    # Paper border has a warm (red-heavy) cast; gains should push R down / B up.
    img = _bordered_image(border_bgr=(180, 200, 230), center_bgr=(30, 30, 30))
    gb, gg, gr = colorcorrect.get_white_balance_gains(img)
    assert gb > 1.0        # blue lifted
    assert gr < 1.0        # red pulled down
    corrected = colorcorrect.apply_white_balance(img, (gb, gg, gr))
    b, g, r = corrected[10, 10].astype(int)   # a border (paper) pixel
    assert abs(b - g) <= 6 and abs(g - r) <= 6   # near-neutral after correction


def test_wb_gains_are_clipped_to_sane_range():
    # Extreme cast must not blow out: gains stay within [0.5, 2.0].
    img = _bordered_image(border_bgr=(40, 120, 250), center_bgr=(0, 0, 0))
    gains = colorcorrect.get_white_balance_gains(img)
    assert all(0.5 <= g <= 2.0 for g in gains)


def test_wb_gains_neutral_image_stays_neutral():
    img = _bordered_image(border_bgr=(210, 210, 210), center_bgr=(50, 50, 50))
    gb, gg, gr = colorcorrect.get_white_balance_gains(img)
    assert abs(gb - 1.0) < 0.02 and abs(gg - 1.0) < 0.02 and abs(gr - 1.0) < 0.02


def test_surrounding_mask_ignores_white_inside_the_cover():
    # White block INSIDE the dark center must not join the border mask.
    img = _bordered_image(border_bgr=(220, 220, 220), center_bgr=(30, 30, 30))
    img[90:110, 90:110] = (250, 250, 250)     # white patch inside the "cover"
    mask = colorcorrect.get_surrounding_white_mask(img)
    assert mask[100, 100] == 0                # inner white patch excluded
    assert mask[10, 10] > 0                   # border paper included


def test_enhance_image_returns_same_shape_uint8():
    img = _bordered_image(border_bgr=(220, 220, 220), center_bgr=(90, 120, 150))
    out = colorcorrect.enhance_image(img)
    assert out.shape == img.shape and out.dtype == np.uint8


import glob
import os

import cv2
import pytest

_REAL_DNGS = sorted(glob.glob(os.path.join("Images in", "*.dng")))


def test_raw_to_jpeg_rejects_non_raw_bytes():
    with pytest.raises(ValueError):
        colorcorrect.raw_to_jpeg(b"definitely not a raw file")


@pytest.mark.skipif(not _REAL_DNGS, reason="no real DNG fixture on this machine")
def test_raw_to_jpeg_decodes_a_real_dng():
    data = open(_REAL_DNGS[0], "rb").read()
    jpeg = colorcorrect.raw_to_jpeg(data)
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    # half_size decode of a 12MP phone DNG is still comfortably above the
    # browser canvas's 1200px working width
    assert max(img.shape[:2]) >= 1200
