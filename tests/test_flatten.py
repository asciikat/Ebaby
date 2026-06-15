import numpy as np
from dvdflip.loader import load_bgr
from dvdflip.flatten import detect_a4, warp_to_a4

def test_detect_and_warp_to_a4(sample_front):
    bgr = load_bgr(sample_front)
    quad, conf = detect_a4(bgr)
    assert quad is not None
    assert conf > 0.45
    flat = warp_to_a4(bgr, quad)
    h, w = flat.shape[:2]
    long_side, short_side = max(h, w), min(h, w)
    assert abs(long_side / short_side - 297 / 210) < 0.02
    assert long_side == 2970 and short_side == 2100
