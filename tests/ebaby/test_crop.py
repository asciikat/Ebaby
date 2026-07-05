import cv2
import numpy as np

from ebaby.stages import crop


def _case_on_white(w=400, h=560, case_w=300, case_h=420, angle=0):
    """A near-white frame with a solid grey rotated rectangle standing in for
    a DVD case, big enough to pass scan_case's coverage gate."""
    frame = np.full((h, w, 3), 245, dtype=np.uint8)
    center = (w // 2, h // 2)
    rect = ((center), (case_w, case_h), angle)
    box = cv2.boxPoints(rect).astype(int)
    cv2.fillConvexPoly(frame, box, (90, 90, 90))
    return frame


def test_detect_crop_box_returns_a_quad_for_a_clear_case():
    frame = _case_on_white()
    result = crop.detect_crop_box(frame)
    assert result is not None
    quad, coverage = result
    assert quad.shape == (4, 2)
    assert 0.06 <= coverage <= 0.99


def test_detect_crop_box_returns_none_for_a_blank_frame():
    blank = np.full((400, 300, 3), 245, dtype=np.uint8)
    assert crop.detect_crop_box(blank) is None


def test_warp_to_quad_produces_axis_aligned_output():
    frame = _case_on_white(angle=8)
    quad, _ = crop.detect_crop_box(frame)
    warped = crop.warp_to_quad(frame, quad)
    assert warped is not None
    assert warped.shape[0] > 0 and warped.shape[1] > 0


def test_compose_on_white_produces_a_square_bgr_image():
    rgba = np.zeros((100, 60, 4), dtype=np.uint8)
    rgba[..., 3] = 255  # fully opaque
    out = crop.compose_on_white(rgba, size=500)
    assert out.shape == (500, 500, 3)


def test_compose_on_white_fills_transparent_pixels_white():
    rgba = np.zeros((10, 10, 4), dtype=np.uint8)  # fully transparent
    out = crop.compose_on_white(rgba, size=20)
    assert tuple(out[10, 10]) == (255, 255, 255)


def test_compose_on_white_leaves_a_white_border_margin():
    rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    rgba[..., 3] = 255  # opaque black square
    out = crop.compose_on_white(rgba, size=200, margin=0.10)
    assert tuple(out[3, 3]) == (255, 255, 255)      # border stays white
    assert tuple(out[100, 100]) == (0, 0, 0)        # centre is the image


def test_crop_edges_feather_into_the_white_background():
    frame = _case_on_white()
    quad, _ = crop.detect_crop_box(frame)
    out = crop.crop_and_compose(frame, quad, size=400)
    h, w = out.shape[:2]
    centre = out[h // 2, w // 2].astype(int).mean()
    # walk in from the left edge to the first non-white column: it must be
    # LIGHTER than the case centre (blending toward white), not a hard edge
    x = 0
    while x < w and out[h // 2, x].astype(int).mean() > 250:
        x += 1
    assert x < w, "case never appeared"
    edge = out[h // 2, x].astype(int).mean()
    assert edge > centre + 10
