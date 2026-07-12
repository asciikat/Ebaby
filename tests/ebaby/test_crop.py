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


def _open_case_like(w=1000, h=752):
    """Stand-in for an 'inside' shot: two solid disc-like circles on white,
    no single rectangle to fit — the shape that broke every strict tier."""
    frame = np.full((h, w, 3), 245, dtype=np.uint8)
    cv2.circle(frame, (w // 3, h // 2), min(w, h) // 4, (90, 90, 90), -1)
    cv2.circle(frame, (2 * w // 3, h // 2), min(w, h) // 4, (90, 90, 90), -1)
    return frame


def test_loose_tier_crops_an_open_case_that_fails_every_strict_tier():
    """The strict tiers require the shape to fill ~85% of its fitted
    rectangle; two discs with a gap between them never will. Before the
    loose tier existed, this fell all the way back to the raw, uncropped
    photo (background included) — exactly what broke the 'inside' shots in
    the 2026-07-06 batch (Goober/King Kong open-case photos)."""
    frame = _open_case_like()
    result = crop.detect_crop_box(frame)
    assert result is not None
    quad, coverage = result
    # must have trimmed SOME of the surrounding white, not returned the
    # full, uncropped frame
    assert not np.allclose(quad, [[0, 0], [frame.shape[1], 0],
                                  [frame.shape[1], frame.shape[0]], [0, frame.shape[0]]])


def test_loose_tier_still_rejects_a_truly_blank_frame():
    """rembg hallucinates a faint ~14% 'foreground' blob even on a uniform
    blank frame (measured directly) — the loose tier's coverage floor must
    sit above that noise floor or every accidentally-blank photo gets a
    bogus crop instead of a clean 'nothing detected' result."""
    blank = np.full((400, 300, 3), 245, dtype=np.uint8)
    assert crop.detect_crop_box(blank) is None


def test_detect_returns_true_corners_not_a_forced_rectangle():
    """A case shot at an angle is a trapezoid. The detector must return those
    real corners (so warp_to_quad de-keystones it), NOT a minAreaRect that
    forces top==bottom and leaves the perspective distortion baked in."""
    frame = np.full((760, 1000, 3), 245, dtype=np.uint8)
    # a deliberately keystoned quad: top edge wider than bottom
    trapezoid = np.array([[300, 120], [720, 120], [800, 640], [220, 640]], np.int32)
    cv2.fillConvexPoly(frame, trapezoid, (85, 85, 85))
    quad, _ = crop.detect_crop_box(frame)
    ordered = crop._order_pts(quad)
    tl, tr, br, bl = ordered
    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    # the two horizontal edges must differ — a forced rectangle would make
    # them equal (that was the bug: minAreaRect erased the perspective)
    assert abs(top - bottom) > 0.05 * max(top, bottom)


def test_quad_from_contour_finds_four_corners_of_a_convex_blob():
    blob = np.array([[100, 100], [400, 110], [390, 500], [110, 490]], np.int32)
    quad = crop._quad_from_contour(blob.reshape(-1, 1, 2))
    assert quad is not None and quad.shape == (4, 2)
