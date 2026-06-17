import numpy as np

from redboxflip import detect
from rb_helpers import make_redbox_image


def test_red_mask_lights_up_the_outline():
    img = make_redbox_image()
    mask = detect.red_mask(img)
    assert mask.dtype == np.uint8
    assert (mask > 0).sum() > 500


def test_find_red_box_locates_the_drawn_box():
    img = make_redbox_image(box=(80, 60, 520, 340))
    quad, conf, method = detect.find_red_box(img)
    assert quad is not None
    assert method.startswith("red")
    assert conf > 0.0
    x0, y0 = quad[:, 0].min(), quad[:, 1].min()
    x1, y1 = quad[:, 0].max(), quad[:, 1].max()
    assert abs(x0 - 80) < 20 and abs(y0 - 60) < 20
    assert abs(x1 - 520) < 20 and abs(y1 - 340) < 20


def test_find_red_box_falls_back_to_whole_image_when_no_red():
    img = np.full((400, 600, 3), 255, np.uint8)
    quad, conf, method = detect.find_red_box(img)
    assert method == "whole-image"
    assert conf == 0.0
    assert quad[:, 0].max() == 599 and quad[:, 1].max() == 399


def test_roi_bounds_insets_inside_the_box():
    quad = np.array([[80, 60], [520, 60], [520, 340], [80, 340]], np.float32)
    x0, y0, x1, y1 = detect.roi_bounds(quad, (400, 600), inset=10)
    assert (x0, y0, x1, y1) == (90, 70, 510, 330)


def test_find_case_locates_the_dark_case_not_the_red_box():
    # case (120,100,480,300) sits inside the red box (80,60,520,340);
    # find_case should hug the content, ignoring the red outline.
    img = make_redbox_image(box=(80, 60, 520, 340), case=(120, 100, 480, 300))
    quad, conf, method = detect.find_case(img)
    assert method == "content"
    x0, y0 = quad[:, 0].min(), quad[:, 1].min()
    x1, y1 = quad[:, 0].max(), quad[:, 1].max()
    assert abs(x0 - 120) < 30 and abs(x1 - 480) < 30
    assert abs(y0 - 100) < 30 and abs(y1 - 300) < 30


def test_find_roi_prefers_content():
    img = make_redbox_image()
    quad, conf, method = detect.find_roi(img)
    assert method == "content"


def test_find_case_returns_none_on_blank():
    img = np.full((400, 600, 3), 255, np.uint8)
    assert detect.find_case(img) is None


# ── A4 detection tests ────────────────────────────────────────────────────────

def _make_a4_scene(landscape=False):
    """Synthetic: near-white rectangle on dark background, A4-ish proportions."""
    if landscape:
        img_w, img_h, pw, ph = 1200, 900, 1100, 780
    else:
        img_w, img_h, pw, ph = 900, 1200, 780, 1100
    img = np.full((img_h, img_w, 3), 30, np.uint8)
    x0 = (img_w - pw) // 2
    y0 = (img_h - ph) // 2
    img[y0:y0 + ph, x0:x0 + pw] = 240
    return img


def test_detect_a4_finds_portrait_paper():
    img = _make_a4_scene(landscape=False)
    result = detect.detect_a4(img)
    assert result is not None
    _M, warped, landscape = result
    assert not landscape
    h, w = warped.shape[:2]
    assert w == int(detect.A4_W_CM * detect.A4_PX_PER_CM)
    assert h == int(detect.A4_H_CM * detect.A4_PX_PER_CM)


def test_detect_a4_finds_landscape_paper():
    img = _make_a4_scene(landscape=True)
    result = detect.detect_a4(img)
    assert result is not None
    _M, warped, landscape = result
    assert landscape
    h, w = warped.shape[:2]
    assert w == int(detect.A4_H_CM * detect.A4_PX_PER_CM)
    assert h == int(detect.A4_W_CM * detect.A4_PX_PER_CM)


def test_detect_a4_returns_none_on_dark_image():
    img = np.full((900, 1200, 3), 30, np.uint8)
    assert detect.detect_a4(img) is None


def test_closed_dvd_rect_within_portrait_a4():
    x0, y0, x1, y1 = detect.closed_dvd_rect_px()
    cw = int(detect.A4_W_CM * detect.A4_PX_PER_CM)
    ch = int(detect.A4_H_CM * detect.A4_PX_PER_CM)
    assert 0 <= x0 < x1 <= cw
    assert 0 <= y0 < y1 <= ch
    assert abs((x1 - x0) / detect.A4_PX_PER_CM - 13.5) < 0.5
    assert abs((y1 - y0) / detect.A4_PX_PER_CM - 19.0) < 0.5


def test_open_dvd_rect_within_landscape_a4():
    x0, y0, x1, y1 = detect.open_dvd_rect_px()
    cw = int(detect.A4_H_CM * detect.A4_PX_PER_CM)   # landscape width
    ch = int(detect.A4_W_CM * detect.A4_PX_PER_CM)   # landscape height
    assert 0 <= x0 < x1 <= cw
    assert 0 <= y0 < y1 <= ch
    assert abs((x1 - x0) / detect.A4_PX_PER_CM - 28.0) < 0.5
    assert abs((y1 - y0) / detect.A4_PX_PER_CM - 19.0) < 0.5
