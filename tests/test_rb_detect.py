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
