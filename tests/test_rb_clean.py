import numpy as np
from PIL import Image

from redboxflip import clean
from redboxflip.models import Face


def test_compose_on_white_square_is_square_centered_and_white_margin():
    rgba = np.zeros((20, 40, 4), np.uint8)
    rgba[..., 0] = 255          # red
    rgba[..., 3] = 255          # opaque
    out = clean.compose_on_white_square(rgba, margin_pct=0)
    assert out.size == (40, 40)
    assert out.getpixel((0, 0)) == (255, 255, 255)
    assert out.getpixel((20, 20)) == (255, 0, 0)


def test_compose_adds_margin():
    rgba = np.zeros((20, 20, 4), np.uint8)
    rgba[..., 3] = 255
    out = clean.compose_on_white_square(rgba, margin_pct=10)
    assert out.size == (24, 24)


def test_auto_upright_rotates_wide_front_to_portrait():
    rgba = np.zeros((30, 60, 4), np.uint8)
    out, rot = clean.auto_upright(rgba, Face.FRONT)
    assert out.shape[0] > out.shape[1]
    assert rot == 90


def test_auto_upright_leaves_inside_landscape():
    rgba = np.zeros((30, 60, 4), np.uint8)
    out, rot = clean.auto_upright(rgba, Face.INSIDE)
    assert rot == 0


def test_erase_red_makes_red_transparent():
    rgba = np.zeros((10, 10, 4), np.uint8)
    rgba[..., 3] = 255
    rgba[0:3, 0:3, 0] = 255
    out = clean.erase_red_to_white(rgba)
    assert out[0, 0, 3] == 0
    assert out[9, 9, 3] == 255
