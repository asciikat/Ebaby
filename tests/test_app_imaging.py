"""Tests for the listing-image pipeline in app.py: background-removed, straightened,
centered on a 1000x1000 white square. These use synthetic arrays and the offline
fallback paths, so they need neither rembg nor a network."""
import cv2
import numpy as np

import app


def test_pad_to_square_is_square_and_white():
    bgr = np.full((300, 200, 3), (10, 20, 30), np.uint8)
    out = app._pad_to_square(bgr, 256, 0.06)
    assert out.shape == (256, 256, 3)
    assert (out[0, 0] == 255).all() and (out[-1, -1] == 255).all()  # white margins
    assert (out[128, 128] != 255).any()                            # object centered


def test_paste_centered_blends_onto_white():
    fg = np.full((300, 200, 3), (40, 50, 200), np.uint8)
    alpha = np.full((300, 200), 255, np.uint8)
    out = app._paste_centered(fg, alpha, 512, 0.06)
    assert out.shape == (512, 512, 3)
    assert (out[0, 0] == 255).all()        # corner is pure white
    assert (out[256, 256] != 255).any()    # object present in the middle


def test_largest_component_drops_specks():
    alpha = np.zeros((100, 100), np.uint8)
    alpha[10:60, 10:60] = 255   # the real object
    alpha[90:95, 90:95] = 255   # a stray speck
    cleaned = app._largest_component(alpha)
    assert cleaned[30, 30] == 255   # object kept
    assert cleaned[92, 92] == 0     # speck removed


def test_deskew_preserves_alpha_channel():
    bgra = np.zeros((120, 120, 4), np.uint8)
    bgra[:, :, :3] = 128
    # a diamond (rotated square) -> minAreaRect angle != 0 -> should rotate
    alpha = np.zeros((120, 120), np.uint8)
    cv2.fillPoly(alpha, [np.array([[60, 20], [100, 60], [60, 100], [20, 60]])], 255)
    bgra[:, :, 3] = alpha
    out = app._deskew_bgra(bgra)
    assert out.shape[2] == 4
    assert int((out[:, :, 3] > 20).sum()) > 0   # object survived the rotation


def test_prepare_listing_image_with_matte(monkeypatch):
    img = np.full((400, 300, 3), 128, np.uint8)

    def fake_matte(bgr):
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        a = np.zeros((400, 300), np.uint8)
        a[50:350, 50:250] = 255
        bgra[:, :, 3] = a
        return bgra

    monkeypatch.setattr(app, "remove_background", fake_matte)
    out = app.prepare_listing_image(img, 512)
    assert out.shape == (512, 512, 3)
    assert (out[0, 0] == 255).all()        # background removed -> white corner
    assert (out[256, 256] != 255).any()    # DVD centered


def test_prepare_listing_image_falls_back_without_matte(monkeypatch):
    # When no matte engine is available, still return a white size x size square.
    monkeypatch.setattr(app, "remove_background", lambda bgr: None)
    img = np.full((400, 600, 3), 200, np.uint8)
    cv2.rectangle(img, (220, 120), (380, 300), (30, 30, 30), -1)
    out = app.prepare_listing_image(img, 256)
    assert out.shape == (256, 256, 3)
    assert (out[0, 0] == 255).all()
