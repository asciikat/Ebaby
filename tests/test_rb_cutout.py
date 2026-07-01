import numpy as np

from redboxflip import cutout
from rb_helpers import make_redbox_image


def test_exact_quad_cutout_keeps_whole_selection_opaque():
    # A solid coloured box; exact crop must return it fully opaque, nothing clipped.
    img = np.zeros((200, 300, 3), np.uint8)
    img[40:160, 60:240] = (200, 120, 50)
    quad = np.array([[60, 40], [240, 40], [240, 160], [60, 160]], np.float32)
    rgba, method = cutout.make_cutout(img, quad, "exact", feather_px=0)
    assert method == "exact"
    assert rgba.shape[2] == 4
    assert (rgba[:, :, 3] == 255).all()          # nothing masked away
    assert abs(rgba.shape[1] - 180) <= 2 and abs(rgba.shape[0] - 120) <= 2


def test_exact_quad_cutout_deskews_a_tilted_quad():
    # Corners on a rotated rectangle -> output is an upright rectangle.
    img = np.full((400, 400, 3), 255, np.uint8)
    quad = np.array([[120, 80], [320, 140], [280, 320], [80, 260]], np.float32)
    rgba = cutout.exact_quad_cutout(img, quad)
    assert rgba is not None
    assert rgba.shape[0] > 10 and rgba.shape[1] > 10
    assert (rgba[:, :, 3] == 255).all()


def test_exact_quad_cutout_rejects_degenerate():
    img = np.zeros((50, 50, 3), np.uint8)
    tiny = np.array([[10, 10], [12, 10], [12, 12], [10, 12]], np.float32)
    assert cutout.exact_quad_cutout(img, tiny) is None
    assert cutout.exact_quad_cutout(img, None) is None


def test_geometric_matte_covers_the_case():
    img = make_redbox_image()
    roi = img[60:340, 80:520]
    alpha = cutout.geometric_matte(roi)
    assert alpha.shape == roi.shape[:2]
    cov = (alpha > 20).mean()
    assert 0.3 < cov < 0.99


def test_feather_softens_edges():
    alpha = np.zeros((50, 50), np.uint8)
    alpha[10:40, 10:40] = 255
    soft = cutout.feather_alpha(alpha, 3)
    edge_vals = np.unique(soft)
    assert len(edge_vals) > 2


def test_validate_coverage():
    assert cutout.validate_coverage(np.full((10, 10), 255, np.uint8)) is False
    good = np.zeros((10, 10), np.uint8); good[2:8, 2:8] = 255
    assert cutout.validate_coverage(good) is True
    assert cutout.validate_coverage(np.zeros((10, 10), np.uint8)) is False


def test_make_cutout_returns_rgba_tight_to_alpha():
    img = make_redbox_image()
    quad = np.array([[80, 60], [520, 60], [520, 340], [80, 340]], np.float32)
    rgba, method = cutout.make_cutout(img, quad, "geometric", feather_px=2)
    assert rgba.shape[2] == 4
    assert method == "geometric"
    a = rgba[:, :, 3]
    assert a[:, 0].max() > 0 or a[0, :].max() > 0


def test_make_cutout_falls_back_when_ai_engine_returns_none(monkeypatch):
    img = make_redbox_image()
    quad = np.array([[80, 60], [520, 60], [520, 340], [80, 340]], np.float32)
    monkeypatch.setattr(cutout, "rembg_matte", lambda roi, model=None: None)
    rgba, method = cutout.make_cutout(img, quad, "rembg", feather_px=0)
    assert method in ("grabcut", "geometric")
    assert rgba.shape[2] == 4
