import numpy as np

from redboxflip import cutout
from rb_helpers import make_redbox_image


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
