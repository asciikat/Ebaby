import numpy as np
from PIL import Image
from redboxflip import imaging


def test_bgr_pil_roundtrip():
    bgr = np.zeros((4, 6, 3), dtype=np.uint8)
    bgr[:, :, 2] = 255  # red in BGR
    pil = imaging.bgr_to_pil(bgr)
    assert pil.size == (6, 4)
    assert pil.getpixel((0, 0)) == (255, 0, 0)  # red in RGB
    back = imaging.pil_to_bgr(pil)
    assert np.array_equal(back, bgr)


def test_resize_max_keeps_aspect_and_caps_edge():
    bgr = np.zeros((100, 200, 3), dtype=np.uint8)
    out = imaging.resize_max(bgr, 100)
    assert max(out.shape[:2]) == 100
    assert out.shape[1] == 100 and out.shape[0] == 50


def test_resize_max_noop_when_small():
    bgr = np.zeros((10, 20, 3), dtype=np.uint8)
    out = imaging.resize_max(bgr, 100)
    assert out.shape == bgr.shape


def test_load_image_bgr_reads_jpg(tmp_path):
    p = tmp_path / "x.jpg"
    Image.new("RGB", (8, 5), (10, 20, 30)).save(p)
    bgr = imaging.load_image_bgr(p)
    assert bgr.shape == (5, 8, 3)
    assert tuple(int(c) for c in bgr[0, 0]) == (30, 20, 10)
