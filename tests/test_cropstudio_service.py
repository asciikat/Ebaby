import cv2
import numpy as np
import pytest

from cropstudio import service


def _jpeg(color):
    img = np.full((40, 30, 3), color, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def test_save_dvd_names_from_front_and_writes_three(tmp_path):
    shots = {0: _jpeg((10, 10, 10)), 1: _jpeg((20, 20, 20)), 2: _jpeg((30, 30, 30))}
    out = service.save_dvd(shots, tmp_path, settings=None, dvd_counter=1,
                           reader=lambda bgr, s: "Goober And The Ghost Chasers")
    assert out["title"] == "Goober And The Ghost Chasers"
    d = tmp_path / "Goober And The Ghost Chasers"
    assert (d / "Goober And The Ghost Chasers - Back Cover.jpg").exists()
    assert (d / "Goober And The Ghost Chasers - Front Cover.jpg").exists()
    assert (d / "Goober And The Ghost Chasers - Inside.jpg").exists()


def test_save_dvd_falls_back_when_reader_returns_none(tmp_path):
    shots = {0: _jpeg((10, 10, 10)), 1: _jpeg((20, 20, 20)), 2: _jpeg((30, 30, 30))}
    out = service.save_dvd(shots, tmp_path, settings=None, dvd_counter=2,
                           reader=lambda bgr, s: None)
    assert out["title"] == "Untitled DVD 2"
    assert (tmp_path / "Untitled DVD 2" / "Untitled DVD 2 - Front Cover.jpg").exists()


def test_save_dvd_uniquifies_duplicate_title(tmp_path):
    shots = {0: _jpeg((1, 1, 1)), 1: _jpeg((2, 2, 2)), 2: _jpeg((3, 3, 3))}
    r = lambda bgr, s: "Heat"
    service.save_dvd(shots, tmp_path, None, 1, reader=r)
    out2 = service.save_dvd(shots, tmp_path, None, 2, reader=r)
    assert out2["dir"].endswith("Heat (2)")


def test_partial_dvd_uses_back_when_no_front(tmp_path):
    shots = {0: _jpeg((1, 1, 1))}        # only Back
    out = service.save_dvd(shots, tmp_path, None, 1, reader=lambda bgr, s: "Solo")
    assert (tmp_path / "Solo" / "Solo - Back Cover.jpg").exists()


def test_ensure_decodable_rejects_junk():
    with pytest.raises(ValueError):
        service.ensure_decodable(b"not a jpeg")
    service.ensure_decodable(_jpeg((5, 5, 5)))   # does not raise
