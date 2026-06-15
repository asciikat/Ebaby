import numpy as np
from dvdflip.loader import load_bgr, gather_inputs

def test_load_dng_returns_bgr(sample_front):
    bgr = load_bgr(sample_front)
    assert isinstance(bgr, np.ndarray)
    assert bgr.ndim == 3 and bgr.shape[2] == 3
    assert bgr.dtype == np.uint8
    assert min(bgr.shape[:2]) > 1000

def test_gather_inputs_finds_dngs(tmp_path):
    (tmp_path / "a.dng").write_bytes(b"x")
    (tmp_path / "b.txt").write_text("no")
    found = gather_inputs(tmp_path)
    assert [p.name for p in found] == ["a.dng"]
