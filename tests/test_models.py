from dvdflip.models import ClassifyResult, Photo, DvdGroup
from dvdflip import config

def test_classify_result_defaults():
    r = ClassifyResult(side="front", rotation_cw=0, title="X", year=1967, confidence=0.9)
    assert r.side == "front" and r.rotation_cw == 0

def test_photo_defaults():
    from pathlib import Path
    p = Photo(source_path=Path("a.dng"))
    assert p.side == "other" and p.a4_found is True and p.deleted is False

def test_dvdgroup_defaults():
    g = DvdGroup(title="king kong escapes")
    assert g.photos == [] and g.approved is False

def test_config_constants():
    assert config.PX_PER_MM == 10
    assert config.OLLAMA_URL.startswith("http://")
    assert "qwen2.5vl" in config.MODEL_TAG
