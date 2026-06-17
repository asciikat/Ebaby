from redboxflip import config
from redboxflip.models import Settings


def test_load_returns_defaults_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "nope.json")
    s = config.load_settings()
    assert isinstance(s, Settings)
    assert s.cutout_engine == "geometric"


def test_save_then_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "s.json")
    s = Settings(input_dir="/in", output_dir="/out", margin_pct=10.0,
                 cutout_engine="sam")
    config.save_settings(s)
    loaded = config.load_settings()
    assert loaded.input_dir == "/in"
    assert loaded.margin_pct == 10.0
    assert loaded.cutout_engine == "sam"


def test_corrupt_file_falls_back_to_defaults(tmp_path, monkeypatch):
    p = tmp_path / "s.json"
    p.write_text("{not json")
    monkeypatch.setattr(config, "SETTINGS_PATH", p)
    s = config.load_settings()
    assert s.cutout_engine == "geometric"
