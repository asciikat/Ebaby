import os
from pathlib import Path

from cropstudio import paths


def test_output_root_defaults_to_project_processed():
    os.environ.pop("CROPSTUDIO_OUTPUT", None)
    root = paths.output_root()
    assert root.name == "processed"
    # project root is the parent of the cropstudio package
    assert root.parent == Path(paths.__file__).resolve().parent.parent


def test_output_root_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CROPSTUDIO_OUTPUT", str(tmp_path / "out"))
    assert paths.output_root() == tmp_path / "out"


def test_make_run_dir_creates_timestamped_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CROPSTUDIO_OUTPUT", str(tmp_path))
    d = paths.make_run_dir(now="20260623_151004")
    assert d == tmp_path / "run_20260623_151004"
    assert d.is_dir()
