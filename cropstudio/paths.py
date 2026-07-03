"""Where Crop Studio writes its output (native Windows path, not the WSL default)."""
import os
import sys
import time
from pathlib import Path


def project_root() -> Path:
    """The Ebaby project root = the parent of the cropstudio package."""
    return Path(__file__).resolve().parent.parent


def output_root() -> Path:
    """processed/ root. CROPSTUDIO_OUTPUT overrides; frozen exe writes next
    to itself; dev default is <project>/processed."""
    env = os.environ.get("CROPSTUDIO_OUTPUT")
    if env:
        return Path(env)
    if getattr(sys, "frozen", False):          # PyInstaller bundle
        return Path(sys.executable).resolve().parent / "processed"
    return project_root() / "processed"


def make_run_dir(now: str = None) -> Path:
    """Create and return processed/run_<timestamp>/ for this session."""
    ts = now or time.strftime("%Y%m%d_%H%M%S")
    d = output_root() / f"run_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d
