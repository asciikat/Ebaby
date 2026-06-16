"""Persisted settings and on-disk paths."""
import json
from pathlib import Path

from .models import Settings

HOME = Path.home()
SETTINGS_PATH = HOME / ".redboxflip.json"
TITLE_CACHE_PATH = HOME / ".redboxflip_titles.json"


def load_settings() -> Settings:
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return Settings.from_dict(data)
    except Exception:
        return Settings()


def save_settings(settings: Settings) -> None:
    try:
        SETTINGS_PATH.write_text(json.dumps(settings.to_dict(), indent=2),
                                 encoding="utf-8")
    except Exception:
        pass


def load_cache_for_batch() -> dict:
    """Thin wrapper so the pipeline doesn't import titles at module load."""
    from .titles import load_cache
    return load_cache()
