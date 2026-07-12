"""Shared pipeline configuration. Reads eBay creds from a .env file."""
import os
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent


def _load_env_file():
    for candidate_dir in [_THIS_DIR, *_THIS_DIR.parents]:
        env_path = candidate_dir / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
            return env_path
    return None


_load_env_file()

EBAY_CLIENT_ID = os.getenv("EBAY_CLIENT_ID", "")
EBAY_CLIENT_SECRET = os.getenv("EBAY_CLIENT_SECRET", "")
EBAY_MARKETPLACE_ID = os.getenv("EBAY_MARKETPLACE_ID", "EBAY_AU")


def ebay_configured():
    return bool(EBAY_CLIENT_ID and EBAY_CLIENT_SECRET)
