"""Reads eBay Browse API credentials from THIS project's own .env.

Deliberately separate from the WSL pipeline's copy (Pipeline/2_barcode_ebay/
.env) — that one was exposed in a chat session and must not be reused as-is.
"""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EbayConfig:
    client_id: str = ""
    client_secret: str = ""
    marketplace_id: str = "EBAY_AU"

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)


def _parse_env_file(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def load_ebay_config(project_root=None) -> EbayConfig:
    """Reads EBAY_CLIENT_ID / EBAY_CLIENT_SECRET / EBAY_MARKETPLACE_ID from
    <project_root>/.env, falling back to the process environment.
    `project_root` defaults to this repo's root (parent of `cropstudio/`).
    """
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    env_path = root / ".env"
    values = dict(os.environ)
    if env_path.exists():
        values.update(_parse_env_file(env_path.read_text(encoding="utf-8")))
    return EbayConfig(
        client_id=values.get("EBAY_CLIENT_ID", ""),
        client_secret=values.get("EBAY_CLIENT_SECRET", ""),
        marketplace_id=values.get("EBAY_MARKETPLACE_ID", "EBAY_AU"),
    )
