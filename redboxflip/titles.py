"""Resolve a DVD title from a barcode: cache -> online lookup -> manual."""
import json
import re
import urllib.parse
import urllib.request

from .config import TITLE_CACHE_PATH

_NOISE = [
    r"\s*\((DVD|Blu-?ray|4K|UHD)[^)]*\)\s*$",
    r"\s*\[(DVD|Blu-?ray|4K|UHD)[^\]]*\]\s*$",
    r"\s*-\s*(DVD|Blu-?ray|4K|UHD)\s*$",
    r"\s*\bRegion\s*\d+.*$",
]


def clean_title(raw: str) -> str:
    if not raw:
        return ""
    text = raw.strip()
    for pat in _NOISE:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return text.strip(" -[]")


def load_cache() -> dict:
    try:
        return json.loads(TITLE_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cache(cache: dict) -> None:
    try:
        TITLE_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass


def lookup_online(barcode: str, timeout: int = 8) -> str:
    """Best-effort title from the free upcitemdb trial endpoint. '' on failure."""
    if not barcode:
        return ""
    try:
        url = "https://api.upcitemdb.com/prod/trial/lookup?upc=" + urllib.parse.quote(barcode)
        req = urllib.request.Request(url, headers={"User-Agent": "redboxflip/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        items = data.get("items") or []
        if items:
            return clean_title(items[0].get("title") or "")
    except Exception:
        pass
    return ""


def resolve_title(barcode, *, do_lookup: bool, cache: dict, manual: str = None):
    """Return (title, source) where source is manual|cache|lookup|none."""
    if manual:
        cleaned = manual.strip()
        if barcode:
            cache[barcode] = cleaned
        return cleaned, "manual"
    if barcode and barcode in cache and cache[barcode]:
        return cache[barcode], "cache"
    if do_lookup and barcode:
        found = lookup_online(barcode)
        if found:
            cache[barcode] = found
            return found, "lookup"
    return "", "none"
