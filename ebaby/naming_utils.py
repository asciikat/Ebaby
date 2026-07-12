"""Shared naming logic so that image filenames and CSV rows use the exact
same slug for the same eBay title / barcode."""
import re
import unicodedata

STOP_WORDS = {
    "a", "an", "and", "or", "of", "to", "from", "in", "on", "for",
    "with", "by", "at", "as", "is", "it",
}


def _asciify(text):
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def slugify_title(title, fallback, max_words=3):
    ascii_title = _asciify((title or "").title())
    words = re.findall(r"\w+", ascii_title)
    kept = [w for w in words if w.lower() not in STOP_WORDS] or words
    deduped = list(dict.fromkeys(kept))
    return "_".join(deduped[:max_words]) or fallback
