"""Extract movie title from the front cover image via OCR.

Strategy: run Tesseract word-level detection, then collect all words whose
character height is within 70 % of the tallest detected word. Those are the
"big" words — almost always the movie title on a front cover. Sort them by
position (top→bottom, left→right) and join into a single string.
"""
import re
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# Australian / common classification badges that are never the movie title.
_RATING_RE = re.compile(
    r'^(G|PG|M|MA|R|RC|NR|RP|E|AO|M15|MA15|R18|PG13|NC17|TV[A-Z0-9]*)$',
    re.IGNORECASE)


def extract_title(img) -> Optional[str]:
    """Return best-guess movie title from a front-cover image, or None.

    img: BGR ndarray (cropped front cover) or PIL Image.
    Requires the 'pytesseract' package and the tesseract-ocr binary.
    Returns None when unavailable or when no confident text is found.
    """
    try:
        import pytesseract
    except ImportError:
        return None

    if isinstance(img, np.ndarray):
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        pil = img.convert("RGB")

    # Normalise size: Tesseract's word segmentation is resolution-sensitive, so
    # scale every cover to the same long edge for consistent height ranking.
    long_edge = max(pil.size)
    if long_edge != 1600:
        f = 1600.0 / long_edge
        pil = pil.resize((max(1, int(pil.width * f)), max(1, int(pil.height * f))),
                         Image.LANCZOS)

    try:
        # psm 11 (sparse text) reads large stylised cover titles far better than
        # psm 3 — e.g. white 3-D "OPEN WATER" on a dark background.
        data = pytesseract.image_to_data(
            pil, output_type=pytesseract.Output.DICT, config="--psm 11")
    except Exception:
        return None

    words = []
    for i, text in enumerate(data["text"]):
        cleaned = _clean(text.strip())   # clean first so "M." → "M" (len 1, filtered)
        conf = int(data["conf"][i])
        if cleaned and len(cleaned) >= 2 and conf > 25:
            words.append({
                "text": cleaned,
                "h": data["height"][i],
                "x": data["left"][i],
                "y": data["top"][i],
                "conf": conf,
            })

    if not words:
        return None

    max_h = max(w["h"] for w in words)
    # Keep only the "big" words (title-sized), confidently read, excluding
    # classification badges. The title is by far the tallest text on a cover;
    # 0.78 keeps it while dropping taglines and stray noise just under it.
    big = [w for w in words
           if w["h"] >= max_h * 0.78 and w["conf"] >= 45
           and not _RATING_RE.match(w["text"])]
    if not big:
        return None

    # Sort by reading order: top-to-bottom, then left-to-right
    big.sort(key=lambda w: (w["y"], w["x"]))
    title = " ".join(w["text"] for w in big)
    return _clean(title) or None


def count_words(img, max_dim: int = 800) -> int:
    """Return count of words detected (confidence ≥ 20). Fast — downscales first."""
    try:
        import pytesseract
    except ImportError:
        return 0
    if isinstance(img, np.ndarray):
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        pil = img.convert("RGB")
    w, h = pil.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    try:
        data = pytesseract.image_to_data(
            pil, output_type=pytesseract.Output.DICT, config="--psm 3")
    except Exception:
        return 0
    return sum(1 for i, t in enumerate(data["text"])
               if t.strip() and int(data["conf"][i]) > 20)


def _clean(text: str) -> str:
    """Strip OCR noise characters and normalise whitespace."""
    text = re.sub(r"[^A-Za-z0-9 ':!&\-]", "", text)
    return " ".join(text.split()).strip()
