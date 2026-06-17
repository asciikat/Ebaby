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

    try:
        data = pytesseract.image_to_data(
            pil, output_type=pytesseract.Output.DICT, config="--psm 3")
    except Exception:
        return None

    words = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        conf = int(data["conf"][i])
        if text and len(text) >= 2 and conf > 25:
            words.append({
                "text": text,
                "h": data["height"][i],
                "x": data["left"][i],
                "y": data["top"][i],
            })

    if not words:
        return None

    max_h = max(w["h"] for w in words)
    # Keep only the "big" words (title-sized characters)
    big = [w for w in words if w["h"] >= max_h * 0.70]
    if not big:
        return None

    # Sort by reading order: top-to-bottom, then left-to-right
    big.sort(key=lambda w: (w["y"], w["x"]))
    title = " ".join(w["text"] for w in big)
    return _clean(title) or None


def _clean(text: str) -> str:
    """Strip OCR noise characters and normalise whitespace."""
    text = re.sub(r"[^A-Za-z0-9 ':!&\-]", "", text)
    return " ".join(text.split()).strip()
