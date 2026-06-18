"""Read DVD info from photos with a local vision LLM (Qwen via Ollama).

Tesseract OCR fails on stylised / glossy / 3-D cover fonts (the common case). A
small vision model reads them reliably. Ollama runs on the Windows host; this
pipeline runs in WSL, so we reach the server through ``curl.exe`` — a Windows
binary invoked over WSL interop hits the Windows ``127.0.0.1`` natively, with no
port-forwarding, firewall rule, or mirrored-networking config.

Everything here is best-effort: any failure (no curl.exe, Ollama down, timeout,
odd output) returns None/{} so the pipeline falls back and never blocks.

The model is a poor judge of facts it can't see: in testing it confidently
invented region, studio and barcode digits. So the extraction PROMPT forbids
guessing, and the caller (`extract.py`) additionally validates sensitive fields
against the text the model actually transcribed. Barcodes never come from the
model at all — a real decoder reads those.
"""
import base64
import json
import re
import shutil
import subprocess
from typing import Optional

import cv2

_TITLE_PROMPT = (
    "You are reading a DVD front cover. Output the exact movie title shown, "
    "in the same words as printed. No punctuation you do not see, no extra "
    "words, no explanation. Title only."
)

_MAX_EDGE = 768        # downscale before sending: 8GB GPU is fast here, slow on full size
_TIMEOUT_S = 120


def _curl_cmd():
    """Prefer Windows curl.exe (WSL->Windows interop); fall back to native curl."""
    for exe in ("curl.exe", "curl"):
        if shutil.which(exe):
            return exe
    return None


def available() -> bool:
    return _curl_cmd() is not None


def _encode(bgr) -> Optional[str]:
    h, w = bgr.shape[:2]
    scale = _MAX_EDGE / float(max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))))
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode() if ok else None


def _generate(prompt, bgr, settings, fmt=None, num_predict=16):
    """One Ollama /api/generate call with an image. Returns response text or None."""
    curl = _curl_cmd()
    if curl is None:
        return None
    img_b64 = _encode(bgr)
    if img_b64 is None:
        return None

    payload = {
        "model": getattr(settings, "qwen_model", "qwen3.5:4b"),
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "think": False,
        "keep_alive": "5m",
        "options": {"num_predict": num_predict, "temperature": 0},
    }
    if fmt is not None:
        payload["format"] = fmt
    url = getattr(settings, "ollama_url", "http://127.0.0.1:11434") + "/api/generate"
    try:
        r = subprocess.run(
            [curl, "-s", "-m", str(_TIMEOUT_S), url,
             "-H", "Content-Type: application/json", "-d", "@-"],
            input=json.dumps(payload), capture_output=True, text=True,
            timeout=_TIMEOUT_S + 15)
        if r.returncode != 0 or not r.stdout.strip():
            return None
        return json.loads(r.stdout).get("response", "")
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Title (kept for the OCR-fallback path and for title-only runs)
# --------------------------------------------------------------------------- #

def _clean(text: str) -> str:
    text = text.strip().strip('"“”\'`').strip()
    return re.sub(r"\s+", " ", text)


def _looks_like_title(text: str) -> bool:
    if not text or len(text) < 2 or len(text) > 60 or "\n" in text:
        return False
    low = text.lower()
    bad = ("none", "i cannot", "i can't", "sorry", "unable", "no title",
           "cannot determine", "the movie title", "this dvd")
    return not any(b in low for b in bad)


def title_from_cover(bgr, settings) -> Optional[str]:
    """Best-guess movie title from a front-cover BGR image, or None."""
    resp = _generate(_TITLE_PROMPT, bgr, settings, num_predict=16)
    if resp is None:
        return None
    title = _clean(resp)
    return title if _looks_like_title(title) else None


# --------------------------------------------------------------------------- #
# Orientation
# --------------------------------------------------------------------------- #

_UPRIGHT_PROMPT = (
    "This is a photo of one face of a DVD case. Read the largest title or "
    "heading text. Decide if the image is upright or upside-down. Answer "
    "upright=true only if the main text reads normally left to right."
)
_UPRIGHT_FMT = {
    "type": "object",
    "properties": {"upright": {"type": "boolean"}},
    "required": ["upright"],
}


def is_upright(bgr, settings) -> Optional[bool]:
    """True/False from the vision model, or None if it couldn't decide."""
    resp = _generate(_UPRIGHT_PROMPT, bgr, settings, fmt=_UPRIGHT_FMT, num_predict=40)
    if not resp:
        return None
    try:
        return bool(json.loads(resp).get("upright"))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Full field extraction (one call per face)
# --------------------------------------------------------------------------- #

_EXTRACT_PROMPT = (
    "You are reading ONE photo of a DVD case (front, back, or the inside/disc) "
    "to help build an eBay listing.\n"
    "STRICT RULES:\n"
    "1. Only report text you can ACTUALLY SEE printed in this image.\n"
    "2. Do NOT use prior knowledge about the movie. Do NOT guess.\n"
    "3. If a field is not printed or not legible, set it to \"Unknown\".\n"
    "4. Never invent a barcode number, region, rating, or year.\n"
    "5. List in low_confidence every field you are not fully sure you read "
    "correctly.\n"
    "Also report whether the image is upright (main text reads normally) and "
    "transcribe all readable text into all_text."
)

_STR = {"type": "string"}
_EXTRACT_FMT = {
    "type": "object",
    "properties": {
        "upright": {"type": "boolean"},
        "all_text": _STR,
        "title": _STR,
        "format": _STR,
        "region": _STR,
        "pal_ntsc": _STR,
        "rating": _STR,
        "release_year": _STR,
        "studio": _STR,
        "edition": _STR,
        "num_discs": _STR,
        "languages": _STR,
        "subtitles": _STR,
        "genre": _STR,
        "special_features": _STR,
        "notes": _STR,
        "condition_notes": _STR,
        "low_confidence": {"type": "array", "items": _STR},
    },
    "required": ["upright", "all_text", "title"],
}


def extract_face(bgr, settings) -> dict:
    """Structured read of one face. Returns a dict (possibly {}) — never raises.

    Keys mirror `_EXTRACT_FMT`. The caller decides confidence and validates the
    sensitive fields; this just returns what the model claims to see.
    """
    resp = _generate(_EXTRACT_PROMPT, bgr, settings, fmt=_EXTRACT_FMT, num_predict=600)
    if not resp:
        return {}
    try:
        data = json.loads(resp)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}
