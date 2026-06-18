import base64
import json
import cv2
import requests

from .config import (OLLAMA_URL, MODEL_TAG, VISION_TIMEOUT,
                     VISION_IMAGE_MAX_SIDE, VISION_NUM_CTX)
from .models import ClassifyResult, VALID_SIDES

CLASSIFY_PROMPT = (
    "You are looking at one photo of a single DVD item on a white background. "
    "The item is about {short:.0f} x {long:.0f} mm. {barcode_hint}\n"
    "Answer ONLY with JSON of this exact shape:\n"
    '{{"side": "front|back|center|spine|other", '
    '"rotation_cw": 0, "confidence": 0.0}}\n'
    "Definitions: 'front' = front cover art; 'back' = back cover (usually has a "
    "barcode and small print); 'center' = an open case / disc tray (much larger, "
    "landscape); 'spine' = thin edge. rotation_cw is the clockwise degrees "
    "(0, 90, 180 or 270) needed to make the item upright and readable. "
    "confidence is 0..1 for how sure you are about side and rotation. "
    "Do NOT read the title or add any other fields — keep the reply tiny."
)

LISTING_PROMPT = (
    "These photos show the front and back of one DVD for an eBay listing. "
    "Read the covers and answer ONLY with JSON:\n"
    '{{"listing_title": "", "description": "", "genre": "", "region": "", '
    '"runtime": "", "studio": "", "year": null}}\n'
    "listing_title: a concise search-friendly eBay title ending in 'DVD' with "
    "year and key terms. description: 1-2 plain sentences. Leave a field as an "
    "empty string if not visible. Do NOT invent a barcode, condition, or price."
)


def _bgr_to_b64(bgr, max_side=VISION_IMAGE_MAX_SIDE):
    h, w = bgr.shape[:2]
    if max(h, w) > max_side:
        f = max_side / float(max(h, w))
        bgr = cv2.resize(bgr, (int(w * f), int(h * f)))
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _chat(messages, model=MODEL_TAG):
    """POST to Ollama /api/chat, return the model's text content."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": model, "messages": messages, "stream": False,
              "format": "json", "keep_alive": "30m",
              "options": {"num_ctx": VISION_NUM_CTX}},
        timeout=VISION_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def ollama_available():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _coerce_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _coerce_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def classify_photo(bgr, size_mm, barcode):
    short_mm, long_mm = size_mm
    barcode_hint = ("A barcode reads " + barcode + " on this side."
                    if barcode else "No barcode was detected on this side.")
    prompt = CLASSIFY_PROMPT.format(short=short_mm, long=long_mm, barcode_hint=barcode_hint)
    messages = [{"role": "user", "content": prompt, "images": [_bgr_to_b64(bgr)]}]
    try:
        raw = _chat(messages)
        data = json.loads(raw)
    except Exception:
        return ClassifyResult(side="other", rotation_cw=0, title="", year=None, confidence=0.0)

    side = data.get("side")
    if side not in VALID_SIDES:
        side = "other"
    rot = _coerce_int(data.get("rotation_cw")) or 0
    rot = min((0, 90, 180, 270), key=lambda r: abs(r - (rot % 360)))
    conf = min(1.0, max(0.0, _coerce_float(data.get("confidence"))))
    return ClassifyResult(
        side=side,
        rotation_cw=rot,
        title=str(data.get("title") or "").strip(),
        year=_coerce_int(data.get("year")),
        confidence=conf,
    )


def draft_listing(front_bgr, back_bgr, barcode):
    images = [_bgr_to_b64(front_bgr)]
    if back_bgr is not None:
        images.append(_bgr_to_b64(back_bgr))
    messages = [{"role": "user", "content": LISTING_PROMPT, "images": images}]
    fields = {"listing_title": "", "description": "", "genre": "",
              "region": "", "runtime": "", "studio": "", "year": None}
    try:
        data = json.loads(_chat(messages))
    except Exception:
        return fields
    for k in fields:
        if k == "year":
            fields[k] = _coerce_int(data.get("year"))
        else:
            fields[k] = str(data.get(k) or "").strip()
    return fields
