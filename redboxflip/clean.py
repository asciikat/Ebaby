"""Post-matte cleanup: erase residual red, auto-upright, compose on white."""
import cv2
import numpy as np
from PIL import Image

from .detect import red_mask
from .models import Face


def erase_red_to_white(rgba: np.ndarray) -> np.ndarray:
    """Set any residual red pixels (in an RGBA array) to transparent."""
    rgb = rgba[:, :, :3]
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    m = red_mask(bgr)
    rgba = rgba.copy()
    rgba[m > 0, 3] = 0
    return rgba


def auto_upright(rgba: np.ndarray, face: Face, barcode_rot=None):
    """Rotate in 90 deg steps toward the expected orientation. Returns (rgba, deg).

    front/back are portrait (taller than wide); inside is landscape.
    A barcode_rot hint is accepted for future use but the deterministic aspect
    rule drives the default.
    """
    h, w = rgba.shape[:2]
    rot = 0
    if face in (Face.FRONT, Face.BACK) and w > h:
        rot = 90
    elif face == Face.INSIDE and h > w:
        rot = 90
    if rot:
        rgba = np.ascontiguousarray(np.rot90(rgba, k=1))   # 90 deg counter-clockwise
    return rgba, rot


def compose_on_white_square(rgba, margin_pct: float) -> Image.Image:
    """Center a (feathered) RGBA cutout on a pure-white square canvas."""
    pil = Image.fromarray(rgba, "RGBA") if isinstance(rgba, np.ndarray) else rgba.convert("RGBA")
    w, h = pil.size
    side = max(w, h)
    margin = int(round(side * margin_pct / 100.0))
    canvas_side = side + 2 * margin
    canvas = Image.new("RGB", (canvas_side, canvas_side), "white")
    ox = (canvas_side - w) // 2
    oy = (canvas_side - h) // 2
    canvas.paste(pil, (ox, oy), pil.getchannel("A"))
    return canvas


def colour_tidy_bgr(bgr: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """Gentle white-balance + local contrast + mild saturation (ported from V1)."""
    if strength <= 0:
        return bgr
    original = bgr.astype(np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    paper = gray >= np.percentile(gray, 95)
    white_ref = original[paper].mean(axis=0) if np.any(paper) else original.reshape(-1, 3).mean(axis=0)
    cast = white_ref / max(white_ref.mean(), 1e-6)
    wb = np.clip(original / np.maximum(cast, 1e-6), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(wb, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
    hh, ss, vv = cv2.split(hsv)
    ss = np.clip(ss * 1.12, 0, 255)
    final = cv2.cvtColor(cv2.merge((hh, ss, vv)).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    return np.clip(original * (1 - strength) + final * strength, 0, 255).astype(np.uint8)


def colour_tidy_rgba(rgba: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """Apply colour tidy to the RGB of an RGBA array, preserving alpha."""
    rgb = rgba[:, :, :3]
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    tidied = colour_tidy_bgr(bgr, strength)
    out = rgba.copy()
    out[:, :, :3] = tidied[:, :, ::-1]
    return out
