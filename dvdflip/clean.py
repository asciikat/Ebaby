import numpy as np
import cv2
from PIL import Image

from .config import PX_PER_MM, PADDING_PCT
from .flatten import order_points


def paper_mask(flat_bgr):
    L = cv2.cvtColor(flat_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    s = cv2.cvtColor(flat_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    bright = L >= max(np.percentile(L, 55), 0.6 * float(L.max()))
    neutral = s <= 45
    return bright & neutral


def white_balance_from_paper(flat_bgr, target=242.0):
    """Neutralise cast using paper as white, lift exposure. Returns (bgr, ok)."""
    pm = paper_mask(flat_bgr)
    img = flat_bgr.astype(np.float32)
    if pm.sum() < 0.02 * pm.size:
        means = img.reshape(-1, 3).mean(axis=0)
        ok = False
    else:
        means = img[pm].mean(axis=0)
        ok = True
    means = np.maximum(means, 1.0)
    gains = means.mean() / means
    bal = img * gains
    if ok:
        paper_after = (bal[pm].mean(axis=0)).mean()
    else:
        paper_after = np.percentile(bal, 95)
    if paper_after > 1.0:
        bal *= (target / paper_after)
    return np.clip(bal, 0, 255).astype(np.uint8), ok


def segment_dvd(flat_bgr, dump=None):
    """Find the DVD on the white sheet -> rotated-rect box (4 pts) or None."""
    h, w = flat_bgr.shape[:2]
    minc = flat_bgr.min(axis=2)
    white_level = float(np.percentile(minc, 92))
    thr = max(120.0, 0.72 * white_level)
    fg = (minc < thr).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    cleaned = np.zeros_like(fg)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = (x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1)
        if not touches:
            cleaned[lab == i] = 255
    if cleaned.any():
        fg = cleaned
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    if dump is not None:
        cv2.imwrite(dump, fg)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.04 * h * w:
        return None
    return cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)


def dvd_size_mm(box):
    """(short_mm, long_mm) of the DVD's rotated rectangle."""
    r = order_points(box)
    w = np.linalg.norm(r[1] - r[0])
    h = np.linalg.norm(r[3] - r[0])
    long_mm = max(w, h) / PX_PER_MM
    short_mm = min(w, h) / PX_PER_MM
    return float(short_mm), float(long_mm)


def crop_rect(bgr, box):
    """Deskew/crop a rotated-rectangle region to an upright image."""
    rect = order_points(box)
    tl, tr, br, bl = rect
    W = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    H = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    W, H = max(W, 2), max(H, 2)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(bgr, M, (W, H), flags=cv2.INTER_CUBIC)


def scan_barcodes(bgr):
    """Decoded barcode strings (multi-rotation). [] if none, None if pyzbar missing."""
    try:
        from pyzbar.pyzbar import decode
    except ImportError:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    longest = max(pil.size)
    if longest < 1600:
        f = 1600.0 / longest
        pil = pil.resize((int(pil.width * f), int(pil.height * f)))
    found = []
    for rot in (0, 90, 180, 270):
        test = pil if rot == 0 else pil.rotate(-rot, expand=True)
        try:
            for b in decode(test):
                code = b.data.decode("utf-8", errors="replace")
                if code and code not in found:
                    found.append(code)
        except Exception:
            pass
        if found:
            break
    return found


def enhance(bgr, strength=0.6):
    """Gentle local-contrast + saturation pop."""
    if strength <= 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.12, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return np.clip(bgr.astype(np.float32) * (1 - strength)
                   + out.astype(np.float32) * strength, 0, 255).astype(np.uint8)


def composite_on_white(dvd_bgr, padding_pct=PADDING_PCT, square=True):
    h, w = dvd_bgr.shape[:2]
    pad = int(max(w, h) * padding_pct / 100.0)
    if square:
        side = max(w, h) + 2 * pad
        canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    else:
        canvas = np.full((h + 2 * pad, w + 2 * pad, 3), 255, dtype=np.uint8)
    y = (canvas.shape[0] - h) // 2
    x = (canvas.shape[1] - w) // 2
    canvas[y:y + h, x:x + w] = dvd_bgr
    return canvas


def resize_max(bgr, max_dim):
    h, w = bgr.shape[:2]
    if max_dim <= 0 or max(h, w) <= max_dim:
        return bgr
    f = max_dim / float(max(h, w))
    return cv2.resize(bgr, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
