import numpy as np
import cv2

from .config import A4_SHORT_MM, A4_LONG_MM, A4_RATIO, PX_PER_MM


def order_points(pts):
    """Return points as [top-left, top-right, bottom-right, bottom-left]."""
    pts = np.asarray(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def detect_a4(bgr, dump=None):
    """Find the A4 sheet. Returns (quad_full_res or None, confidence)."""
    h, w = bgr.shape[:2]
    scale = 1100.0 / max(h, w)
    small = cv2.resize(bgr, (int(w * scale), int(h * scale)))
    sh, sw = small.shape[:2]
    img_area = float(sh * sw)

    L = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[:, :, 0]
    s = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1]
    Lb = cv2.GaussianBlur(L, (5, 5), 0)
    otsu, _ = cv2.threshold(Lb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thr = max(0.40 * float(L.max()), float(otsu) * 0.80)
    mask = ((Lb >= thr) & (s <= 95)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    if dump is not None:
        cv2.imwrite(dump, mask)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 0.0
    for c in cnts:
        hull = cv2.convexHull(c)
        area = cv2.contourArea(hull)
        if area < img_area * 0.05:
            continue
        rrect = cv2.minAreaRect(c)
        (bw, bh) = rrect[1]
        if bw < 1 or bh < 1:
            continue
        ratio = max(bw, bh) / min(bw, bh)
        fill = area / max(bw * bh, 1.0)
        ratio_score = max(0.0, 1.0 - abs(ratio - A4_RATIO) / 0.5)
        fill_score = max(0.0, min(1.0, (fill - 0.6) / 0.4))
        area_score = min(1.0, area / img_area / 0.5)
        score = 0.55 * ratio_score + 0.30 * fill_score + 0.15 * area_score
        if score > best_score:
            best_score = score
            best = cv2.boxPoints(rrect).astype(np.float32) / scale
    return best, float(best_score)


def warp_to_a4(bgr, quad, px_per_mm=PX_PER_MM):
    """Perspective-warp the sheet flat, PRESERVING the detected quad's own aspect.

    We deliberately do NOT force exact A4 proportions. When A4 detection is weak
    (e.g. the open case leaves only a thin paper frame) the detected quad's aspect
    differs from 1.414, and forcing it squashes the contents (a disc becomes an
    ellipse). Preserving the quad's measured aspect keeps shapes true; the long
    edge is scaled to ~A4 length so pixels still map ~1:1 to mm for the size test.
    """
    rect = order_points(quad)
    tl, tr, br, bl = rect
    wid = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    hei = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    long_px = max(wid, hei, 1.0)
    scale = (A4_LONG_MM * px_per_mm) / long_px
    W = max(int(round(wid * scale)), 2)
    H = max(int(round(hei * scale)), 2)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(bgr, M, (W, H), flags=cv2.INTER_CUBIC)
