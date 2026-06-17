"""Find the hand-drawn red placement box and derive a region of interest.

Ladder: HSV red mask -> largest 4-point contour -> bounding box of all red ->
whole image. Each step returns a confidence and a method label so low-confidence
detections can be flagged for review.
"""
import cv2
import numpy as np

# Red wraps around the hue circle, so we need two ranges.
_RED_LOWER_1 = np.array([0, 80, 60], np.uint8)
_RED_UPPER_1 = np.array([10, 255, 255], np.uint8)
_RED_LOWER_2 = np.array([170, 80, 60], np.uint8)
_RED_UPPER_2 = np.array([180, 255, 255], np.uint8)

RED_INSET_PX = 6   # how far to crop inside the red line by default


def red_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary mask (uint8 0/255) of red pixels."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, _RED_LOWER_1, _RED_UPPER_1) | \
        cv2.inRange(hsv, _RED_LOWER_2, _RED_UPPER_2)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)


def order_points(pts) -> np.ndarray:
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


def find_red_box(bgr: np.ndarray):
    """Return (quad 4x2 float32, confidence 0..1, method)."""
    h, w = bgr.shape[:2]
    img_area = float(h * w)
    mask = red_mask(bgr)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.05 * img_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best, best_area = approx.reshape(4, 2).astype(np.float32), area
    if best is not None:
        conf = min(1.0, (best_area / img_area) / 0.6)
        return order_points(best), conf, "red-contour"

    ys, xs = np.where(mask > 0)
    if xs.size > 0.02 * img_area:
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
        return quad, 0.4, "red-bbox"

    quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    return quad, 0.0, "whole-image"


def find_case(bgr: np.ndarray):
    """Locate the DVD case directly as the largest non-white, non-red blob.

    On the real template a sheet has two red boxes (one holding the case, one
    empty) plus X-marks and handwriting, and the red outlines merge into one
    contour -- so parsing "the box" is unreliable. The case itself is the only
    large patch of content that is neither white paper nor red marker, which is a
    far more robust signal. Returns (quad, confidence, "content") or None.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    red = red_mask(bgr) > 0
    content = ((gray < 205) & ~red).astype(np.uint8) * 255

    kx, ky = max(9, w // 60), max(9, h // 60)
    content = cv2.morphologyEx(
        content, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)), iterations=2)
    content = cv2.morphologyEx(
        content, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=1)

    contours, _ = cv2.findContours(content, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 0.05 * h * w:          # too small to be a case
        return None

    x, y, bw, bh = cv2.boundingRect(c)
    pad_x, pad_y = int(bw * 0.025), int(bh * 0.025)   # keep the plastic rim
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w, x + bw + pad_x)
    y1 = min(h, y + bh + pad_y)
    quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    conf = min(1.0, (area / (h * w)) / 0.40)
    return quad, conf, "content"


def find_roi(bgr: np.ndarray):
    """Region of interest for the cutout: case content first, red box as fallback."""
    found = find_case(bgr)
    if found is not None:
        return found
    return find_red_box(bgr)


# ── A4 physical constants ──────────────────────────────────────────────────────
A4_W_CM = 21.0     # portrait width
A4_H_CM = 29.7     # portrait height
A4_PX_PER_CM = 100  # canonical warp resolution (pixels per centimetre)

# Closed case (back / front) on portrait A4
# Bottom-left of DVD case at (3 cm from left, 3 cm from bottom)
_CLOSED_L_CM = 3.0
_CLOSED_B_CM = 3.0
_CLOSED_W_CM = 13.5
_CLOSED_H_CM = 19.0

# Open case (inside) on landscape A4
# Top-right of DVD case at (1 cm from right, 1 cm from top)
_OPEN_R_CM = 1.0
_OPEN_T_CM = 1.0
_OPEN_W_CM = 28.0
_OPEN_H_CM = 19.0


def detect_a4(bgr: np.ndarray):
    """Find the A4 paper and warp it to a canonical pixel space.

    Returns (M, warped_bgr, landscape) where M is the 3×3 homography and
    warped_bgr is the perspective-corrected sheet at A4_PX_PER_CM resolution.
    Returns None when no A4-like quadrilateral is found.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best, best_area = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.15 * h * w:
            continue
        peri = cv2.arcLength(c, True)
        for eps in (0.02, 0.03, 0.05):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4 and area > best_area:
                best, best_area = approx.reshape(4, 2).astype(np.float32), area
                break

    if best is None:
        return None

    pts = order_points(best)
    w_avg = (np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[2] - pts[3])) / 2
    h_avg = (np.linalg.norm(pts[3] - pts[0]) + np.linalg.norm(pts[2] - pts[1])) / 2
    landscape = w_avg > h_avg

    # Reject if aspect ratio is too far from A4 (within ±40 %)
    expected = (A4_H_CM / A4_W_CM) if landscape else (A4_W_CM / A4_H_CM)
    ratio = w_avg / max(h_avg, 1.0)
    if abs(ratio - expected) / expected > 0.40:
        return None

    px = A4_PX_PER_CM
    if landscape:
        cw, ch = int(A4_H_CM * px), int(A4_W_CM * px)   # 2970 × 2100
    else:
        cw, ch = int(A4_W_CM * px), int(A4_H_CM * px)   # 2100 × 2970

    dst = np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]], np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(bgr, M, (cw, ch))
    return M, warped, landscape


def closed_dvd_rect_px() -> tuple:
    """Crop rect (x0, y0, x1, y1) in portrait-A4 pixel space for back/front."""
    s = A4_PX_PER_CM
    ch = int(A4_H_CM * s)
    x0 = int(_CLOSED_L_CM * s)
    y1 = ch - int(_CLOSED_B_CM * s)
    y0 = y1 - int(_CLOSED_H_CM * s)
    x1 = x0 + int(_CLOSED_W_CM * s)
    return x0, y0, x1, y1


def open_dvd_rect_px() -> tuple:
    """Crop rect (x0, y0, x1, y1) in landscape-A4 pixel space for inside."""
    s = A4_PX_PER_CM
    cw = int(A4_H_CM * s)   # landscape width = 29.7 cm → 2970 px
    x1 = cw - int(_OPEN_R_CM * s)
    x0 = max(0, x1 - int(_OPEN_W_CM * s))
    y0 = int(_OPEN_T_CM * s)
    y1 = y0 + int(_OPEN_H_CM * s)
    return x0, y0, x1, y1


def roi_bounds(quad, shape, inset: int = RED_INSET_PX):
    """Axis-aligned (x0, y0, x1, y1) just inside the quad, clipped to the image."""
    h, w = shape[:2]
    x0 = max(0, int(round(quad[:, 0].min())) + inset)
    y0 = max(0, int(round(quad[:, 1].min())) + inset)
    x1 = min(w, int(round(quad[:, 0].max())) - inset)
    y1 = min(h, int(round(quad[:, 1].max())) - inset)
    if x1 - x0 < 10 or y1 - y0 < 10:   # inset too aggressive; back off
        x0 = max(0, int(round(quad[:, 0].min())))
        y0 = max(0, int(round(quad[:, 1].min())))
        x1 = min(w, int(round(quad[:, 0].max())))
        y1 = min(h, int(round(quad[:, 1].max())))
    return x0, y0, x1, y1
