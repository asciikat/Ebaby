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
