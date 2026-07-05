"""Auto-crop a DVD case photo — adapted from redboxflip/scan.py (scan_case)
and redboxflip/cutout.py (rembg_matte).

detect_crop_box() returns the detected quad explicitly (not just a final
warped image) so the caller can hand it to the browser's corner-drag editor
as the SEEDED box — the old app's editor threw this away and reset to a
dumb 10%-margin rectangle every time; that is the exact bug this module's
API shape is designed to prevent a repeat of.
"""
import cv2
import numpy as np

_PROC_EDGE = 1000
_REMBG_SESSIONS = {}


def rembg_matte(bgr, model="isnet-general-use"):
    """Alpha matte from rembg, or None if rembg/model unavailable."""
    try:
        from rembg import remove, new_session
        from PIL import Image
    except Exception:
        return None
    try:
        if model not in _REMBG_SESSIONS:
            _REMBG_SESSIONS[model] = new_session(model)
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        out = remove(pil, session=_REMBG_SESSIONS[model])
        return np.asarray(out.convert("RGBA").getchannel("A"))
    except Exception:
        return None


def _order_pts(pts):
    pts = np.array(pts, np.float32)
    s = pts.sum(1)
    d = np.diff(pts, 1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def warp_to_quad(img, box):
    box = _order_pts(box)
    tl, tr, br, bl = box
    W = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    H = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if W < 10 or H < 10:
        return None
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(box, dst)
    return cv2.warpPerspective(img, M, (W, H))


def _box_from_mask(m, scale, full_shape):
    """Clean a binary mask and fit one rectangle to the case, or None.
    Glare/reflections often split the matte into pieces — every contour at
    least 30% the size of the biggest is merged before fitting, so a split
    case still yields ONE box instead of half a box."""
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)), iterations=3)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    biggest = max(cv2.contourArea(c) for c in cnts)
    keep = [c for c in cnts if cv2.contourArea(c) >= 0.3 * biggest]
    merged = np.vstack(keep)
    coverage = float(sum(cv2.contourArea(c) for c in keep)) / float(m.shape[0] * m.shape[1])
    if not (0.06 <= coverage <= 0.99):
        return None

    rect = cv2.minAreaRect(merged)
    rect_area = rect[1][0] * rect[1][1]
    # a DVD case FILLS its fitted rectangle (~0.99); diffuse rembg ghosts on
    # blank/odd frames don't (~0.75) — reject anything that isn't a solid slab
    if rect_area <= 0 or (coverage * m.shape[0] * m.shape[1]) / rect_area < 0.85:
        return None

    box = cv2.boxPoints(rect) / scale
    # expand 1.5% outward from the centroid: the rembg matte tends to sit a
    # hair INSIDE the case, which used to shave the edges off the crop
    center = box.mean(axis=0)
    box = center + (box - center) * 1.015
    h_full, w_full = full_shape[:2]
    box[:, 0] = np.clip(box[:, 0], 0, w_full - 1)
    box[:, 1] = np.clip(box[:, 1], 0, h_full - 1)
    return box.astype(np.float32), coverage


def detect_crop_box(bgr, rembg_model="isnet-general-use"):
    """(quad, coverage) or None. quad is 4x2 float32 corners in FULL-resolution
    image coordinates, in the exact order warp_to_quad expects — this is the
    box the browser editor should seed its draggable corners from.

    Three lines of attack, strongest first:
    1. rembg matte at a confident threshold (180)
    2. the same matte at looser thresholds (120, 60) — soft/uncertain mattes
       on dark or glossy cases still outline the right region
    3. paper segmentation — the case is whatever ISN'T bright white paper;
       works even when the AI matte fails completely"""
    h, w = bgr.shape[:2]
    scale = _PROC_EDGE / float(max(h, w))
    small = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))))

    alpha_small = rembg_matte(small, rembg_model)
    if alpha_small is not None and alpha_small.shape == small.shape[:2]:
        for thr in (180, 120, 60):
            _, m = cv2.threshold(alpha_small, thr, 255, cv2.THRESH_BINARY)
            found = _box_from_mask(m, scale, bgr.shape)
            if found is not None:
                return found

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    for v_thr, s_thr in ((160, 60), (140, 80), (120, 90)):
        paper = (v > v_thr) & (s < s_thr)
        case = np.where(paper, 0, 255).astype(np.uint8)
        found = _box_from_mask(case, scale, bgr.shape)
        if found is not None:
            return found
    return None


def compose_on_white(rgba, size=1600, margin=0.10):
    """Fit an RGBA crop onto a white square of side `size`, alpha-composited,
    with a white border of `margin` (fraction of the square) around the case
    so the listing photo breathes instead of touching the frame."""
    h, w = rgba.shape[:2]
    scale = min(size / h, size / w) * (1.0 - max(0.0, min(margin, 0.45)))
    resized = cv2.resize(rgba, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    rh, rw = resized.shape[:2]
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    y0, x0 = (size - rh) // 2, (size - rw) // 2

    rgb = resized[..., :3].astype(np.float32)
    alpha = (resized[..., 3:4].astype(np.float32)) / 255.0
    bg = canvas[y0:y0 + rh, x0:x0 + rw].astype(np.float32)
    blended = rgb * alpha + bg * (1 - alpha)
    canvas[y0:y0 + rh, x0:x0 + rw] = blended.astype(np.uint8)
    return canvas


def _feathered_alpha(h, w, frac=0.004):
    """Fully opaque except the outer ~0.4% of each edge, which fades smoothly
    to transparent — just enough that the case doesn't end in a hard scissor
    line, without visibly softening the cover."""
    f = max(2, int(round(frac * max(h, w))))
    alpha = np.full((h, w), 255, dtype=np.uint8)
    alpha[:f, :] = 0
    alpha[-f:, :] = 0
    alpha[:, :f] = 0
    alpha[:, -f:] = 0
    return cv2.GaussianBlur(alpha, (0, 0), max(1.0, f / 2.0))


def crop_and_compose(bgr, quad, size=1600):
    """Given a (possibly user-adjusted) quad, warp the case out and centre it
    on the white square. NO matting here — rembg is only for FINDING the case
    (detect_crop_box); running it on the warped crop erases white parts of
    the actual cover art. The warp already isolates the case exactly.
    Channels stay BGR end-to-end because the caller writes with cv2.imwrite."""
    warped_bgr = warp_to_quad(bgr, quad)
    if warped_bgr is None:
        raise ValueError("quad produced a degenerate warp (too small)")
    h, w = warped_bgr.shape[:2]
    return compose_on_white(np.dstack([warped_bgr, _feathered_alpha(h, w)]),
                            size=size)
