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

    # Fit the case's TRUE four corners. A case shot at a slight angle is a
    # trapezoid, not a rectangle — feeding those real corners to the
    # perspective warp is what actually de-keystones the cover. minAreaRect
    # can only ever return a rectangle, so warping it is rectangle->rectangle:
    # a crop + rotate with ZERO perspective correction (the long-standing bug).
    quad = _quad_from_contour(merged)
    if quad is None or cv2.contourArea(quad) < 0.85 * rect_area:
        quad = cv2.boxPoints(rect)  # no clean 4-gon -> fall back to the rect
    box = quad / scale
    return _expand_and_clip(box, full_shape), coverage


def _quad_from_contour(merged):
    """The four corners of `merged` as a convex quadrilateral (same coords as
    the input), or None if no clean 4-gon emerges. Convex hull + polygon
    approximation at a widening tolerance — the standard document-scanner
    corner fit — so a keystoned case yields a trapezoid the warp can flatten."""
    hull = cv2.convexHull(merged)
    peri = cv2.arcLength(hull, True)
    for eps in (0.01, 0.02, 0.03, 0.04, 0.05):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    return None


def _expand_and_clip(box, full_shape):
    # expand 1.5% outward from the centroid: the rembg matte tends to sit a
    # hair INSIDE the case, which used to shave the edges off the crop
    center = box.mean(axis=0)
    box = center + (box - center) * 1.015
    h_full, w_full = full_shape[:2]
    box[:, 0] = np.clip(box[:, 0], 0, w_full - 1)
    box[:, 1] = np.clip(box[:, 1], 0, h_full - 1)
    return box.astype(np.float32)


def _loose_box_from_mask(m, scale, full_shape):
    """Last-resort fit: an axis-aligned bounding box over EVERY sizeable
    blob, with NO solidity requirement. An open DVD case (the 'inside' shot)
    is a hinged double panel plus two disc circles — it never fills a
    rectangle the way a flat cover does, so the strict rect-fit in
    _box_from_mask always rejects it and the caller used to fall all the way
    back to the uncropped photo, background and all. This still trims that
    background out, even if the box isn't perfectly tight."""
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)), iterations=3)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    frame_area = m.shape[0] * m.shape[1]
    # every blob at least 1% of the frame — small enough to catch a lone disc
    # label, large enough to ignore speckle noise
    keep = [c for c in cnts if cv2.contourArea(c) >= 0.01 * frame_area]
    if not keep:
        return None
    coverage = float(sum(cv2.contourArea(c) for c in keep)) / float(frame_area)
    # floor is well above rembg's spurious ~14% hallucination on a blank/
    # caseless photo (measured directly), well below real open-case content
    # (measured at 32%-45% on real inside shots) — see test_crop.py
    if not (0.18 <= coverage <= 0.98):
        return None
    merged = np.vstack(keep)
    x, y, w, h = cv2.boundingRect(merged)
    box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32) / scale
    return _expand_and_clip(box, full_shape), coverage


def detect_crop_box(bgr, rembg_model="isnet-general-use"):
    """(quad, coverage) or None. quad is 4x2 float32 corners in FULL-resolution
    image coordinates, in the exact order warp_to_quad expects — this is the
    box the browser editor should seed its draggable corners from.

    Four lines of attack, strongest first:
    1. rembg matte at a confident threshold (180)
    2. the same matte at looser thresholds (120, 60) — soft/uncertain mattes
       on dark or glossy cases still outline the right region
    3. paper segmentation — the case is whatever ISN'T bright white paper;
       works even when the AI matte fails completely
    4. loose axis-aligned bounding box, no solidity check — an OPEN case
       (the 'inside' shot: hinged double panel + two disc circles) never
       fills a rectangle the way a flat cover does, so tiers 1-3 always
       reject it. This still trims the surrounding background out, which
       beats the old behaviour of falling back to the raw, uncropped photo."""
    h, w = bgr.shape[:2]
    scale = _PROC_EDGE / float(max(h, w))
    small = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))))

    alpha_small = rembg_matte(small, rembg_model)
    loosest_alpha_mask = None
    if alpha_small is not None and alpha_small.shape == small.shape[:2]:
        for thr in (180, 120, 60):
            _, m = cv2.threshold(alpha_small, thr, 255, cv2.THRESH_BINARY)
            found = _box_from_mask(m, scale, bgr.shape)
            if found is not None:
                return found
            loosest_alpha_mask = m  # keep the loosest (thr=60) for tier 4

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    loosest_paper_mask = None
    for v_thr, s_thr in ((160, 60), (140, 80), (120, 90)):
        paper = (v > v_thr) & (s < s_thr)
        case = np.where(paper, 0, 255).astype(np.uint8)
        found = _box_from_mask(case, scale, bgr.shape)
        if found is not None:
            return found
        loosest_paper_mask = case  # keep the loosest (120,90) for tier 4

    for mask in (loosest_alpha_mask, loosest_paper_mask):
        if mask is None:
            continue
        found = _loose_box_from_mask(mask, scale, bgr.shape)
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
