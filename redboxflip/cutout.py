"""Matte the DVD case off its background.

Engines, in fallback order per selection:
  rembg   : rembg -> grabcut -> geometric
  sam     : sam   -> grabcut -> geometric
  grabcut : grabcut -> geometric
  geometric: geometric only
The chosen AI engine runs only on the red-box ROI so surrounding paper cannot
confuse it. Output is an RGBA array cropped tight to the (feathered) alpha.
"""
import cv2
import numpy as np

from .detect import red_mask, roi_bounds

_REMBG_SESSIONS = {}
_SAM_PREDICTOR = None

ENGINE_LADDER = {
    "exact": ["exact"],
    "rembg": ["rembg", "grabcut", "geometric"],
    "sam": ["sam", "grabcut", "geometric"],
    "grabcut": ["grabcut", "geometric"],
    "geometric": ["geometric"],
}


def _order_quad(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    d = (pts[:, 0] - pts[:, 1])
    return np.array([pts[np.argmin(s)], pts[np.argmax(d)],
                     pts[np.argmax(s)], pts[np.argmin(d)]], np.float32)


def exact_quad_cutout(bgr, quad):
    """Perspective-warp exactly the 4 selected corners to an upright rectangle.

    No matting: the output is precisely what the quad enclosed, deskewed, fully
    opaque. This is the manual-crop escape hatch for transparent cases that the
    AI mattes clip — the user can see the case edge, so they place the corners
    and get exactly that, nothing removed. Returns RGBA, or None if degenerate.
    """
    if quad is None:
        return None
    tl, tr, br, bl = _order_quad(quad)
    W = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    H = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    if W < 10 or H < 10:
        return None
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], np.float32), dst)
    warped = cv2.warpPerspective(bgr, M, (W, H))
    rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    alpha = np.full((H, W), 255, np.uint8)
    return np.dstack([rgb, alpha])


def validate_coverage(alpha: np.ndarray, min_cov: float = 0.05,
                      max_cov: float = 0.985) -> bool:
    cov = float((alpha > 20).mean())
    return min_cov <= cov <= max_cov


def feather_alpha(alpha: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return alpha
    k = int(px) * 2 + 1
    return cv2.GaussianBlur(alpha, (k, k), 0)


def geometric_matte(roi_bgr: np.ndarray) -> np.ndarray:
    """Solid mask of the largest non-white, non-red blob (its convex hull)."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    not_white = gray < 235
    red = red_mask(roi_bgr) > 0
    fg = (not_white & ~red).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(roi_bgr.shape[:2], np.uint8)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(out, [cv2.convexHull(c)], -1, 255, -1)
    return out


def grabcut_matte(roi_bgr: np.ndarray) -> np.ndarray:
    """GrabCut seeded by a rectangle just inside the ROI edges."""
    h, w = roi_bgr.shape[:2]
    if h < 20 or w < 20:
        return None
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    inset_x, inset_y = max(2, w // 25), max(2, h // 25)
    rect = (inset_x, inset_y, w - 2 * inset_x, h - 2 * inset_y)
    try:
        cv2.grabCut(roi_bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0)
    return alpha.astype(np.uint8)


def rembg_matte(roi_bgr: np.ndarray, model: str = "isnet-general-use"):
    """Alpha matte from rembg, or None if rembg/model unavailable."""
    try:
        from rembg import remove, new_session
        from PIL import Image
    except Exception:
        return None
    try:
        if model not in _REMBG_SESSIONS:
            _REMBG_SESSIONS[model] = new_session(model)
        pil = Image.fromarray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
        out = remove(pil, session=_REMBG_SESSIONS[model])
        return np.asarray(out.convert("RGBA").getchannel("A"))
    except Exception:
        return None


def sam_matte(roi_bgr: np.ndarray, checkpoint: str = ""):
    """Alpha matte from MobileSAM, prompted by a box covering the ROI.

    Returns None if mobile_sam or the checkpoint is unavailable.
    """
    global _SAM_PREDICTOR
    try:
        import torch
        from mobile_sam import sam_model_registry, SamPredictor
    except Exception:
        return None
    import os
    if not checkpoint or not os.path.exists(checkpoint):
        return None
    try:
        if _SAM_PREDICTOR is None:
            sam = sam_model_registry["vit_t"](checkpoint=checkpoint)
            sam.eval()
            _SAM_PREDICTOR = SamPredictor(sam)
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        _SAM_PREDICTOR.set_image(rgb)
        h, w = roi_bgr.shape[:2]
        box = np.array([2, 2, w - 2, h - 2])
        masks, scores, _ = _SAM_PREDICTOR.predict(box=box, multimask_output=True)
        best = masks[int(np.argmax(scores))]
        return (best.astype(np.uint8) * 255)
    except Exception:
        return None


def _run_engine(name, roi_bgr, model, checkpoint):
    if name == "rembg":
        return rembg_matte(roi_bgr, model)
    if name == "sam":
        return sam_matte(roi_bgr, checkpoint)
    if name == "grabcut":
        return grabcut_matte(roi_bgr)
    if name == "geometric":
        return geometric_matte(roi_bgr)
    return None


def make_cutout(bgr, quad, engine: str, feather_px: int,
                rembg_model: str = "isnet-general-use", sam_checkpoint: str = ""):
    """Return (rgba uint8 HxWx4 tight to alpha bbox, method)."""
    if engine == "exact":
        rgba = exact_quad_cutout(bgr, quad)
        if rgba is not None:
            return rgba, "exact"
        # degenerate selection -> fall back to a plain matte rather than crash

    x0, y0, x1, y1 = roi_bounds(quad, bgr.shape)
    roi = bgr[y0:y1, x0:x1].copy()

    alpha, method = None, None
    for name in ENGINE_LADDER.get(engine, ["geometric"]):
        a = _run_engine(name, roi, rembg_model, sam_checkpoint)
        if a is not None and a.shape == roi.shape[:2] and validate_coverage(a):
            alpha, method = a, name
            break
    if alpha is None:
        alpha, method = geometric_matte(roi), "geometric"

    alpha = feather_alpha(alpha, feather_px)
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([rgb, alpha])

    ys, xs = np.where(alpha > 20)
    if xs.size and ys.size:
        rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return rgba, method
