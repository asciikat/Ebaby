"""Detect, deskew, and tighten a DVD case from a phone photo.

This is the core of the rebuilt pipeline. Instead of locating the A4 sheet and
cropping a fixed rectangle (fragile: paper isn't always flat or fully in frame),
we segment the case directly with rembg, fit a *rotated* rectangle to it, and
perspective-warp that rectangle flat. One step buys deskew + a tight crop +
clean edges — no paper, no table shadow, no pencil marks.

The rembg alpha is carried through the warp as a 4th channel so the final
composite can drop any non-case pixel (e.g. dark table that sneaks into the
corners of an open case's rotated rectangle) to white.

Closed cases (front/back) are then trimmed to the exact DVD aspect ratio
(0.711) which removes the transparent sleeve flap that rembg includes as part of
the object. Open cases (inside) get a light empty-border trim.
"""
import os
import shutil

import cv2
import numpy as np
import pytesseract

from . import cutout


def _locate_tesseract():
    """Point pytesseract at the tesseract binary if it isn't already on PATH.

    The Windows installer (winget / UB-Mannheim) drops the exe in Program Files
    but doesn't always add it to PATH, so the OCR flip-vote, face word-count and
    title fallback would silently no-op. Probe the standard install locations so
    the app works out of the box. A no-op on Linux/WSL where it's on PATH."""
    if shutil.which("tesseract"):
        return
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return


_locate_tesseract()

# A closed DVD case is 13.5 cm wide x 19.0 cm tall -> W/H = 0.711.
DVD_CLOSED_RATIO = 0.711

_PROC_EDGE = 1000   # downscale long edge for rembg (speed; mask is then upscaled)


def _order_pts(pts):
    pts = np.array(pts, np.float32)
    s = pts.sum(1)
    d = np.diff(pts, 1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def _warp_to_quad(img, box):
    box = _order_pts(box)
    tl, tr, br, bl = box
    W = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    H = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if W < 10 or H < 10:
        return None
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(box, dst)
    return cv2.warpPerspective(img, M, (W, H))


def _empty_mask(bgr):
    """True where a pixel is bare paper or clear plastic-over-paper (no print).

    Bright + low-saturation + low local edge energy. Catches the sleeve flap and
    any paper margin while leaving printed artwork (even dark or white-on-dark)
    as content.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    edge = cv2.blur(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)), (11, 11))
    return (v > 135) & (s < 50) & (edge < 12)


def _content_profile(bgra, axis):
    content = (~_empty_mask(bgra[:, :, :3])).astype(np.float32)
    return content.mean(axis)


def _fit_width(bgra, ratio=DVD_CLOSED_RATIO):
    """Crop width to height*ratio, positioned to capture the printed cover.

    The case height is detected reliably; only width is inflated by the clear
    sleeve flap. Slide a correct-width window to where the printed content is.
    """
    H, W = bgra.shape[:2]
    target = int(round(H * ratio))
    if target >= W:
        return bgra
    col = _content_profile(bgra, axis=0)
    csum = np.cumsum(np.concatenate([[0.0], col]))
    best, bx = -1.0, 0
    for x in range(W - target + 1):
        v = csum[x + target] - csum[x]
        if v > best:
            best, bx = v, x
    return bgra[:, bx:bx + target]


def _longest_run(profile, thr):
    good = profile > thr
    best_len = best_start = 0
    cur = None
    for i, v in enumerate(good):
        if v and cur is None:
            cur = i
        if (not v or i == len(good) - 1) and cur is not None:
            end = i + 1 if v else i
            if end - cur > best_len:
                best_len, best_start = end - cur, cur
            cur = None
    return best_start, best_start + best_len


def _trim_empty_border(bgra, thr=0.4, pad_frac=0.012):
    """Crop away pure-empty (paper) margins around the content. Conservative."""
    H, W = bgra.shape[:2]
    y0, y1 = _longest_run(_content_profile(bgra, axis=1), thr)
    x0, x1 = _longest_run(_content_profile(bgra, axis=0), thr)
    if y1 - y0 < H * 0.4 or x1 - x0 < W * 0.4:
        return bgra    # detection looked wrong; don't risk over-cropping
    ph, pw = int(pad_frac * H), int(pad_frac * W)
    return bgra[max(0, y0 - ph):min(H, y1 + ph), max(0, x0 - pw):min(W, x1 + pw)]


def scan_case(bgr, rembg_model="isnet-general-use"):
    """Segment + deskew the case. Returns (warped_bgra_portrait, coverage) or None.

    The warp is returned in canonical portrait (long side vertical) as a 4-channel
    BGRA array: the rembg alpha rides along so the composite can whiten any
    non-case pixel. `coverage` is the case area as a fraction of the frame — used
    to tell the (large) open case apart from the (smaller) closed cases.
    """
    h, w = bgr.shape[:2]
    scale = _PROC_EDGE / float(max(h, w))
    small = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))))

    alpha_small = cutout.rembg_matte(small, rembg_model)
    if alpha_small is None or alpha_small.shape != small.shape[:2]:
        return None

    _, m = cv2.threshold(alpha_small, 180, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)), iterations=3)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    coverage = float(cv2.contourArea(c)) / float(small.shape[0] * small.shape[1])
    if not (0.06 <= coverage <= 0.99):
        return None

    box = cv2.boxPoints(cv2.minAreaRect(c)) / scale     # full-resolution corners
    alpha_full = cv2.resize(alpha_small, (w, h), interpolation=cv2.INTER_LINEAR)
    bgra = np.dstack([bgr, alpha_full])
    warped = _warp_to_quad(bgra, box)
    if warped is None:
        return None
    if warped.shape[1] > warped.shape[0]:               # canonical portrait
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped, coverage


def orient_and_tighten(warped_portrait, face):
    """Orient for the face and tighten the crop. Input/-output are BGRA."""
    from .models import Face
    if face == Face.INSIDE:
        img = cv2.rotate(warped_portrait, cv2.ROTATE_90_CLOCKWISE)   # -> landscape
        return _trim_empty_border(img)
    return _fit_width(warped_portrait, DVD_CLOSED_RATIO)


_OSD_MIN_CONF = 2.0   # below this, OSD is noise on these covers; defer to the VLM
_VOTE_MIN_STRONG = 4  # need at least this many confident words to trust the vote


def _text_strength(bgr):
    """How many big, confident words Tesseract finds. Upright text scores far
    higher than upside-down text — the basis of `flip_by_text`."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    if max(gray.shape) < 1400:                       # OCR likes a bit of size
        f = 1400.0 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=f, fy=f)
    try:
        d = pytesseract.image_to_data(gray, config="--psm 3",
                                      output_type=pytesseract.Output.DICT)
    except Exception:
        return 0
    return sum(1 for i, c in enumerate(d["conf"])
               if int(c) >= 60 and len(d["text"][i].strip()) >= 3)


def flip_by_text(bgr):
    """Should this cover be flipped 180°?  By OCR confidence, or None if unsure.

    Scores the image and its 180° rotation; printed text reads with many more
    confident words the right way up. Returns True (flip), False (keep), or None
    when there's too little text to tell (e.g. a disc/inside face) so the caller
    can fall back to the vision model. This is the most reliable signal on the
    text-dense back cover, where OSD and the small VLM both waver.
    """
    if bgr.ndim == 3 and bgr.shape[2] == 4:
        bgr = bgr[:, :, :3]
    bgr = np.ascontiguousarray(bgr)
    up = _text_strength(bgr)
    down = _text_strength(cv2.rotate(bgr, cv2.ROTATE_180))
    if max(up, down) < _VOTE_MIN_STRONG or abs(up - down) < 2:
        return None
    return down > up


def osd_rotation(bgr):
    """Confident upright/upside-down verdict from Tesseract OSD, or None.

    Returns 0 (upright), 180 (upside-down), or None when OSD can't decide. We
    only trust 0/180 above a confidence floor — on real covers OSD nails the
    text-dense faces but fails or guesses on disc/inside faces, so an unsure
    result hands off to the vision model rather than risking a wrong flip.

    The deskew already fixes 90° tilt, so a 90/270 verdict means OSD is confused
    -> None. Run this on the high-res deskewed cover (largest legible text).
    """
    if bgr.ndim == 3 and bgr.shape[2] == 4:
        bgr = bgr[:, :, :3]
    bgr = np.ascontiguousarray(bgr)
    try:
        osd = pytesseract.image_to_osd(bgr, output_type=pytesseract.Output.DICT)
    except Exception:
        return None
    rotate = int(osd.get("rotate", 0))
    conf = float(osd.get("orientation_conf", 0.0))
    if conf < _OSD_MIN_CONF or rotate not in (0, 180):
        return None
    return rotate


def clean_mask(alpha):
    """Turn a carried rembg alpha into a solid, hole-free case mask.

    Threshold, keep the largest blob filled solid (so glossy interior never
    punches a hole), then feather the edge slightly. Falls back to fully opaque
    if nothing survives, so a shot is never blanked out.
    """
    _, m = cv2.threshold(alpha, 110, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.full_like(alpha, 255)
    solid = np.zeros_like(alpha)
    cv2.drawContours(solid, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return cv2.GaussianBlur(solid, (7, 7), 0)
