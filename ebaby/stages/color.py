"""RAW colour correction — ported from Colorprodawgv1.py, then reworked for
listing-photo quality.

Pipeline per photo:
  1. decode RAW at 16-bit and carry everything in float32 [0,255] until the
     final save — no 8-bit rounding between steps, so smooth gradients don't
     band and highlights don't clip early;
  2. white-balance off the paper that SURROUNDS the case (border-touching
     low-saturation blobs only — cover art is never mistaken for neutral),
     using the MEDIAN so a stray dark edge in the mask can't skew the cast;
  3. lift the whole frame so the paper reads bright white, with a soft
     highlight roll-off so glossy specular reflections compress toward white
     instead of clipping to a flat blob;
  4. midtone gamma lift — opens up dark cover art without washing out the
     paper or greying the blacks;
  5. pop: gentle S-curve contrast, VIBRANCE (protects already-saturated cover
     colours instead of a flat saturation multiply that shifts hues), and a
     small-radius threshold-gated unsharp that crisps text without haloing the
     case edge or amplifying paper noise.

Runs before barcode scanning and rename so downstream stages see clean images.
"""
import cv2
import numpy as np
import rawpy


def _denoise_levels():
    return {
        "off": (rawpy.FBDDNoiseReductionMode.Off, None, 0),
        "light": (rawpy.FBDDNoiseReductionMode.Full, None, 1),
        "medium": (rawpy.FBDDNoiseReductionMode.Full, 100.0, 1),
        "strong": (rawpy.FBDDNoiseReductionMode.Full, 250.0, 2),
    }


def _as_float(img):
    return img if img.dtype == np.float32 else img.astype(np.float32)


def _match_dtype(f, ref):
    """Return float untouched when the caller works in float (the pipeline),
    or clip+round back to the reference's integer dtype (standalone/tests)."""
    if ref.dtype == np.float32:
        return f
    return np.clip(f, 0, 255).astype(ref.dtype)


def _u8(img):
    return img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)


def get_surrounding_white_mask(bgr_img: np.ndarray) -> np.ndarray:
    u = _u8(bgr_img)
    h, w = u.shape[:2]
    hsv = cv2.cvtColor(u, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    kernel = np.ones((5, 5), np.uint8)
    for v_thr, s_thr in ((150, 45), (120, 60), (100, 80)):
        white = ((v > v_thr) & (s < s_thr)).astype(np.uint8)
        if white.sum() == 0:
            continue
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)
        num, labels = cv2.connectedComponents(white)
        border_labels = set()
        border_labels.update(labels[0, :].tolist())
        border_labels.update(labels[-1, :].tolist())
        border_labels.update(labels[:, 0].tolist())
        border_labels.update(labels[:, -1].tolist())
        border_labels.discard(0)
        if not border_labels:
            continue
        surrounding = np.isin(labels, list(border_labels)).astype(np.uint8) * 255
        if (surrounding.sum() / 255) > (h * w * 0.02):
            return surrounding
    return ((v > 120) & (s < 60)).astype(np.uint8) * 255


def get_white_balance_gains(bgr_img: np.ndarray):
    """Per-channel gains that make the surrounding paper neutral. Uses the
    MEDIAN of the paper pixels, not the mean, so a little cover-art or table
    bleed into the mask can't drag the cast off."""
    mask = get_surrounding_white_mask(bgr_img)
    if mask.sum() == 0:
        return 1.0, 1.0, 1.0
    sel = mask > 0
    med_b = float(np.median(bgr_img[:, :, 0][sel]))
    med_g = float(np.median(bgr_img[:, :, 1][sel]))
    med_r = float(np.median(bgr_img[:, :, 2][sel]))
    if med_b == 0 or med_g == 0 or med_r == 0:
        return 1.0, 1.0, 1.0
    target = (med_b + med_g + med_r) / 3.0
    return (
        float(np.clip(target / med_b, 0.5, 2.0)),
        float(np.clip(target / med_g, 0.5, 2.0)),
        float(np.clip(target / med_r, 0.5, 2.0)),
    )


def apply_white_balance(bgr_img: np.ndarray, gains) -> np.ndarray:
    f = _as_float(bgr_img) * np.array(gains, dtype=np.float32)
    return _match_dtype(f, bgr_img)


def lift_whites(bgr_img: np.ndarray, target: float = 247.0,
                max_gain: float = 1.35, knee: float = 248.0) -> np.ndarray:
    """Brighten the whole frame so the surrounding PAPER reads bright white,
    lifting the cover with it. Gain is capped and anchored to the paper's
    median brightness, so a correctly-exposed shot barely moves. A soft knee
    above `knee` compresses the very brightest values toward 255 instead of
    hard-clipping — glossy specular highlights keep their shape."""
    mask = get_surrounding_white_mask(bgr_img)
    if mask.sum() == 0:
        return bgr_img
    v = cv2.cvtColor(_u8(bgr_img), cv2.COLOR_BGR2HSV)[:, :, 2]
    paper = float(np.median(v[mask > 0]))
    if paper <= 0 or paper >= target:
        return bgr_img
    gain = min(target / paper, max_gain)
    f = _as_float(bgr_img) * gain
    span = 255.0 - knee
    if span > 0:
        hi = f > knee
        f[hi] = knee + span * (1.0 - np.exp(-(f[hi] - knee) / span))
    return _match_dtype(f, bgr_img)


def lift_midtones(bgr_img: np.ndarray, gamma: float = 0.82) -> np.ndarray:
    """Gamma lift: opens up dark cover art (the murky-purple problem) while
    pinning black and white endpoints, so it brightens the BODY of the image
    without washing out the paper or greying the blacks."""
    if gamma == 1.0:
        return bgr_img
    f = 255.0 * np.power(np.clip(_as_float(bgr_img), 0, 255) / 255.0, gamma)
    return _match_dtype(f, bgr_img)


def _s_curve_float(f, strength):
    """Sigmoid contrast in float, endpoints pinned to [0,255] (same shape the
    old uint8 LUT produced, without the 256-step quantisation)."""
    if strength == 0.0:
        return f
    nx = (f / 255.0 - 0.5) * 2.0
    y = 1.0 / (1.0 + np.exp(-strength * nx))
    y0 = 1.0 / (1.0 + np.exp(strength))    # value at nx = -1
    y1 = 1.0 / (1.0 + np.exp(-strength))   # value at nx = +1
    return (y - y0) / (y1 - y0) * 255.0


def _vibrance(f, amount):
    """Saturation boost that PROTECTS already-vivid pixels: the push shrinks
    as a pixel's existing saturation grows, so neon cover colours get punchier
    without clipping or hue-shifting the way a flat HSV multiply does."""
    if amount == 0.0:
        return f
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    lum = (0.114 * b + 0.587 * g + 0.299 * r)[..., None]
    mx = np.maximum(np.maximum(b, g), r)
    mn = np.minimum(np.minimum(b, g), r)
    sat = ((mx - mn) / 255.0)[..., None]        # current saturation proxy
    scale = 1.0 + amount * (1.0 - sat)          # gentler where already vivid
    return lum + (f - lum) * scale


def _unsharp(f, sharpen, radius=1.2, threshold=2.0):
    """Small-radius unsharp mask. The small radius crisps back-cover text
    without the halo a wide radius throws around the hard case/paper edge; the
    threshold leaves flat paper (and its sensor noise) untouched."""
    if sharpen <= 1.0:
        return f
    blurred = cv2.GaussianBlur(f, (0, 0), radius)
    high = f - blurred
    if threshold > 0:
        high = np.where(np.abs(high) < threshold, 0.0, high)
    return f + high * (sharpen - 1.0)


def enhance_image(bgr_img: np.ndarray, contrast: float = 1.0,
                   saturation: float = 1.1, sharpen: float = 1.2) -> np.ndarray:
    # exact identity when every stage is neutral (callers/tests rely on it)
    if contrast == 0.0 and saturation == 1.0 and sharpen <= 1.0:
        return bgr_img
    f = _as_float(bgr_img)
    f = _s_curve_float(f, contrast)
    f = _vibrance(f, saturation - 1.0)
    f = _unsharp(f, sharpen)
    return _match_dtype(f, bgr_img)


# Listing-photo look, tuned on a real dark cover (Goober back, 2026-07-06):
# strong paper lift toward clean white, a midtone gamma that opens up murky
# cover art, vibrance run hot (it protects already-vivid pixels so it can),
# contrast eased so the gamma lift isn't crushed straight back down.
POP_CONTRAST = 1.3
POP_SATURATION = 1.4
POP_SHARPEN = 1.5
LIFT_TARGET = 250.0
LIFT_MAX_GAIN = 1.6
MIDTONE_GAMMA = 0.82


def process_raw(raw_path, denoise: str = "light") -> np.ndarray:
    """Decode a RAW to float32 BGR in [0,255], at 16-bit precision so the
    downstream float pipeline has real headroom (smooth gradients, recoverable
    highlights) instead of 8-bit stair-steps."""
    fbdd, noise_thr, median_passes = _denoise_levels().get(denoise, _denoise_levels()["light"])
    kwargs = dict(use_camera_wb=True, output_bps=16,
                  fbdd_noise_reduction=fbdd, median_filter_passes=median_passes)
    if noise_thr is not None:
        kwargs["noise_thr"] = noise_thr
    with rawpy.imread(str(raw_path)) as raw:
        rgb16 = raw.postprocess(**kwargs)              # uint16 [0,65535]
        bgr16 = cv2.cvtColor(rgb16, cv2.COLOR_RGB2BGR)
        return bgr16.astype(np.float32) * (255.0 / 65535.0)


def _finish(f, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.clip(f, 0, 255).astype(np.uint8))


def color_correct_file(raw_path, out_path, denoise: str = "light",
                        contrast: float = POP_CONTRAST,
                        saturation: float = POP_SATURATION,
                        sharpen: float = POP_SHARPEN, gains=None):
    """Decode one RAW, white-balance + white-lift + pop, save as PNG. Returns
    the white-balance gains used, so every shot of one disc can share a single
    balance (pass it back in via `gains=`) for a consistent-looking set."""
    bgr = process_raw(raw_path, denoise=denoise)       # float32 [0,255]
    current_gains = gains if gains is not None else get_white_balance_gains(bgr)
    f = apply_white_balance(bgr, current_gains)
    f = lift_whites(f, target=LIFT_TARGET, max_gain=LIFT_MAX_GAIN)
    f = lift_midtones(f, MIDTONE_GAMMA)
    f = enhance_image(f, contrast, saturation, sharpen)
    _finish(f, out_path)
    return current_gains


def color_correct_plain_file(src_path, out_path, contrast: float = POP_CONTRAST,
                              saturation: float = POP_SATURATION,
                              sharpen: float = POP_SHARPEN, gains=None):
    """Same treatment for JPG/PNG uploads so phone-JPEG batches get the
    identical look RAW batches do. Returns the gains used (truthy) or None if
    the file couldn't be read."""
    bgr = cv2.imread(str(src_path))
    if bgr is None:
        return None
    f = bgr.astype(np.float32)
    current_gains = gains if gains is not None else get_white_balance_gains(f)
    f = apply_white_balance(f, current_gains)
    f = lift_whites(f, target=LIFT_TARGET, max_gain=LIFT_MAX_GAIN)
    f = lift_midtones(f, MIDTONE_GAMMA)
    f = enhance_image(f, contrast, saturation, sharpen)
    _finish(f, out_path)
    return current_gains
