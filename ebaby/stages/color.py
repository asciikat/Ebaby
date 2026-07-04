"""RAW colour correction — ported from Colorprodawgv1.py.

White-balances each photo using the paper that SURROUNDS the DVD case (only
border-touching white blobs; print inside the cover art is never mistaken
for the neutral reference), then applies a mild S-curve/saturation/sharpen
enhancement. Runs before barcode scanning and rename so downstream stages
see clean, colour-corrected images.
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


def get_surrounding_white_mask(bgr_img: np.ndarray) -> np.ndarray:
    h, w = bgr_img.shape[:2]
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
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
    mask = get_surrounding_white_mask(bgr_img)
    if mask.sum() == 0:
        return 1.0, 1.0, 1.0
    mean_b = cv2.mean(bgr_img[:, :, 0], mask=mask)[0]
    mean_g = cv2.mean(bgr_img[:, :, 1], mask=mask)[0]
    mean_r = cv2.mean(bgr_img[:, :, 2], mask=mask)[0]
    if mean_b == 0 or mean_g == 0 or mean_r == 0:
        return 1.0, 1.0, 1.0
    target = (mean_b + mean_g + mean_r) / 3.0
    return (
        float(np.clip(target / mean_b, 0.5, 2.0)),
        float(np.clip(target / mean_g, 0.5, 2.0)),
        float(np.clip(target / mean_r, 0.5, 2.0)),
    )


def apply_white_balance(bgr_img: np.ndarray, gains) -> np.ndarray:
    b, g, r = cv2.split(bgr_img.astype("float32"))
    b *= gains[0]
    g *= gains[1]
    r *= gains[2]
    return np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)


def create_s_curve_lut(strength: float = 1.0) -> np.ndarray:
    if strength == 0.0:
        return np.arange(256, dtype=np.uint8)
    x = np.arange(256)
    nx = (x / 255.0 - 0.5) * 2
    y = 1 / (1 + np.exp(-strength * nx))
    y = (y - y.min()) / (y.max() - y.min()) * 255
    return np.clip(y, 0, 255).astype(np.uint8)


def enhance_image(bgr_img: np.ndarray, contrast: float = 1.0,
                   saturation: float = 1.1, sharpen: float = 1.2) -> np.ndarray:
    img_contrast = cv2.LUT(bgr_img, create_s_curve_lut(contrast))
    if saturation != 1.0:
        hsv = cv2.cvtColor(img_contrast, cv2.COLOR_BGR2HSV).astype("float32")
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
        img_color = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    else:
        img_color = img_contrast
    if sharpen > 1.0:
        blurred = cv2.GaussianBlur(img_color, (0, 0), 3.0)
        factor = sharpen - 1.0
        return cv2.addWeighted(img_color, 1.0 + factor, blurred, -factor, 0)
    return img_color


def process_raw(raw_path, denoise: str = "light") -> np.ndarray:
    fbdd, noise_thr, median_passes = _denoise_levels().get(denoise, _denoise_levels()["light"])
    kwargs = dict(use_camera_wb=True, fbdd_noise_reduction=fbdd, median_filter_passes=median_passes)
    if noise_thr is not None:
        kwargs["noise_thr"] = noise_thr
    with rawpy.imread(str(raw_path)) as raw:
        rgb = raw.postprocess(**kwargs)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def color_correct_file(raw_path, out_path, denoise: str = "light",
                        contrast: float = 1.0, saturation: float = 1.1,
                        sharpen: float = 1.2, gains=None) -> None:
    """Decode one RAW file, white-balance + enhance, save as PNG at out_path."""
    bgr = process_raw(raw_path, denoise=denoise)
    current_gains = gains if gains is not None else get_white_balance_gains(bgr)
    corrected = apply_white_balance(bgr, current_gains)
    final = enhance_image(corrected, contrast, saturation, sharpen)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), final)
