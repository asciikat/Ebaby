#!/usr/bin/env python3
"""DVD Photo Processor V2 — A4-reference pipeline for eBay listings.

Designed for top-down phone photos of a DVD laid on a white A4 sheet (the
sheet may sit on a busy/reflective surface and need not fill the frame).

Pipeline per image:
    1. Load  — RAW .dng via rawpy (honours orientation); JPG/PNG/TIFF/HEIC via PIL.
    2. Detect the A4 sheet (largest near-A4 bright quad) and perspective-warp
       the frame flat to true A4 proportions  → deskew.
    3. White-balance using the A4 paper itself as the neutral-white reference,
       and lift exposure so the paper reads clean white.
    4. Segment the DVD off the white paper and model it as a straight-edged
       rotated rectangle (no jagged contour tracing), then deskew it upright.
    5. Classify  — Center (open disc-tray spread, much larger than a cover),
       Back (barcode present) or Front (single cover, no barcode).
    6. Composite on a white square canvas and save; write listing TXT/CSV.

Run on Ubuntu / WSL2:
    python3 -m venv ~/ebay-venv/venv && source ~/ebay-venv/venv/bin/activate
    pip install numpy opencv-python-headless rawpy pillow pillow-heif pyzbar
    # system: sudo apt install libzbar0 python3-tk    (tk only for the GUI)
    python3 ebayflip_V2.py            # batch the default 'Images in' folder
    python3 ebayflip_V2.py --debug    # also dump per-stage diagnostic images
"""

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("Missing dependency: opencv. Install: pip install opencv-python-headless numpy")

try:
    from PIL import Image, ImageOps
except ImportError:
    sys.exit("Missing dependency: pillow. Install: pip install pillow")

try:
    import rawpy
    RAWPY_OK = True
except ImportError:
    RAWPY_OK = False

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_OK = True
except ImportError:
    HEIC_OK = False

try:
    import pytesseract
    # Probe the tesseract binary once; disable cleanly if absent.
    pytesseract.get_tesseract_version()
    TESS_OK = True
except Exception:
    TESS_OK = False


# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "Images in"
DEFAULT_OUTPUT = SCRIPT_DIR / "processed"

RAW_EXT = {".dng"}
PIL_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
HEIC_EXT = {".heic", ".heif"}
SUPPORTED_EXT = RAW_EXT | PIL_EXT | HEIC_EXT

# A4 is 210 x 297 mm.  We warp the detected sheet to this at PX_PER_MM
# resolution so that, post-warp, pixel sizes map directly to millimetres —
# which makes the Front/Back vs Center size test physical and reliable.
A4_SHORT_MM = 210.0
A4_LONG_MM = 297.0
A4_RATIO = A4_LONG_MM / A4_SHORT_MM        # 1.414
PX_PER_MM = 10                              # warped A4 = 2970 x 2100 px

# A4-detection acceptance.  >= AUTO: trust it; >= REVIEW: usable but flag for
# the GUI review fallback; below REVIEW: ask for manual corners (GUI) / skip (CLI).
AUTO_CONFIDENCE = 0.72
REVIEW_CONFIDENCE = 0.45

# A DVD case face is ~135 x 190 mm; an open case spans ~190 x 273 mm.  The long
# edge cleanly separates the two populations at ~230 mm.
CENTER_LONG_MM = 230.0

VIEW_LABELS = {"front": "Front cover", "back": "Back cover",
               "center": "Open case / disc tray", "other": "Other"}


# ============================================================
# Small utilities
# ============================================================

def now_stamp():
    return time.strftime("%Y%m%d_%H%M%S")


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


def gather_inputs(folder, exclude_dir=None):
    folder = Path(folder)
    out = []
    try:
        exclude = Path(exclude_dir).resolve() if exclude_dir else None
    except Exception:
        exclude = None
    if not folder.is_dir():
        return out
    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf not in SUPPORTED_EXT:
            continue
        if suf in RAW_EXT and not RAWPY_OK:
            continue
        if suf in HEIC_EXT and not HEIC_OK:
            continue
        if exclude is not None:
            try:
                p.resolve().relative_to(exclude)
                continue
            except (ValueError, OSError):
                pass
        out.append(p)
    return out


# ============================================================
# Stage 1 — Loading
# ============================================================

def load_bgr(path):
    """Load any supported image to a BGR uint8 array, orientation-corrected."""
    path = Path(path)
    if path.suffix.lower() in RAW_EXT:
        if not RAWPY_OK:
            raise RuntimeError("rawpy not installed; cannot read .dng")
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,      # sane starting WB; we refine via paper
                no_auto_bright=True,     # keep linear-ish; we set exposure ourselves
                output_bps=8,
                output_color=rawpy.ColorSpace.sRGB,
                user_flip=-1,            # apply the orientation tag from the file
            )
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    with Image.open(path) as im:
        im.load()
        rgb = ImageOps.exif_transpose(im).convert("RGB")
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


# ============================================================
# Stage 2 — A4 detection + flatten
# ============================================================

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

    # Paper is the bright extreme and near-neutral. A low fraction of the bright
    # end captures even dim, shadowed margins (e.g. the thin frame around a big
    # black open case) while the much darker background stays out.
    thr = max(0.40 * float(L.max()), float(otsu) * 0.80)
    mask = ((Lb >= thr) & (s <= 95)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # Close hard to knit the paper frame into one ring across the dark DVD.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    if dump is not None:
        cv2.imwrite(dump, mask)

    # Select by ENCLOSED area (a thin frame's outer contour still encloses the
    # whole sheet), then model the rectangle with minAreaRect.
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
    """Perspective-warp the sheet flat to true A4 proportions."""
    rect = order_points(quad)
    tl, tr, br, bl = rect
    wid = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    hei = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    if wid >= hei:   # sheet captured in landscape
        W, H = int(A4_LONG_MM * px_per_mm), int(A4_SHORT_MM * px_per_mm)
    else:
        W, H = int(A4_SHORT_MM * px_per_mm), int(A4_LONG_MM * px_per_mm)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(bgr, M, (W, H), flags=cv2.INTER_CUBIC)


# ============================================================
# Stage 3 — White balance from the paper
# ============================================================

def paper_mask(flat_bgr):
    """Mask of the white-paper pixels in a flattened A4 (DVD excluded)."""
    L = cv2.cvtColor(flat_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    s = cv2.cvtColor(flat_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    bright = L >= max(np.percentile(L, 55), 0.6 * float(L.max()))
    neutral = s <= 45
    return bright & neutral


def white_balance_from_paper(flat_bgr, target=242.0):
    """Neutralise colour cast using the paper as white, then lift to `target`.

    Returns (balanced_bgr, ok). Falls back to grey-world if no paper found.
    """
    pm = paper_mask(flat_bgr)
    img = flat_bgr.astype(np.float32)
    if pm.sum() < 0.02 * pm.size:
        means = img.reshape(-1, 3).mean(axis=0)
        ok = False
    else:
        means = img[pm].mean(axis=0)   # B, G, R of paper
        ok = True
    means = np.maximum(means, 1.0)
    gains = means.mean() / means                      # neutralise cast
    bal = img * gains
    # Lift exposure so paper reads as clean (near-)white without blowing out.
    if ok:
        paper_after = (bal[pm].mean(axis=0)).mean()
    else:
        paper_after = np.percentile(bal, 95)
    if paper_after > 1.0:
        bal *= (target / paper_after)
    return np.clip(bal, 0, 255).astype(np.uint8), ok


# ============================================================
# Stage 4 — DVD segmentation (straight-edged rectangle)
# ============================================================

def segment_dvd(flat_bgr, dump=None):
    """Find the DVD on the white sheet → rotated rectangle box (4 pts) or None.

    Whiteness = min(B,G,R): paper is high on every channel, while the black
    case is dark and colourful cover art has at least one low channel. This is
    far more robust than a saturation test (bright paper carries sat. noise).
    """
    h, w = flat_bgr.shape[:2]
    minc = flat_bgr.min(axis=2)
    white_level = float(np.percentile(minc, 92))     # paper white reference
    thr = max(120.0, 0.72 * white_level)
    fg = (minc < thr).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    # Drop foreground touching the sheet border BEFORE closing: a slightly
    # oversized A4 warp leaves a dark glass ring/strip around the edge. The DVD
    # is separated from it by the white paper margin, so removing border-
    # connected components now keeps the DVD; closing first would bridge them.
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    cleaned = np.zeros_like(fg)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = (x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1)
        if not touches:
            cleaned[lab == i] = 255
    if cleaned.any():
        fg = cleaned
    # Now consolidate the DVD body (joins cover art across the case interior).
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
    box = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
    return box


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


def orient_for_view(dvd_bgr, view):
    """Front/Back → portrait; Center → landscape."""
    h, w = dvd_bgr.shape[:2]
    if view == "center":
        if h > w:
            return cv2.rotate(dvd_bgr, cv2.ROTATE_90_CLOCKWISE)
    else:
        if w > h:
            return cv2.rotate(dvd_bgr, cv2.ROTATE_90_CLOCKWISE)
    return dvd_bgr


def _ocr_text_score(bgr):
    """(summed word-confidence, word-count) for confidently-read text."""
    if not TESS_OK:
        return 0.0, 0
    try:
        d = pytesseract.image_to_data(bgr, output_type=pytesseract.Output.DICT,
                                      config="--psm 11 --dpi 300")
    except Exception:
        return 0.0, 0
    score, n = 0.0, 0
    for t, c in zip(d["text"], d["conf"]):
        try:
            c = float(c)
        except (TypeError, ValueError):
            c = -1.0
        if len(t.strip()) >= 3 and c > 40:
            score += c
            n += 1
    return score, n


def upright_vote(dvd_bgr):
    """Decide if a portrait/landscape crop is upside down, using readable text.

    Returns (needs_flip, confident). `needs_flip` True means rotate 180.
    Confident only when text clearly reads better one way than the other.
    """
    s0, n0 = _ocr_text_score(dvd_bgr)
    s180, n180 = _ocr_text_score(cv2.rotate(dvd_bgr, cv2.ROTATE_180))
    if max(n0, n180) >= 5 and abs(s0 - s180) > 0.25 * max(s0, s180, 1.0):
        return (s180 > s0), True
    return False, False


# ============================================================
# Stage 5 — Classification
# ============================================================

def scan_barcodes(bgr):
    """Return list of decoded barcode strings (multi-rotation). [] if none, None if unavailable."""
    try:
        from pyzbar.pyzbar import decode
    except ImportError:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    # Upscale small crops; barcodes need resolution.
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


def classify_view(box, has_barcode):
    """Center if physically large; else Back if barcode; else Front."""
    rrect_w = np.linalg.norm(order_points(box)[1] - order_points(box)[0])
    rrect_h = np.linalg.norm(order_points(box)[3] - order_points(box)[0])
    long_mm = max(rrect_w, rrect_h) / PX_PER_MM
    if long_mm >= CENTER_LONG_MM:
        return "center"
    return "back" if has_barcode else "front"


# ============================================================
# Stage 6 — Finish & compose
# ============================================================

def enhance(bgr, strength=0.6):
    """Gentle local-contrast + saturation pop (paper already white-balanced)."""
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


def composite_on_white(dvd_bgr, padding_pct=7, square=True):
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


# ============================================================
# Orchestration
# ============================================================

def process_one(path, out_dir, settings, log=print, debug_dir=None):
    name = Path(path).stem

    def dbg(tag):
        return str(Path(debug_dir) / f"{name}_{tag}.png") if debug_dir else None

    bgr = load_bgr(path)
    quad, conf = detect_a4(bgr, dump=dbg("a4mask"))
    if quad is None or conf < REVIEW_CONFIDENCE:
        log(f"  {Path(path).name}: A4 not found (conf {conf:.2f}) — skipped")
        return None

    flat = warp_to_a4(bgr, quad)
    if debug_dir:
        cv2.imwrite(dbg("flat"), resize_max(flat, 1200))

    flat, wb_ok = white_balance_from_paper(flat)
    if debug_dir:
        cv2.imwrite(dbg("balanced"), resize_max(flat, 1200))

    box = segment_dvd(flat, dump=dbg("dvdmask"))
    if box is None:
        log(f"  {Path(path).name}: DVD not found on sheet — skipped")
        return None

    dvd = crop_rect(flat, box)
    barcodes = scan_barcodes(dvd) or []
    view = classify_view(box, has_barcode=bool(barcodes))
    dvd = orient_for_view(dvd, view)

    # Resolve the 180 ambiguity from readable text where possible; ambiguous
    # crops (stylised titles, blank trays) are reconciled by run consensus.
    needs_flip, orient_conf = upright_vote(dvd)
    if orient_conf and needs_flip:
        dvd = cv2.rotate(dvd, cv2.ROTATE_180)

    if settings.get("enhance", True):
        dvd = enhance(dvd, settings.get("enhance_strength", 0.6))

    final = composite_on_white(dvd, settings.get("padding", 7), settings.get("square", True))
    final = resize_max(final, settings.get("max_dim", 1600))

    ext = ".jpg"
    out_path = out_dir / f"{name}_{view}{ext}"
    n = 2
    while out_path.exists():
        out_path = out_dir / f"{name}_{view}_{n:02d}{ext}"
        n += 1
    cv2.imwrite(str(out_path), final,
                [cv2.IMWRITE_JPEG_QUALITY, settings.get("quality", 92)])

    log(f"  {Path(path).name} -> {out_path.name}  [{view}]  conf {conf:.2f}"
        f"{'  barcode ' + barcodes[0] if barcodes else ''}"
        f"{'' if orient_conf else '  (orient: by consensus)'}"
        f"{'' if wb_ok else '  (grey-world WB)'}")
    return {"input": Path(path).name, "output": out_path.name, "view": view,
            "barcodes": barcodes, "a4_confidence": round(conf, 3),
            "white_balanced_from_paper": bool(wb_ok),
            "path": out_path, "orient_flip": bool(needs_flip),
            "orient_confident": bool(orient_conf)}


def apply_orientation_consensus(results, log=print):
    """Flip text-ambiguous crops to match the run's confident-orientation vote.

    All shots in a run share one capture orientation, so a confidently-read
    cover (usually the text-heavy back) tells us which way the rest sit.
    """
    # Only covers share the case's print orientation; the open case (center)
    # has its own geometry, so the cover consensus must not be forced onto it.
    votes = [r["orient_flip"] for r in results
             if r.get("orient_confident") and r["view"] in ("front", "back")]
    if not votes:
        return
    consensus_flip = sum(votes) > len(votes) / 2.0
    if not consensus_flip:
        return
    for r in results:
        if r.get("orient_confident") or r["view"] not in ("front", "back"):
            continue
        p = r.get("path")
        if not p or not Path(p).exists():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        cv2.imwrite(str(p), cv2.rotate(img, cv2.ROTATE_180),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        log(f"  oriented {Path(p).name} by consensus (180)")


# ============================================================
# Listing files
# ============================================================

def group_and_write(results, out_dir):
    """Group by DVD (a new front/center starts a group) and write TXT + CSV."""
    groups, cur = [], None
    for it in results:
        v = it["view"]
        if cur is None or (v in ("front", "center") and cur.get("seen_" + v)):
            cur = {"items": [], "barcode": ""}
            groups.append(cur)
        cur["items"].append(it)
        cur["seen_" + v] = True
        if not cur["barcode"] and it["barcodes"]:
            cur["barcode"] = it["barcodes"][0]

    txt = out_dir / "batch_listing.txt"
    lines = ["DVD eBay listing draft", f"Generated: {time.strftime('%Y-%m-%d %H:%M')}",
             f"Items: {len(groups)}", ""]
    for i, g in enumerate(groups, 1):
        lines += ["=" * 50, f"DVD {i}", f"Barcode: {g['barcode']}", "", "Photos:"]
        for it in g["items"]:
            lines.append(f"  {it['output']}  [{it['view']}]")
        lines += ["", "Condition: ", "Notes: ", "Price AUD: ", ""]
    txt.write_text("\n".join(lines), encoding="utf-8")

    csv_path = out_dir / "batch_listing.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["dvd_number", "barcode", "photos", "condition", "notes", "price_aud"])
        for i, g in enumerate(groups, 1):
            photos = "; ".join(f"{it['output']} [{it['view']}]" for it in g["items"])
            wtr.writerow([i, g["barcode"], photos, "", "", ""])
    return txt, csv_path, len(groups)


# ============================================================
# CLI
# ============================================================

def run_cli(args):
    in_dir = Path(args.input).expanduser()
    out_base = Path(args.output).expanduser()
    if not in_dir.is_dir():
        sys.exit(f"Input folder not found: {in_dir}")
    out_base.mkdir(parents=True, exist_ok=True)
    run_dir = out_base / f"run_{now_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = None
    if args.debug:
        debug_dir = run_dir / "_debug"
        debug_dir.mkdir(exist_ok=True)

    files = gather_inputs(in_dir, exclude_dir=out_base)
    print(f"Input : {in_dir}")
    print(f"Output: {run_dir}")
    print(f"Found {len(files)} image(s). rawpy={RAWPY_OK} heic={HEIC_OK}")
    if not files:
        return

    settings = {"enhance": not args.no_enhance, "enhance_strength": 0.6,
                "padding": args.padding, "square": not args.no_square,
                "max_dim": args.max_dim, "quality": args.quality}

    results, ok, fail, skip = [], 0, 0, 0
    for idx, path in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] {path.name}")
        try:
            r = process_one(path, run_dir, settings, log=print, debug_dir=debug_dir)
            if r is None:
                skip += 1
            else:
                results.append(r); ok += 1
        except Exception as e:
            fail += 1
            print(f"  FAILED: {e}", file=sys.stderr)
            if args.debug:
                traceback.print_exc()

    apply_orientation_consensus(results, log=print)

    if results and not args.no_listing:
        txt, csv_path, ngroups = group_and_write(results, run_dir)
        print(f"Wrote {txt.name} and {csv_path.name}  ({ngroups} DVD group(s))")

    (run_dir / "run_log.json").write_text(json.dumps(
        {"input": str(in_dir), "output": str(run_dir),
         "ok": ok, "skipped": skip, "failed": fail, "items": results},
        indent=2, default=str), encoding="utf-8")
    print(f"Done. {ok} ok, {skip} skipped, {fail} failed -> {run_dir}")


def main():
    p = argparse.ArgumentParser(description="DVD photo processor V2 (A4-reference).")
    p.add_argument("-i", "--input", default=str(DEFAULT_INPUT))
    p.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--debug", action="store_true", help="dump per-stage diagnostic images")
    p.add_argument("--no-enhance", action="store_true")
    p.add_argument("--no-square", action="store_true")
    p.add_argument("--no-listing", action="store_true")
    p.add_argument("--padding", type=int, default=7)
    p.add_argument("--max-dim", type=int, default=1600)
    p.add_argument("--quality", type=int, default=92)
    args = p.parse_args()
    run_cli(args)


if __name__ == "__main__":
    main()
