"""Batch orchestration: gather -> group -> process each shot -> save + listing."""
import time
from pathlib import Path

import cv2
import numpy as np

from . import barcode, clean, cutout, detect, extract, naming, ocr, scan, titles, vlm
from .imaging import load_image_bgr, resize_max_pil, save_jpeg
from .models import Face, FACE_ORDER, Field, HIGH, UNKNOWN, ShotResult, DvdGroup

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _front_title(cover_bgr, composed_pil, settings):
    """Read the front-cover title. Qwen (vision LLM) first, OCR as fallback.

    Used by the single-shot path and the no-extract batch path. The full-extract
    batch path reads the title inside the per-face Qwen scan instead.
    """
    engine = getattr(settings, "title_engine", "auto")
    if engine in ("auto", "qwen") and vlm.available():
        t = vlm.title_from_cover(cover_bgr, settings)
        if t or engine == "qwen":
            return t
    if engine == "qwen":
        return None
    return ocr.extract_title(composed_pil if composed_pil is not None else cover_bgr)


def gather_inputs(folder):
    folder = Path(folder)
    files = [p for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in SUPPORTED]
    return sorted(files, key=lambda p: p.name.lower())


def assign_groups(paths):
    """Slice into chunks of 3 (one DVD per chunk). Face assignment done later."""
    return [paths[i:i + len(FACE_ORDER)] for i in range(0, len(paths), len(FACE_ORDER))]


def _assign_faces_from_scans(scans, paths):
    """Assign faces from already-computed scans, by visual content.

    `scans` maps path -> (warped_portrait_bgr, coverage) or None.

    The open case (inside) is physically ~twice the size of a closed case, so it
    has by far the largest coverage of the frame -> INSIDE. Of the two closed
    faces, the back is densely printed (synopsis, credits, barcode) while the
    front is mostly artwork -> the one with more OCR words is BACK.

    Falls back to FACE_ORDER positional assignment when scans are missing or the
    chunk isn't a full set of three.
    """
    valid = [p for p in paths if scans.get(p) is not None]
    if len(paths) != len(FACE_ORDER) or len(valid) != len(FACE_ORDER):
        return {p: FACE_ORDER[i % len(FACE_ORDER)] for i, p in enumerate(paths)}

    coverage = {p: scans[p][1] for p in paths}
    inside = max(paths, key=lambda p: coverage[p])
    rest = [p for p in paths if p != inside]
    words = {p: ocr.count_words(scans[p][0][:, :, :3]) for p in rest}
    back = max(rest, key=lambda p: words[p])
    front = [p for p in rest if p != back][0]
    return {front: Face.FRONT, back: Face.BACK, inside: Face.INSIDE}


def _decide_flip(cover_bgra, settings, qwen_upright=None):
    """Should this oriented cover be rotated 180°?

    Priority: OCR-confidence vote (most reliable on text faces, free, no GPU) ->
    the vision model's upright verdict (covers disc/inside faces with little
    text) -> OSD as a last resort. `qwen_upright` lets a caller reuse a Qwen
    verdict it already has instead of paying for a second call.
    """
    if not getattr(settings, "auto_orient", True):
        return False
    bgr = np.ascontiguousarray(cover_bgra[:, :, :3])
    by_text = scan.flip_by_text(bgr)
    if by_text is not None:
        return by_text
    if qwen_upright is None and vlm.available():
        qwen_upright = vlm.is_upright(bgr, settings)
    if qwen_upright is not None:
        return not qwen_upright
    rot = scan.osd_rotation(bgr)
    if rot is not None:
        return rot == 180
    return False


def _compose_oriented(path, cover_bgra, face, settings, flip, extra_rotation, t0, ocr_title):
    """Compose a final square from an already-oriented cover (BGRA)."""
    img = cv2.rotate(cover_bgra, cv2.ROTATE_180) if flip else cover_bgra
    if extra_rotation:
        for _ in range((extra_rotation // 90) % 4):
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    bgr = np.ascontiguousarray(img[:, :, :3])
    mask = scan.clean_mask(img[:, :, 3])         # drops table corners etc. to white
    if settings.colour_tidy:
        bgr = clean.colour_tidy_bgr(bgr, 0.6)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    composed = clean.compose_on_white_square(np.dstack([rgb, mask]), settings.margin_pct)
    composed = resize_max_pil(composed, settings.max_edge_px)
    rotation = ((180 if flip else 0) + extra_rotation) % 360
    return composed, ShotResult(
        input_path=str(path), face=face, ocr_title=ocr_title,
        detect_method="rembg-warp", detect_conf=1.0,
        cutout_method="deskew", rotation=rotation, status="ok",
        elapsed_ms=int((time.perf_counter() - t0) * 1000),
    )


def _compose_from_scan(path, warped_portrait, face, settings, extra_rotation, t0,
                       qwen_upright=None):
    """Single-shot path: orient + auto-flip + (front) title + compose."""
    cover = scan.orient_and_tighten(warped_portrait, face)   # BGRA
    flip = _decide_flip(cover, settings, qwen_upright)
    cover_bgr = np.ascontiguousarray(cover[:, :, :3]).copy()  # pre-tidy, for title read
    ocr_title = _front_title(cover_bgr, None, settings) if face == Face.FRONT else None
    return _compose_oriented(path, cover, face, settings, flip, extra_rotation, t0, ocr_title)


def _compose_fallback(path, bgr, face, settings, manual_quad, extra_rotation, t0):
    """Legacy path: ROI detect -> matte engine -> compose. Used when the rembg
    deskew can't find the case, or when the user supplied a manual crop quad."""
    if manual_quad is not None:
        quad, conf, det_method = manual_quad, 1.0, "manual"
    else:
        quad, conf, det_method = detect.find_roi(bgr)

    rgba, cut_method = cutout.make_cutout(
        bgr, quad, settings.cutout_engine, settings.feather_px,
        rembg_model=settings.rembg_model, sam_checkpoint=settings.sam_checkpoint)
    rgba = clean.erase_red_to_white(rgba)

    ocr_title = _front_title(bgr, None, settings) if face == Face.FRONT else None

    rgba, rot = clean.auto_upright(rgba, face)
    rot = (rot + extra_rotation) % 360
    if extra_rotation:
        for _ in range((extra_rotation // 90) % 4):
            rgba = np.ascontiguousarray(np.rot90(rgba, k=-1))

    if settings.colour_tidy:
        rgba = clean.colour_tidy_rgba(rgba, 0.6)

    composed = clean.compose_on_white_square(rgba, settings.margin_pct)
    composed = resize_max_pil(composed, settings.max_edge_px)
    return composed, ShotResult(
        input_path=str(path), face=face, ocr_title=ocr_title,
        detect_method=det_method, detect_conf=round(float(conf), 3),
        cutout_method=cut_method, rotation=rot, status="ok",
        elapsed_ms=int((time.perf_counter() - t0) * 1000),
    )


def process_shot(path, face, settings, manual_quad=None, extra_rotation=0):
    """Detect + deskew + compose one shot. Returns (PIL RGB, ShotResult).

    Primary path: segment the case with rembg, perspective-warp it flat, tighten,
    auto-fix a 180° flip, compose on white. Falls back to the legacy matte engine
    when the case can't be found or the user hand-cropped (manual_quad).
    """
    t0 = time.perf_counter()
    bgr = load_image_bgr(path)

    if manual_quad is None:
        sc = scan.scan_case(bgr, settings.rembg_model)
        if sc is not None:
            return _compose_from_scan(path, sc[0], face, settings, extra_rotation, t0)

    return _compose_fallback(path, bgr, face, settings, manual_quad, extra_rotation, t0)


def _decode_barcode(bgrs, faces):
    """Decode an EAN/UPC off the full-res originals (back first). None if no read."""
    order = sorted(bgrs, key=lambda p: 0 if faces.get(p) == Face.BACK else 1)
    for p in order:
        try:
            digits, _method, _deg = barcode.decode(bgrs[p])
        except Exception:
            digits = None
        if digits and digits.isdigit() and len(digits) >= 8:
            return digits
    return None


def _make_run_dir(output_dir):
    base = Path(output_dir)
    run = base / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True, exist_ok=True)
    return run


def _unique_dir(run_dir, stem):
    """A DVD folder that never collides: two same-titled DVDs would otherwise
    write to one folder and overwrite each other's photos."""
    candidate = run_dir / stem
    n = 2
    while candidate.exists():
        candidate = run_dir / f"{stem} ({n})"
        n += 1
    return candidate


def _read_title(front_upright_bgr, composed, scan_data, settings, gi):
    """Best DVD title: dedicated Qwen title reader -> scan -> front OCR -> fallback.

    The focused 'title only' prompt reads stylised covers far better than the
    multi-field extraction, so it leads when a front cover is available.
    """
    engine = getattr(settings, "title_engine", "auto")
    if (front_upright_bgr is not None and engine in ("auto", "qwen")
            and vlm.available()):
        t = vlm.title_from_cover(front_upright_bgr, settings)
        if t:
            return titles.clean_title(t)
    if scan_data and scan_data.title.known:
        return titles.clean_title(scan_data.title.value)
    front = composed.get(Face.FRONT)
    raw = front[1].ocr_title if front and front[1] else None
    if not raw and front_upright_bgr is not None:
        raw = ocr.extract_title(front_upright_bgr)
    return titles.clean_title(raw) if raw else f"Untitled DVD {gi}"


def _process_chunk(chunk, settings, gi, progress_cb, done, total):
    """Scan, classify, compose, and read title for one DVD (title-only mode)."""
    scans, bgrs = {}, {}
    for path in chunk:
        bgrs[path] = load_image_bgr(path)
        try:
            scans[path] = scan.scan_case(bgrs[path], settings.rembg_model)
        except Exception:
            scans[path] = None
    faces = _assign_faces_from_scans(scans, chunk)

    composed = {}          # Face -> (PIL, ShotResult)
    front_upright = None   # upright front-cover BGR, for title reading

    for path in chunk:
        face = faces[path]
        t0 = time.perf_counter()
        try:
            sc = scans.get(path)
            if sc is not None:
                pil, res = _compose_from_scan(path, sc[0], face, settings, 0, t0)
                if face == Face.FRONT:
                    cover = scan.orient_and_tighten(sc[0], face)
                    by_text = scan.flip_by_text(np.ascontiguousarray(cover[:, :, :3]))
                    flip = by_text if by_text is not None else _decide_flip(cover, settings)
                    up = cv2.rotate(cover, cv2.ROTATE_180) if flip else cover
                    front_upright = np.ascontiguousarray(up[:, :, :3]).copy()
            else:
                pil, res = _compose_fallback(path, bgrs[path], face, settings, None, 0, t0)
        except Exception as e:
            res = ShotResult(input_path=str(path), face=face, status="failed", error=str(e))
            pil = None
        composed[face] = (pil, res)
        done += 1
        if progress_cb:
            progress_cb(done, total, path.name)

    # Barcode is cheap (pyzbar on the back-cover original) and accurate, so it
    # stays in title-only mode — only the slow Qwen field scan was dropped.
    bc = _decode_barcode(bgrs, faces)
    title = _read_title(front_upright, composed, None, settings, gi)
    return composed, None, bc, title, done


def run_batch(settings, progress_cb=None):
    """Process every input, group into DVDs, save named files + listing + log."""
    paths = gather_inputs(settings.input_dir)
    grouped = assign_groups(paths)
    run_dir = _make_run_dir(settings.output_dir)

    total = len(paths)
    done = 0
    dvd_groups = []

    for gi, chunk in enumerate(grouped, 1):
        try:
            composed, scan_data, bc, title, done = _process_chunk(
                chunk, settings, gi, progress_cb, done, total)
        except Exception as e:
            # One bad DVD must never sink the whole batch.
            done += len(chunk)
            if progress_cb:
                progress_cb(done, total, f"DVD {gi} failed: {e}")
            dvd_groups.append(DvdGroup(
                index=gi, barcode=None, title=f"Untitled DVD {gi}",
                region=settings.default_region,
                shots=[ShotResult(input_path=str(p), face=FACE_ORDER[i % 3],
                                  status="failed", error=str(e))
                       for i, p in enumerate(chunk)]))
            continue

        dvd_dir = _unique_dir(run_dir, naming.safe_stem(title))
        dvd_dir.mkdir(parents=True, exist_ok=True)
        group = DvdGroup(index=gi, barcode=bc, title=title,
                         region=settings.default_region, shots=[], scan=None)
        for face, (pil, res) in composed.items():
            res.title, res.region = title, group.region
            if pil is not None:
                out = dvd_dir / naming.output_filename(title, face)
                save_jpeg(pil, out, settings.jpeg_quality)
                res.output_path = str(out)
            group.shots.append(res)
        group.shots.sort(key=lambda s: FACE_ORDER.index(s.face))

        dvd_groups.append(group)

    naming.write_listing(run_dir, dvd_groups)      # master summary
    naming.write_run_log(run_dir, dvd_groups, settings.to_dict())
    return run_dir, dvd_groups
