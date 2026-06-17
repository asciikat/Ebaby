"""Batch orchestration: gather -> group -> process each shot -> save + listing."""
import time
from pathlib import Path

from . import clean, cutout, detect, naming, titles
from .barcode import decode as decode_barcode
from .config import load_cache_for_batch
from .imaging import load_image_bgr, resize_max_pil, save_jpeg
from .models import Face, FACE_ORDER, ShotResult, DvdGroup
from .titles import resolve_title

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def gather_inputs(folder):
    folder = Path(folder)
    files = [p for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in SUPPORTED]
    return sorted(files, key=lambda p: p.name.lower())


def assign_groups(paths):
    """Slice the batch into [(path, Face)] groups of the fixed cycle."""
    groups = []
    for i in range(0, len(paths), len(FACE_ORDER)):
        chunk = paths[i:i + len(FACE_ORDER)]
        groups.append([(p, FACE_ORDER[j]) for j, p in enumerate(chunk)])
    return groups


def process_shot(path, face, settings, manual_quad=None, extra_rotation=0):
    """Run detect -> cutout -> clean -> compose. Returns (PIL RGB, ShotResult)."""
    t0 = time.perf_counter()
    bgr = load_image_bgr(path)

    if manual_quad is not None:
        quad, conf, det_method = manual_quad, 1.0, "manual"
    else:
        quad, conf, det_method = detect.find_roi(bgr)

    rgba, cut_method = cutout.make_cutout(
        bgr, quad, settings.cutout_engine, settings.feather_px,
        rembg_model=settings.rembg_model, sam_checkpoint=settings.sam_checkpoint)
    rgba = clean.erase_red_to_white(rgba)

    barcode_digits = None
    if face == Face.BACK:
        barcode_digits, _m, _r = decode_barcode(bgr)

    rgba, rot = clean.auto_upright(rgba, face)
    rot = (rot + extra_rotation) % 360
    if extra_rotation:
        import numpy as np
        steps = (extra_rotation // 90) % 4
        for _ in range(steps):
            rgba = np.ascontiguousarray(np.rot90(rgba, k=-1))  # clockwise

    if settings.colour_tidy:
        rgba = clean.colour_tidy_rgba(rgba, 0.6)

    composed = clean.compose_on_white_square(rgba, settings.margin_pct)
    composed = resize_max_pil(composed, settings.max_edge_px)

    result = ShotResult(
        input_path=str(path), face=face, barcode=barcode_digits,
        detect_method=det_method, detect_conf=round(float(conf), 3),
        cutout_method=cut_method, rotation=rot, status="ok",
        elapsed_ms=int((time.perf_counter() - t0) * 1000),
    )
    return composed, result


def _make_run_dir(output_dir):
    base = Path(output_dir)
    run = base / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True, exist_ok=True)
    return run


def run_batch(settings, progress_cb=None):
    """Process every input, group into DVDs, save named files + listing + log."""
    paths = gather_inputs(settings.input_dir)
    grouped = assign_groups(paths)
    run_dir = _make_run_dir(settings.output_dir)
    cache = load_cache_for_batch()

    total = len(paths)
    done = 0
    dvd_groups = []

    for gi, shots in enumerate(grouped, 1):
        composed = {}     # Face -> (PIL, ShotResult)
        for path, face in shots:
            try:
                pil, res = process_shot(path, face, settings)
            except Exception as e:
                res = ShotResult(input_path=str(path), face=face,
                                 status="failed", error=str(e))
                pil = None
            composed[face] = (pil, res)
            done += 1
            if progress_cb:
                progress_cb(done, total, path.name)

        back = composed.get(Face.BACK)
        barcode = back[1].barcode if back else None
        title, _src = resolve_title(barcode, do_lookup=settings.title_lookup,
                                    cache=cache)
        if not title:
            title = barcode or f"Untitled DVD {gi}"

        dvd_dir = run_dir / naming.safe_stem(title)
        dvd_dir.mkdir(parents=True, exist_ok=True)
        group = DvdGroup(index=gi, barcode=barcode, title=title,
                         region=settings.default_region, shots=[])
        for face, (pil, res) in composed.items():
            res.title, res.region = title, group.region
            if pil is not None:
                out = dvd_dir / naming.output_filename(title, face)
                save_jpeg(pil, out, settings.jpeg_quality)
                res.output_path = str(out)
            group.shots.append(res)
        group.shots.sort(key=lambda s: FACE_ORDER.index(s.face))
        dvd_groups.append(group)

    titles.save_cache(cache)
    naming.write_listing(run_dir, dvd_groups)
    naming.write_run_log(run_dir, dvd_groups, settings.to_dict())
    return run_dir, dvd_groups
