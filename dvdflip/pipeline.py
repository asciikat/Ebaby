from pathlib import Path
import cv2

from .config import REVIEW_CONFIDENCE, OUTPUT_MAX_DIM, JPEG_QUALITY
from .loader import load_bgr
from .flatten import detect_a4, warp_to_a4
from .clean import (white_balance_from_paper, segment_dvd, crop_rect,
                    dvd_size_mm, classify_view, orient_for_view, upright_vote,
                    scan_barcodes, enhance, composite_on_white, resize_max)
from .models import Photo


def _write_work(bgr, work_dir, stem):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / f"{stem}.jpg"
    cv2.imwrite(str(out), bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return out


def process_image(path, work_dir):
    """CV-clean one image, ask the model to classify it, return a Photo."""
    path = Path(path)
    stem = path.stem
    bgr = load_bgr(path)

    quad, conf = detect_a4(bgr)
    if quad is None or conf < REVIEW_CONFIDENCE:
        fallback = resize_max(bgr, OUTPUT_MAX_DIM)
        return Photo(source_path=path, a4_found=False, side="other",
                     work_image=_write_work(fallback, work_dir, stem))

    flat, _ = white_balance_from_paper(warp_to_a4(bgr, quad))
    box = segment_dvd(flat)
    if box is None:
        fallback = resize_max(flat, OUTPUT_MAX_DIM)
        return Photo(source_path=path, a4_found=False, side="other",
                     work_image=_write_work(fallback, work_dir, stem))

    dvd = crop_rect(flat, box)
    barcodes = scan_barcodes(dvd) or []
    barcode = barcodes[0] if barcodes else None
    size_mm = dvd_size_mm(box)

    view = classify_view(box, has_barcode=bool(barcode))
    dvd = orient_for_view(dvd, view)
    flip, sure = upright_vote(dvd)
    if sure and flip:
        dvd = cv2.rotate(dvd, cv2.ROTATE_180)

    dvd = enhance(dvd)
    final = resize_max(composite_on_white(dvd), OUTPUT_MAX_DIM)

    return Photo(
        source_path=path,
        work_image=_write_work(final, work_dir, stem),
        side=view,
        rotation_cw=0,
        confidence=conf,
        barcode=barcode,
        size_mm=size_mm,
        a4_found=True,
        orient_flip=flip,
        orient_confident=sure,
    )


def reconcile_orientation(photos):
    """Flip text-ambiguous covers 180 to match confidently-oriented covers in the group.

    Geometry can't tell up from down; OCR confidently uprights text-heavy covers
    (usually the back). Every shot in a group shares one capture orientation, so an
    ambiguous cover (e.g. a stylised front) follows the confident vote.
    """
    covers = [p for p in photos if p.side in ("front", "back")]
    votes = [p.orient_flip for p in covers if p.orient_confident]
    if not votes or sum(votes) <= len(votes) / 2.0:
        return
    for p in covers:
        if p.orient_confident or not p.work_image:
            continue
        img = cv2.imread(str(p.work_image))
        if img is None:
            continue
        cv2.imwrite(str(p.work_image), cv2.rotate(img, cv2.ROTATE_180),
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        p.extra_rotation_cw = (p.extra_rotation_cw + 180) % 360
