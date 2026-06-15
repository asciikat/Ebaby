from pathlib import Path
import cv2

from . import vision
from .config import REVIEW_CONFIDENCE, OUTPUT_MAX_DIM, JPEG_QUALITY
from .loader import load_bgr
from .flatten import detect_a4, warp_to_a4
from .clean import (white_balance_from_paper, segment_dvd, crop_rect,
                    dvd_size_mm, scan_barcodes, enhance, composite_on_white,
                    resize_max)
from .models import Photo

_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


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

    result = vision.classify_photo(dvd, size_mm, barcode)
    if result.rotation_cw in _ROT:
        dvd = cv2.rotate(dvd, _ROT[result.rotation_cw])

    dvd = enhance(dvd)
    final = resize_max(composite_on_white(dvd), OUTPUT_MAX_DIM)

    return Photo(
        source_path=path,
        work_image=_write_work(final, work_dir, stem),
        side=result.side,
        rotation_cw=result.rotation_cw,
        confidence=result.confidence,
        barcode=barcode,
        size_mm=size_mm,
        a4_found=True,
        title=result.title,
        year=result.year,
    )
