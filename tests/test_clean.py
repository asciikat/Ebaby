import numpy as np
from dvdflip.loader import load_bgr
from dvdflip.flatten import detect_a4, warp_to_a4
from dvdflip.clean import (white_balance_from_paper, segment_dvd, crop_rect,
                           dvd_size_mm, scan_barcodes, composite_on_white)

def _flat(path):
    bgr = load_bgr(path)
    quad, _ = detect_a4(bgr)
    return warp_to_a4(bgr, quad)

def test_segment_and_size_front(sample_front):
    flat, _ = white_balance_from_paper(_flat(sample_front))
    box = segment_dvd(flat)
    assert box is not None
    short_mm, long_mm = dvd_size_mm(box)
    assert 120 < short_mm < 160 and 170 < long_mm < 210

def test_barcode_reads_on_back(sample_back):
    flat, _ = white_balance_from_paper(_flat(sample_back))
    box = segment_dvd(flat)
    dvd = crop_rect(flat, box)
    codes = scan_barcodes(dvd)
    assert codes and "0025192828928" in codes

def test_composite_is_square_white(sample_front):
    flat, _ = white_balance_from_paper(_flat(sample_front))
    dvd = crop_rect(flat, segment_dvd(flat))
    out = composite_on_white(dvd)
    assert out.shape[0] == out.shape[1]
    assert (out[0, 0] == [255, 255, 255]).all()
