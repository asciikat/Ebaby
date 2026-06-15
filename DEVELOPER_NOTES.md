# DVD Photo Processor V2 — Developer Notes

_Last updated: 2026-06-15. Target runtime: **Ubuntu 24.04 on WSL2**._

## 1. What this is

`ebayflip_V2.py` turns top-down phone photos of a DVD (laid on a white **A4
sheet**) into clean, deskewed, white-background eBay photos, auto-classified as
**front / back / center**. It is a ground-up rewrite of `ebayflip_V1.py`,
built and tuned against three real RAW samples of *King Kong Escapes*.

Per-image pipeline:

| # | Stage | Function | Notes |
|---|-------|----------|-------|
| 1 | Load | `load_bgr` | `.dng` via **rawpy** (honours orientation tag); JPG/PNG/TIFF/HEIC via PIL |
| 2 | Detect A4 + deskew | `detect_a4`, `warp_to_a4` | largest near-A4 bright quad → perspective warp to true A4 (10 px/mm) |
| 3 | White balance | `white_balance_from_paper` | uses the **paper itself** as the neutral-white reference; lifts exposure |
| 4 | Segment DVD | `segment_dvd`, `crop_rect` | min-channel whiteness → **straight-edged rotated rectangle** (no jagged contours) |
| 5 | Classify | `classify_view`, `scan_barcodes` | size-vs-A4 → center; else barcode → back; else front |
| 6 | Orient + compose | `upright_vote`, `apply_orientation_consensus`, `composite_on_white` | text-based 180° fix + run consensus; white square canvas |

## 2. Current status — WORKING

End-to-end on the 3 sample DNGs (run `processed/run_*`), **all three correct**
(verified visually):

| Input | Output | View | Barcode | Notes |
|-------|--------|------|---------|-------|
| `...153112.dng` | `..._front.jpg` | front | – | upright after consensus flip |
| `...153120.dng` | `..._back.jpg` | back | `0025192828928` | confident orientation from text |
| `...153137.dng` | `..._center.jpg` | center | – | upright at geometric baseline |

What's solid:
- **DNG decode** + orientation tag handling (Xiaomi 15 Ultra, 4096×3072, 10-bit).
- **A4 detection**: front/back ~0.86 confidence, open case ~0.48 (usable).
- **Deskew** to true A4 proportions — geometry is correct.
- **White balance from paper** — paper reads clean white, colours natural. This
  is the big win from shooting RAW.
- **Segmentation** with a straight rotated-rectangle mask — meets the
  "no jagged corners" requirement.
- **Classification** front/back/center is correct on the samples.
- **Barcode** read from the back cover (pyzbar, multi-rotation, upscaled).
- Listing `batch_listing.txt` / `.csv` + `run_log.json` written.

## 3. Known issues / what is fragile

### 3.1 Orientation (the hard problem) — PARTIALLY SOLVED
The 180° "which way up" is **not derivable from geometry** — it depends on how
the disc was laid on the paper. Current approach:
- Per image, OCR the crop at 0° and 180° (`upright_vote`); if text reads clearly
  better one way, trust it. **Works well for text-heavy backs.**
- Ambiguous covers (stylised titles) are reconciled by a **run-level consensus**
  (`apply_orientation_consensus`): confident covers vote, ambiguous covers follow.

**Where it can still go wrong:**
- A run/DVD whose **only** confident-text image is missing (e.g. a front-only
  batch with a stylised title and no readable back) → orientation may be wrong.
- Consensus is **run-wide**, not per-DVD-group. A run with **multiple different
  DVDs** could cross-contaminate votes if their capture orientations differ.
- **Center (open case)** is deliberately excluded from consensus (its geometry
  differs from the covers). It is left at its geometric baseline; for some discs
  that baseline could be upside down (no reliable cue — sparse disc-label text).
- Requires **tesseract** + `pytesseract`. Without them, `upright_vote` always
  returns "uncertain" and orientation falls back to geometric baseline.
- Minor: the log prints "(orient: by consensus)" for *any* non-confident image,
  even when consensus didn't actually flip it (e.g. center). Cosmetic only.

### 3.2 A4 detection on the open case is weak (conf 0.48)
The big black case leaves only a thin paper frame; detection bled slightly into
the reflective glass at the bottom, lowering confidence and over-sizing the quad.
Mitigated downstream (border-touching components are dropped in `segment_dvd`),
but it sits close to `REVIEW_CONFIDENCE = 0.45`. A darker/▢busier surface or a
case flush to the paper edge could push it below threshold → **skipped** in CLI.

### 3.3 Segmentation assumptions
`segment_dvd` assumes:
- The paper is the brightest near-neutral region after white balance.
- The DVD is separated from any glass "ring"/strip by a **white paper margin**
  (so border-connected components can be dropped without losing the DVD).
- If the DVD touches the sheet edge (no margin) the border-removal could delete
  it; there is a fallback (`if cleaned.any()`) but it is not bullet-proof.
Bright specular glare on the glossy case could also punch holes (closed by
morphology, but extreme cases may fragment the mask).

### 3.4 Classification threshold
`CENTER_LONG_MM = 230`. A boxset, slim case, or oversized special edition could
fall on the wrong side. The size measurement is reliable because everything is
scaled to physical A4 mm, but the **threshold itself is a heuristic**.

### 3.5 Listing grouping
`group_and_write` starts a new DVD group on each new front/center **in shot
order**. Multiple DVDs per run therefore rely on a consistent shot sequence.
Barcode→title lookup (present in V1) is **not yet ported** to V2.

## 4. Not yet implemented (planned)
- **GUI review-fallback** (user chose "auto with review fallback"). V1's Tk
  corner dialog + preview still need porting; right now low-confidence images
  are **skipped** in CLI rather than sent to a manual corner-placement dialog.
  WSLg provides `DISPLAY=:0`, so a Tk GUI will display.
- **Barcode → title lookup** (UPC API) for the listing files.
- **Per-DVD-group** orientation consensus (instead of run-wide).
- HEIC / non-DNG formats are coded but **untested** in this pipeline.

## 5. Tuning constants (top of file)
| Constant | Value | Meaning |
|----------|-------|---------|
| `PX_PER_MM` | 10 | warp resolution; A4 → 2970×2100 px |
| `A4_RATIO` | 1.414 | 297/210, used to score the sheet quad |
| `AUTO_CONFIDENCE` | 0.72 | A4 conf ≥ → trust without review |
| `REVIEW_CONFIDENCE` | 0.45 | A4 conf below → skip (CLI) / review (GUI) |
| `CENTER_LONG_MM` | 230 | long edge ≥ → classify as center |
| A4 mask threshold | `max(0.40·Lmax, 0.80·otsu)` | in `detect_a4`; low end of bright histogram |
| paper threshold | `max(120, 0.72·pct92(minc))` | in `segment_dvd`; min-channel whiteness |

## 6. Running & debugging
```bash
# venv (already created on this machine at ~/ebay-venv/venv)
source ~/ebay-venv/venv/bin/activate
pip install numpy opencv-python-headless rawpy pillow pillow-heif pyzbar pytesseract
# system libs already present: libzbar0, python3-tk, tesseract-ocr

cd "/mnt/c/Users/mardi/Documents/Ebay code"
python3 ebayflip_V2.py                 # batch  ./Images in  ->  ./processed/run_<ts>
python3 ebayflip_V2.py --debug         # also dump per-stage images to run_<ts>/_debug/
python3 ebayflip_V2.py -i <in> -o <out>
```
Debug dumps per image: `*_a4mask.png`, `*_flat.png`, `*_balanced.png`,
`*_dvdmask.png` — invaluable for diagnosing detection/segmentation.

## 7. Environment facts
- Bash tool here is **MSYS2/git-bash**, *not* WSL. Reach Ubuntu via
  `wsl.exe -e bash -lc '...'`.
- Python 3.12 in WSL; venv at `~/ebay-venv/venv`.
- ImageMagick is **not** installed (the `convert` on PATH is Windows'
  FAT→NTFS tool). `exiftool` is available in MSYS for DNG metadata/preview.
- No passwordless sudo in WSL → cannot `apt install` non-interactively. All
  needed system libs happen to be present already.

## 8. Recommended next steps (priority order)
1. Port the **GUI review fallback** so low-confidence A4 (e.g. the open case)
   and uncertain orientations get a one-click manual fix instead of a skip.
2. Make orientation consensus **per DVD group**, and add a barcode-position
   tiebreaker for backs.
3. Improve **A4 detection on the open case** (e.g. detect the 4 paper edges via
   Hough lines as a second method; take the better-scoring of the two).
4. Re-add **barcode→title lookup** to enrich the listing files.
5. Test on more DVDs (boxsets, slim cases, glossy glare, different lighting).
