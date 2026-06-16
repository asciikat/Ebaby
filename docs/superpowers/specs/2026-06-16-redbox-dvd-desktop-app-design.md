# Red-Box DVD Desktop App — design spec

_Date: 2026-06-16. Status: approved design (pending spec review), pre-implementation._

## 1. What we are building

A personal, single-user **desktop GUI app** (Tkinter) that runs **inside Ubuntu /
WSL2** — **no web browser, no FastAPI**. It turns flat phone-scanner photos of DVD
cases, laid on a **hand-drawn red-box template**, into clean **1:1 white-square**
eBay product photos, reads the **barcode** off the back cover to name the files by
title, and writes a plain-text info file per run.

**I/O contract:** input is **`.jpg`** files from the phone's auto-document-scanner
(already perspective-corrected / flat) dropped in an input folder. Output is
**`.jpg`** on a pure-white square canvas, named by DVD title and face.

This **replaces the whole pipeline.** The `dvdflip/` package (WSL + FastAPI web
review + model-free CV) is retired. The legacy `ebayflip_V2.py` and the new V1
desktop app (`ebayflip_V1.txt`) remain as **reference only** — we lift V1's proven
pixel/UI code, but build a fresh, self-contained package.

## 2. The capture protocol (what the user actually does)

The user draws a template on a white A-sheet: a red outline box (the placement
guide) with corner **X** marks and handwritten labels (`Front` / `Back` /
`Centre`). One DVD face is laid in the box and scanned with the phone's document
scanner, which outputs a flat, perspective-corrected JPG.

**Fixed shot order, 3 shots per DVD:**

1. **Back cover** — *first*, because it carries the barcode. The barcode is scanned
   here; the resulting **title names all three files** for this DVD.
2. **Front cover.**
3. **Inside / open** — the case opened flat (disc + inner sleeve), a wide landscape shot.

A new back cover (a freshly decoded barcode) **starts a new DVD group.**

The handwritten `Front`/`Back`/`Centre` labels are physical placement reminders for
the user — the app does **not** OCR them (red-marker handwriting OCR is the least
reliable option and we rejected it).

## 3. Core principle

> **Automation-first, with an intelligent fallback ladder at every stage and a
> complete manual override for everything.**

The happy path is: drop photos in, hit **Run**, walk away — detect, crop, upright,
barcode, title lookup, naming all happen automatically. You only open the review
grid if something looks wrong. When a stage's primary method fails, it falls
through to progressively dumber-but-safer methods before ever asking you. And every
automatic decision (crop, rotation, face, grouping, title, region, even the barcode
digits) can be overridden by hand in the GUI.

## 4. Architecture

A fresh self-contained package, `redboxflip/`. Each module is small and
single-purpose; data between modules is plain (image arrays + dataclasses) so each
can be tested alone. Proven functions are lifted from V1 (`ebayflip_V1.txt`).

| Module | Responsibility | Source |
|--------|----------------|--------|
| `__main__.py` | Entry point; launch GUI (default) or headless CLI (`--input/--output`) | new |
| `config.py` | Settings dataclass, defaults, constants, config persistence (`~/.redboxflip.json`) | adapt V1 |
| `models.py` | `Shot`, `DvdGroup`, `Settings`, `ProcessResult` dataclasses | new |
| `detect.py` | Red-box ROI detection (the detection ladder) | new + V1 geometry |
| `cutout.py` | Matte the case off the background via a pluggable engine (`rembg` **or** SAM box-prompt) → GrabCut(red-box) → geometric → manual; feather the edge | new + V1 AI/edge code |
| `clean.py` | Erase residual red, auto-upright, composite on white square, colour tidy | port V1 |
| `barcode.py` | Robust multi-decoder barcode scanning (the crux) | rebuild from V1 |
| `titles.py` | Title resolution chain: online lookup → cache → manual; pluggable seam for offline guesser | new |
| `pipeline.py` | `process_one`, batch orchestration, grouping into DVDs | new |
| `naming.py` | Title→filename, listing `.txt` / `.csv`, `run_log.json` | port V1 |
| `gui/app.py` | Main window: folders, settings, Run, progress | new (Tkinter) |
| `gui/grid.py` | Results grid grouped by DVD, status badges, "step through" | new |
| `gui/editor.py` | Per-shot editor: corner-drag crop + magnifier, rotate, face, title, region, re-scan | port V1 `CornerDialog` |

Modules may collapse during implementation if a split proves pointless, but the
GUI / pipeline / CV / barcode / titles seams stay distinct.

## 5. Per-photo pipeline (with fallback ladders)

For each input JPG, in order:

**Stage A — Load.** Open JPG, apply EXIF transpose, convert to working array.

**Stage B — Red-box ROI detection.** Ladder:
1. HSV red mask (two hue ranges for red's hue wraparound) → morphological close →
   largest contour → 4-point approx = the box. Crop to its interior.
2. If no confident red quad: LAB **a**-channel (red–green) threshold as a second
   red detector.
3. If still none: V1-style content/edge detection on the whole flat scan.
4. Final fallback: use the whole image as the ROI.
Each step records a confidence; low confidence flags the shot for review.

**Stage C — Cut the case out (matte).** The red box is the *guide*; the goal is a
precise mask hugging the real edges of the DVD case (back / front / inner), with a
**slight feathered softness** so the composite looks natural. Intelligent ladder,
auto-first, manual always available:
1. **AI matte (primary) — selectable engine** (chosen in Settings, overridable
   per-image in the editor):
   - **rembg** (default; U2Net/`isnet-general-use` family): crop to the red-box ROI and
     run rembg on *just that region* so surrounding paper/clutter can't confuse it →
     alpha matte; or
   - **SAM box-prompt** (MobileSAM — light, box-prompt capable): feed the red box itself
     as a literal box-prompt so the model segments exactly what sits inside the outline
     → alpha matte.
   Either way, validate coverage (reject empty/tiny masks, as V1's `validate_ai_rgba`
   did) before trusting the result, and fall through the ladder if it fails.
2. **GrabCut seeded by the red box (fallback, no download):** use the red rectangle
   interior as the GrabCut init rect and the red-line band as probable-background →
   foreground mask. Instant, CPU-only.
3. **Geometric rectangle (final auto fallback):** bounding box of non-white/non-red
   content inside the ROI; the inside/open shot is allowed a wider aspect.
4. **Manual:** corner-drag / mask fix in the editor.

The mask edge is **feathered a few px** (the requested softness). Any near-red pixels
that survive are erased to white. **No perspective re-warp is ever applied** — the
scan is already flat, so we only mask, rotate in 90° steps, and place at true pixel
scale ⇒ **no distortion**.

**Stage D — Auto-upright.** Ladder:
1. **Back shots:** the rotation at which the barcode decodes is the upright
   reference (we already try all rotations to scan — reuse that result).
2. Tesseract OCR text-orientation vote (if `tesseract` is installed).
3. Aspect heuristic (front/back → portrait; inside → landscape).
4. Manual rotate L / R / 180 buttons in the editor.

**Stage E — Compose.** Drop the upright, feathered cutout centered on a **pure-white
1:1 square** with a configurable margin, composited through its alpha so the white
shows cleanly around the soft edge and **no red outline is visible**. Optional gentle
colour tidy (white-balance + mild contrast), default on but conservative. Resize to a
max edge, save high-quality JPG.

**Stage F — Barcode (back shots only).** See §6.

## 6. Barcode subsystem — the crux

This is the part that has been failing and the part the user cares most about.
Reading the **digits** is made near-bulletproof:

- **Scan the full-resolution original back photo**, not the downscaled/cropped
  output — thin bars survive only at full res. (Likely root cause of past failures.)
- **Multiple decoders, first hit wins:** `pyzbar` (ZBar) + `zxing-cpp` (strong on
  EAN-13, e.g. the sample `9325336022306`) + OpenCV `cv2.barcode.BarcodeDetector`
  (opencv-contrib).
- **Cartesian sweep until a hit:** rotations `{0, 90, 180, 270}` × scales
  `{1.0, 1.5, 2.0}` × preprocessings `{none, grayscale, CLAHE, adaptive-threshold,
  unsharp}`.
- Optional barcode-region localization (detector ROI) then upscale that region.
- **Hard dependency check at launch** with a plain install message, so a missing
  `libzbar0` produces a clear instruction — never a silent failure.
- **Manual override:** if nothing decodes, the grid shows the shot red; you type the
  number (or just the title) by hand. Manual entry is first-class.

The decoded number is stored on the DVD group and used to look up the title.

## 7. Title resolution chain

A pluggable ladder; first success wins, result cached:

1. **Local cache** (`~/.redboxflip_titles.json`) — repeat barcodes are instant/offline.
2. **Online lookup** by barcode (default: a free UPC service such as upcitemdb trial),
   with a short timeout and one optional secondary source.
3. **Manual entry** in the grid — pre-filled with any guess, blank otherwise. The
   title field drives all three filenames live.
4. **Pluggable seam (not in core):** an optional offline guesser (cover-text OCR, or
   a local LLM such as Ollama/Qwen) could register here later. **Deliberately
   excluded** from the build — it has caused more problems than it solved on this
   machine and is not a dependency.

If no title is resolved, filenames fall back to the barcode number, else
`Untitled DVD N`.

**Honest expectation:** decoding the *number* is solved; turning a number into a
*title* depends on external databases that do not always carry Australian DVDs.
Lookup auto-fills when it can and **never blocks** — manual entry is always one field
away.

## 8. Grouping, naming, and outputs

**Grouping:** fixed cycle Back → Front → Inside; a new back (new barcode) starts a
new DVD group. The grid lets you merge/split groups and re-label faces if a shot was
skipped or doubled.

**Filenames** (sanitized title):
- `{Title} - Back Cover.jpg`
- `{Title} - Front Cover.jpg`
- `{Title} - Inside.jpg`

**Output layout:** `output/run_<timestamp>/{Title}/` holding that DVD's three JPGs
(a per-DVD subfolder keeps each eBay listing's photos together).

**Per-run files** in `output/run_<timestamp>/`:
- `dvd_listing.txt` — one block per DVD: **Title, Region, Barcode, basic info from
  the lookup** (brand/category/etc.), and the three photo filenames. This is the text
  file the user asked for.
- `dvd_listing.csv` — same data, spreadsheet-friendly.
- `run_log.json` — per-shot diagnostics (which detector/decoder fired, confidence,
  rotations tried, timings) so future "why did this fail" questions are answerable.

**Region:** not generally available from the barcode, so it is a user default —
**Region 4 / PAL (Australia)** out of the box — editable globally in settings and
per-DVD in the grid.

## 9. The one window (no web)

All three requested flows live in a single Tkinter app:

- **Main window:** input folder, output folder, **Run**, a progress log, and settings
  (**cutout engine: rembg / SAM box-prompt**, margin %, JPEG quality, max edge px,
  default region, colour-tidy on/off, title lookup on/off, auto-open output folder,
  auto-open review grid).
- **Fire-and-forget:** hit Run; on finish, show "_N DVDs done_" and auto-open the
  output folder. No clicks needed.
- **Review grid** (`gui/grid.py`): thumbnails grouped by DVD, each tagged
  front/back/inside with a status badge (barcode ok / title found / needs review).
  Opens automatically after a run unless disabled.
- **One-by-one:** a **Step through** button walks the editor over each shot
  (Save & Next), V1-style.
- **Editor** (`gui/editor.py`, reuses V1's `CornerDialog` + zoom magnifier):
  drag the 4 crop corners, rotate L/R/180, set the face, **switch the cutout engine
  (rembg / SAM / GrabCut) and re-run the matte**, **edit the title** (renames the
  group's files live), set region, and **re-scan / hand-type the barcode**.

Every automatic result is visible and overridable here — the "complete manual
override" requirement.

## 10. Dependencies & runtime

- **Runtime:** Python 3 in WSL2 Ubuntu; the GUI renders via **WSLg** (native on
  Windows 11 — no X server setup). `Run DVD Flip.bat` is updated to launch the new
  app inside WSL.
- **pip:** `opencv-contrib-python`, `numpy`, `pillow`, `pyzbar`, `zxing-cpp`,
  `rembg`, `onnxruntime`, `requests` (or stdlib `urllib`), `pytesseract` (optional).
- **SAM engine (optional, for the SAM cutout route):** `mobile_sam` (or `segment-anything`)
  + its weights (~40MB for MobileSAM). Only needed if the SAM engine is selected.
- **apt:** `python3-tk`, `python3-pil.imagetk`, `libzbar0`, `tesseract-ocr` (optional,
  for the upright OCR fallback).
- **One-time model downloads:** the `rembg` model (~170MB for `isnet-general-use`) and,
  if used, the MobileSAM weights (~40MB) download on first use. Slow on this link but
  cached forever after; GrabCut covers cutout until they land.
- A clear, actionable error at startup if a hard dependency is missing; if the selected
  AI engine is unavailable the cutout ladder silently drops to GrabCut.

## 11. Error handling

- Every stage degrades down its ladder rather than crashing; the worst case is "use
  whole image / flag for review," never an exception that stops the batch.
- A failed image is logged, counted, and skipped — the batch continues.
- Missing hard deps → explicit install instructions and clean exit.
- Network/lookup failure → silent fall-through to cache/manual; never blocks.

## 12. Testing strategy

- **Fixtures:** the four real template scans (`IMG_20260616_*.jpg`) and the two
  liked outputs are committed to `samples/redbox/` as regression fixtures. _(These
  are not yet in the repo — they must be added before/early in implementation.)_
- **Unit tests:** red-box detection on the samples + synthetic boxes; case crop
  bounds; red-erase leaves no red; compose produces an exact square on pure white;
  barcode decode returns `9325336022306` from the sample back at full res and at
  90/180/270; naming/sanitizing; grouping a 9-shot batch into 3 DVDs.
- **End-to-end:** run the sample batch headless; assert 1 DVD, 3 correctly-named
  square JPGs, a `dvd_listing.txt` with the right fields.

## 13. Non-goals (YAGNI)

- No web UI / FastAPI / browser anything.
- No local **LLM for titles** in the core (the Ollama/Qwen seam in §7 stays optional
  and off by default). _AI image segmentation for the cutout (§5 Stage C) **is** in
  the core — that's a light vision model, not the LLM that caused grief._
- No automatic region detection (user default + manual override instead).
- No OCR of the handwritten template labels.
- No multi-user, no cloud, no packaging beyond the WSL launcher.

## 14. Decisions made during brainstorming

- Replace the whole pipeline; retire `dvdflip/`.
- One face per photo; crop tight to the case (red box = ROI prior).
- All three flows (batch grid / one-by-one wizard / fire-and-forget) in one Tkinter
  GUI, no web.
- Fixed order **Back → Front → Inside**, back-first for the barcode.
- Title comes from the barcode lookup and names all three files; manual fallback.
- Default region: Region 4 / PAL (Australia), editable.
- Ollama/Qwen left out of core; titles chain keeps a pluggable seam for it.
- Cutout is an **AI matte** with a feathered soft edge, composited on pure white, no
  re-warp (no distortion). The AI engine is **user-selectable: rembg (default) or SAM
  box-prompt** (both guided by the red box); GrabCut and geometric crop are the
  no-download fallbacks, manual override always available.

## 15. Open items to confirm during planning

- Title-lookup source: default to upcitemdb trial + manual fallback, or a specific
  database the user prefers?
- Output as per-DVD subfolders (proposed) vs flat folder with title in the filename.
- Whether to keep a conservative colour tidy on by default, or leave images untouched.
