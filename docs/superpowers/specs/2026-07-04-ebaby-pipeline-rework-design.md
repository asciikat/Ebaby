# Ebaby v3 — Pipeline-First Rework — Design

Date: 2026-07-04
Status: Approved pending user review of this document

## 1. Problem

The current desktop app (`redboxflip/`, Tkinter) inverts the user's actual workflow:

- It crops first and names last: title-based naming happens at output-save time
  (`redboxflip/pipeline.py:314-322`), and the title entry field sits inside the
  interactive crop editor (`redboxflip/gui/editor.py:70-73`). The user requires
  all renaming to be FINISHED before any cropping begins.
- The decoded barcode is stored but never used to name anything, and there is no
  eBay API integration — title lookup uses upcitemdb's free trial API
  (`redboxflip/titles.py:45`), which has no price/category data. A recent commit
  replaced barcode naming with OCR guessing, moving further from the requirement.
- The crop editor discards the auto-detected box and resets corners to a fixed
  10%-margin rectangle (`redboxflip/gui/editor.py:106-109`); rotation wipes user
  corner adjustments (`editor.py:103`).
- Grouping is blind chunks-of-3 (`redboxflip/pipeline.py:38-40`); the user's
  number-keyed 2-shot (brand-new) and letter-keyed 3-shot (used) set system does
  not exist in the app.
- No listing CSV with prices/specifics is produced.

Meanwhile the user's own pipeline scripts in WSL Debian
(`/home/M/Ebaby Code/Pipeline/`) already implement the desired behaviour and are
proven: collision-safe sequential rename, RAW colour correction, barcode
locate/crop/decode, eBay Browse API (AU) lookup with retries, slug-based rename,
and a 13-column listing CSV. The app ignored them. This rework makes them the
engine.

## 2. Requirements (from user, 2026-07-04)

1. Workflow order is fixed: upload → sequential rename → colour correct →
   barcode scan → eBay lookup → rename all images to title slugs → listing CSV →
   only then cropping. Renaming never happens during or after crop.
2. Input is RAW `.NEF`/`.DNG` from the camera (Colorprodawgv1 colour correction
   is part of the flow; barcode scanning uses full-res embedded previews).
3. Crop stage: auto-crop everything, user reviews and fixes only failures.
   The manual editor must be seeded with the auto-detected box.
4. Two drop zones at upload: Used (3-shot letter-keyed sets) and Brand New
   (2-shot number-keyed sets), both handled in one run.
5. Build on the existing WSL pipeline scripts; do not re-invent their logic.

## 3. Architecture (Approach A — approved)

A local web app that runs inside WSL Debian and is used from the Windows browser.

- **Source code**: new `ebaby/` Python package in this git repo
  (`C:\Users\mardi\Documents\Ebay code`), executed from WSL via `/mnt/c/...`.
- **Runtime**: WSL Debian, venv `Ebabyv2` (`/home/M/Ebabyv2`, Python 3.13) which
  already has cv2/numpy/rawpy/pyzbar/rembg (CPU onnxruntime). `exiftool` and
  FastAPI/uvicorn are preflight-checked at startup and reported in the UI if
  missing.
- **Batch data**: `/home/M/Ebaby Runs/<batch-name>/` in WSL home (fast IO),
  one folder per batch (see §5).
- **UI**: FastAPI serving a single-page HTML/JS app (vanilla JS + canvas, no
  build step) at `http://localhost:8765`. Browser drag-and-drop performs the
  upload, which is how images enter WSL.
- **Launcher**: `Run Ebaby.bat` on Windows — starts the server in WSL
  (`wsl.exe -d Debian`), waits for the port, opens the default browser.
- **eBay credentials**: reuse the existing `.env` (EBAY_CLIENT_ID /
  EBAY_CLIENT_SECRET / EBAY_MARKETPLACE_ID=EBAY-AU) via the existing
  `pipeline_config.py` loading logic.
- `redboxflip/`, `dvdflip/`, `ebayflip_V2.py` are untouched and retired in place.

### Package layout

```
ebaby/
  __init__.py
  server.py            # FastAPI app, stage endpoints, static files
  batch.py             # batch folder layout, state.json, resume logic
  stages/
    rename_seq.py      # from bn_rename.py + sh_rename.py (plan/apply, 2-phase)
    color.py           # from Colorprodawgv1.py (RAW decode, WB, enhance)
    barcode_locate.py  # from locate_and_crop_barcodes.py (exiftool previews)
    barcode_decode.py  # from dvd_barcode_scanner.py (multi-view pyzbar)
    ebay.py            # from ebay_api.py + ebay_csv_extractor_new.py
    rename_title.py    # from ebay_Api_rename_new.py (slug rename, fallbacks)
    crop.py            # rembg cutout + tight crop + compose-on-white
                       # (adapted from redboxflip cutout/scan; the one piece
                       # of the old app worth keeping)
  naming_utils.py      # verbatim slugify_title (images and CSV must agree)
  static/              # index.html, app.js, app.css (progress rail, tables,
                       # thumbnail grid, canvas corner editor)
tests/
  test_ebaby_*.py
```

Refactor rule: each source script's core logic moves into a function with the
same behaviour; the original argparse CLIs in WSL stay where they are, unmodified.

## 4. Stage flow

State machine per batch: `upload → rename → color → barcode → ebay → crop → done`.
Each stage writes its outputs and updates `state.json` before the next stage is
allowed to start. The UI shows a progress rail; stages run server-side with
progress polling.

1. **Upload.** Two drop zones (Used = sets of 3, New = sets of 2). Files are
   streamed into `1_originals/{used,new}/`. Validation before proceeding:
   counts divide evenly by set size (else the specific zone is flagged — same
   guard as the scripts' STOP), RAW extensions expected. The rename plan is
   displayed (sorted by filename, the scripts' default; RAW+JPEG pairs share a
   stem): used → `a_front, a_back, a_inside…`, new → `01_front, 01_back…` with
   `_new` reserved for the title-rename stage, as in the current scripts.
   One click applies the plan (two-phase temp-name rename, duplicate-name guard).
2. **Colour correct.** Each RAW → white-balanced, enhanced PNG in `2_color/`
   using Colorprodawgv1 logic (surrounding-white mask WB with border-blob
   constraint, gain clamp 0.5–2.0, S-curve/saturation/sharpen defaults, light
   FBDD denoise). Filenames keep their sequential keys.
3. **Barcodes.** For every `*_back*` RAW: extract embedded JPEG preview via
   exiftool, locate the barcode region (gradient ladder + aspect sanity check),
   crop, then decode with the multi-view ladder (scales, CLAHE, Otsu ± invert,
   unsharp, adaptive threshold; EAN13→UPC-A normalisation). Results table in
   the UI: per set key, decoded digits or MISS. A text input on each MISS lets
   the user type the digits off the case; manual entries are stored alongside
   auto ones. Proceeding with remaining misses is allowed (they become
   placeholder-named sets, same as the scripts).
4. **eBay.** One app token (client-credentials). Per barcode: New-condition and
   Used-condition searches (lowest price incl. shipping across top 10 each) +
   one item-detail call for localizedAspects, 0.5 s spacing, retry/backoff as in
   `ebay_api.py`. Output: `Ebay_Details.csv` with the exact 13 columns of
   `ebay_csv_extractor_new.py` (Barcode, Image Set Name, Title, Region Code,
   Genre, Type, Season, Actor, Studio, Language, Rating, Lowest Price New (AUD),
   Lowest Price Used (AUD)). The UI shows the table; the user may edit any Title
   (slug recomputes) before applying. Then ALL images for each set are renamed
   to `slug_role[_new].ext` into `4_renamed/` — eBay-match title slug, else
   barcode digits, else `NoBarcode_<key>` — identical fallback ladder to
   `ebay_Api_rename_new.py`, with de-dup suffixes shared between CSV and files.
   eBay completely unreachable → user chooses: retry, or continue with
   barcode-digit names (CSV rows marked not-found).
5. **Crop.** For every renamed PNG: rembg matte → largest component →
   quad/tight box → perspective-corrected crop → compose on white 1:1 square →
   JPEG in `5_cropped/` under the SAME slug filename. Thumbnail grid with
   confidence flag; clicking a thumbnail opens the canvas editor showing the
   original image with the DETECTED quad pre-seeded — user drags corners,
   rotation preserved separately from corners, re-crop on save. No title field
   anywhere in this stage.
6. **Done.** Summary: output folder path, CSV path, count of sets, misses that
   used fallback names. Button to open the folder (Windows explorer via
   `explorer.exe` on the WSL path).

## 5. Batch folder layout

```
/home/M/Ebaby Runs/<batch>/
  state.json           # current stage, per-set records, manual overrides
  1_originals/used/    # uploaded RAW, sequentially renamed in place
  1_originals/new/
  2_color/             # <key>_<role>.png
  3_barcodes/          # <key>_back_barcode.png crops + <key>.txt digits
  4_renamed/           # <slug>_<role>[_new].png
  5_cropped/           # <slug>_<role>[_new].jpg  (eBay-ready)
  Ebay_Details.csv
```

`state.json` is the single source of truth for resume: reopening the app lists
existing batches and re-enters at the recorded stage. Every stage is
re-runnable; re-running a stage clears only that stage's outputs (the same
regenerable-folder principle the scripts already use).

## 6. Error handling

- **Preflight** (server start): venv imports (cv2, numpy, rawpy, pyzbar, rembg),
  exiftool on PATH, .env creds present. Failures shown in the UI banner with the
  exact install command, not buried in a log.
- **Upload**: uneven set counts block only the affected zone; partial sets are
  never silently mis-grouped (the root grouping failure of the old app).
- **Barcode misses**: explicit UI state, manual entry, or accepted fallback —
  never a crash, never silent.
- **eBay outages**: EbayUnavailable surfaces as a one-line message + choice
  (retry / continue by barcode digits). Rate-limit respected via delay + retry
  with backoff.
- **Crop failures** (rembg finds nothing usable): image flagged in the grid,
  full-frame box seeded in the editor; batch continues.
- **Crash-safety**: stage results are on disk before the stage is marked done;
  a killed server never loses completed stages.

## 7. Testing

Pytest, runnable in WSL (`python3 -m pytest tests/ -k ebaby`):

- rename_seq: plan generation for 2-shot/3-shot sets, RAW+JPEG pairing,
  letter/number schemes, uneven-count rejection, two-phase collision safety.
- naming: slugify_title behaviour pinned (stop-words, dedup, max 3 words,
  fallback); CSV "Image Set Name" and image slug always agree, including de-dup
  suffixes.
- rename_title: fallback ladder (title → barcode → NoBarcode_key), `_new`
  tagging for number keys, clobber protection.
- ebay: response parsing (lowest price incl. shipping, aspects mapping,
  New-preferred specifics) against recorded fixtures; no live calls in tests.
- barcode_decode: decode ladder on sample crop fixtures; EAN13→UPCA
  normalisation.
- batch/state: stage transitions, resume, stage re-run clearing.
- crop: quad detection returns a seedable box on the sample images; compose
  output is square/white.

End-to-end smoke: a scripted batch using the repo's `samples/` images (plus a
small RAW fixture) walked through every stage with eBay mocked.

## 8. Non-goals

- No eBay listing creation/selling API (user OAuth) — research/pricing only,
  same scope as the current scripts.
- No changes to the legacy `redboxflip/`, `dvdflip/`, `ebayflip_V2.py` code.
- No GPU work; rembg stays on CPU onnxruntime.
- No Qwen/OCR title guessing in the main flow — titles come from eBay via
  barcode, with manual entry as the correction path.
