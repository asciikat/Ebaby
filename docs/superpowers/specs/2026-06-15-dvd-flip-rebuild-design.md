# DVD Flip (rebuild) — design spec

_Date: 2026-06-15. Status: approved design, pre-implementation._

## 1. What we are building

A personal, single-user desktop tool that turns top-down phone photos of a DVD
(laid on a plain white **A4 sheet**) into clean, deskewed, white-background
eBay product photos, and drafts a copy-paste listing for each DVD.

**I/O contract:** the expected input is **`.dng` RAW files** (shot on the user's
Xiaomi 15 Ultra) dropped into the `Images in` folder; the export is always
**`.jpg`**. JPG/HEIC inputs are accepted on a best-effort basis, but DNG→JPG is
the canonical workflow the tool is built and tested for.

This is a **ground-up rebuild** of the existing `ebayflip_V2.py`. V2 works but
its weak spots are all in the *decision* logic (which side is this? which way is
up? which photos belong together?), which V2 does with hand-tuned heuristics
(OCR voting, size thresholds, run-wide consensus, shot-order grouping). The
rebuild keeps the parts of V2 that are reliable — the pixel operations — and
replaces **every** heuristic with a single call to a **local vision model**.

The existing `ebayflip_V2.py` and `DEVELOPER_NOTES.md` remain in the repo as
**reference only**. We do not modify or depend on them.

## 2. Core principle

> **Classic computer vision does the pixels. A local vision model makes the
> decisions.**

CV is precise at geometry/colour and bad at semantics. A vision model is the
opposite. So we split the work cleanly along that line, with plain data passing
between the two halves.

## 3. Pipeline

Five stages, each a small single-purpose module. Data between modules is plain
(an image array + a small dict), so any stage can be swapped or tested alone.

| # | Stage | Does | Source |
|---|-------|------|--------|
| 1 | Load & flatten | Decode **`.dng`** RAW via rawpy (honour orientation); JPG/HEIC best-effort. Detect the A4 sheet, perspective-warp to true A4, establish a real-world **mm scale** | ported from V2 |
| 2 | Clean the photo | White-balance off the paper, segment + crop the DVD as a straight rotated rectangle, composite on a white square; decode the **barcode** with pyzbar | ported from V2 |
| 3 | AI decides | Send the clean image to the local vision model; get back side / orientation / title / year | **new** |
| 4 | You review | Browser review screen: confirm or one-click fix each call, edit the draft listing | **new** |
| 5 | Save | Write named JPGs + per-DVD `listing.txt` + batch CSV | new (CSV ported in spirit) |

### Stage 1–2 (the "pixels" half) — reused from V2

Lift V2's working functions: RAW/HEIC decode + orientation, `detect_a4` /
`warp_to_a4`, `white_balance_from_paper`, `segment_dvd` / `crop_rect`,
`composite_on_white`, and pyzbar barcode decode.

**pyzbar stays** for the barcode rather than asking the model to read digits —
vision models hallucinate digit strings; pyzbar returns the exact value or
nothing.

Output of this half, per photo: one clean deskewed white-background image, the
object's measured size in mm, and the barcode string (or `None`).

### Stage 3 (the brain) — replaces all V2 heuristics

For each clean photo, call the local vision model with the image plus two hints:
the measured size in mm (from the A4 ruler) and the barcode if found. The model
returns:

```json
{
  "side": "front | back | center | spine | other",
  "rotation_cw": 0,            // degrees clockwise to stand it upright: 0/90/180/270
  "title": "King Kong Escapes",
  "year": 1967,
  "confidence": 0.0            // 0.0–1.0
}
```

This single answer replaces V2's orientation OCR-voting, size-threshold
classifier, run-wide consensus, and shot-order assumptions.

**Model:** Qwen2.5-VL served by **Ollama** (`localhost:11434`). Default tag
`qwen2.5vl:7b`; because the machine has a fast GPU, a larger tag (e.g. `32b`)
can be configured for better small-cover-text reading. The model is a single
config value (`config.py`) so it is swappable without touching pipeline code.

### Grouping & listing draft

- **Grouping into DVDs:** group photos by their normalised AI-read title
  (lowercase, trimmed) so the front/back/centre of one disc become one DVD.
  Fallback when a title is unreadable: cluster by capture time (one disc's
  photos are taken seconds apart), labelled "Untitled DVD N". Groups can be
  merged/split by hand on the review screen.
- **Listing draft (full draft):** once grouped, the model takes one more look at
  the front (and back) and drafts: a search-friendly **title line**, a short
  **description**, and visible **attributes** (genre, region, runtime, studio).
  The **barcode** comes from pyzbar. **Condition** and **price** are always left
  blank for the user.

## 4. Review screen

A local web app (Python + FastAPI) opened in the browser. One card per DVD:

- **Photo tiles** (the actual cleaned photo): a confidence chip — green when the
  model is sure, amber "upright?" when not (with an amber outline) so the eye
  goes straight to the uncertain ones.
- Per tile: a **side dropdown** (reclassify front/back/centre), a **rotate**
  button (90° per press), and a **delete-photo** button for dud shots.
- **Listing fields**, all editable. Condition and price are amber-labelled as
  the only required-by-user fields. A **copy-to-clipboard** button copies the
  drafted listing.
- **Approve this DVD** / **Skip for now** per card; **Save all approved** writes
  every approved DVD at once. **‹ DVD 1 of N ›** steps through the batch.

**Behaviour:** reclassify / rotate / edits update the preview **instantly**;
nothing is written to disk until Approve / Save.

## 5. Output

```
processed/run_<date>_<time>/
├─ King Kong Escapes/
│   ├─ king-kong-escapes_front.jpg
│   ├─ king-kong-escapes_back.jpg
│   ├─ king-kong-escapes_center.jpg
│   └─ listing.txt            # copy-paste ready, one DVD
├─ <next DVD>/…
├─ batch_listings.csv         # one row per DVD, all fields
└─ run_log.json               # what happened, for debugging
```

- **Photos:** `title_side.jpg`, composited on a clean white square.
- **`listing.txt`:** drafted title line, description, attributes, barcode — the
  text the clipboard button copies.
- **`batch_listings.csv`:** one row per DVD (title, year, genre, region,
  runtime, barcode, description, blank condition, blank price, photo filenames).
  Used as a **personal tracking sheet** — listings are created **manually by
  copy-paste** on eBay's site, so the CSV is not formatted for bulk upload. A
  bulk-upload export could be added later without changing the pipeline.

## 6. Run experience

**Daily flow:**
1. Drop phone photos into the **`Images in`** folder.
2. **Double-click** the desktop shortcut / `.bat`.
3. It processes every photo (CV cleanup + AI calls on the GPU) and **opens the
   review screen in the browser** automatically.
4. Review, fix flags, Approve → Save.
5. Results appear in `processed/run_…`.

**Platform:** native **Windows** (no WSL/venv). Ollama for Windows serves the
model off the GPU. One-time setup: install Python for Windows, install Ollama +
`ollama pull qwen2.5vl:7b`, run one `pip install`
(`numpy opencv-python rawpy pillow pillow-heif pyzbar fastapi uvicorn requests`).

## 7. Edge cases & error handling

- **Ollama down / model missing:** launcher pings `localhost:11434` first; if
  unreachable, shows "Start Ollama, then retry" instead of crashing.
- **A4 sheet not found (low confidence):** the photo still appears in review
  showing the **raw** image with an amber "couldn't flatten — use as-is or
  skip?" prompt. Nothing is silently dropped (V2 skipped these).
- **Title unreadable:** group by capture-time, label "Untitled DVD N".
- **No barcode:** field left blank, no error.
- **Unsupported / corrupt file:** logged to `run_log.json`, surfaced in review
  as a skipped item with the reason.

## 8. Code structure

```
dvdflip/
  __main__.py      # launcher: run batch, start web app, open browser
  loader.py        # stage 1a: decode DNG/JPG/HEIC, orientation
  flatten.py       # stage 1b: detect A4, warp to true A4, mm scale
  clean.py         # stage 2: white balance, crop DVD, composite, barcode
  vision.py        # stage 3: call Ollama, parse JSON, listing draft
  listing.py       # grouping + listing.txt + batch_listings.csv
  webapp.py        # FastAPI review screen + save endpoints
  config.py        # constants, model tag, paths, thresholds
  templates/review.html
tests/
  test_pipeline.py # regression on the 3 sample DNGs (CV real, model mocked)
samples/           # the 3 King Kong Escapes DNGs
```

## 9. Verification

- The **3 *King Kong Escapes* DNGs** are the regression fixture. The pipeline
  must: produce 3 clean white-background photos, classify them front / back /
  centre, read barcode `0025192828928`, and group them as a single DVD.
- CV stages run for real against the samples; the vision-model call is **mocked**
  to a fixed JSON answer so the test is deterministic and runs offline.
- Per-module unit tests where it adds value (A4 detection confidence on the
  samples, JSON parsing/validation in `vision.py`, grouping logic in
  `listing.py`).

## 10. Explicitly out of scope (YAGNI)

- eBay bulk-CSV / File Exchange upload format.
- Barcode → online title lookup (UPC API).
- Multi-user, cloud, or mobile versions.
- Any reuse of `ebayflip_V2.py` at runtime (reference only).
