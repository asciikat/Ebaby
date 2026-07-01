# Crop Studio — full pipeline merge (RAW ingest, barcode/eBay, smarter crop, single exe)

_Design doc. 2026-07-02._

## 1. Purpose

A separate, older pipeline lives in WSL Debian (`Pipeline/1_prep` +
`Pipeline/2_barcode_ebay`): rename shots into ordered sets (`bn_rename.py` for
2-shot new/sealed stock, `sh_rename.py` for 3-shot used/opened stock) → RAW
color-correction (`Colorprodawgv1.py`) → barcode crop/decode + eBay Browse API
lookup + CSV export (`pipeline_2.sh`). It never crops/composites to a clean
white-background photo — that's cropstudio's job, done separately.

This design merges the useful parts of that pipeline into cropstudio (the
active, actively-developed app) and packages the result as a single Windows
executable, so there's one app instead of a WSL script pipeline + a separate
browser tool. It also fixes two problems found while testing cropstudio in
this session: an auto-detect crop that can silently produce a near-blank crop,
and a title-resolution flow with no grounding (Qwen can hallucinate a title
from a bad crop, e.g. read "The Pool Boy" off a mis-cropped "Sexy Beast"
cover).

## 2. Scope (locked decisions)

| Decision | Choice |
|----------|--------|
| Packaging | Single Windows `.exe`: cropstudio's existing FastAPI+HTML app, embedded in a native window via `pywebview`, built with PyInstaller. No console, no separate browser tab. Dev mode (`Run Crop Studio.bat` → `python -m cropstudio`) keeps working unchanged. |
| RAW input | Supported. `.NEF`/`.DNG` decoded + color-corrected server-side (white-balance from surrounding paper + S-curve/saturation/sharpen, ported from `Colorprodawgv1.py`) before the crop UI ever sees the shot. JPG input skips this stage entirely. |
| Barcode decode | Reuse `redboxflip.barcode`'s existing 3-decoder consensus (pyzbar/zxingcpp/opencv). Do **not** port the old pipeline's single-pass `dvd_barcode_scanner.py` — redboxflip's is already more robust. |
| **Title resolution priority** | **Barcode + eBay Browse API match is primary** (grounded in real catalog data). Qwen vision (`redboxflip.vlm.title_from_cover`) is **fallback only** — runs *only* when the Back-cover barcode fails to decode, or decodes but eBay has no match for it. Qwen **never** overrides a barcode-resolved title. |
| eBay lookup scope | Price + specifics (genre/actor/studio/rating/etc, via `ebay_api.py` ported logic) attached to every DVD regardless of which title source won. Never used to overwrite a title that came from Qwen. |
| New/used stock tagging | Auto-inferred by **shot count at flush time**: 2 shots (Front+Back only) = new/sealed; 3 shots (+ Inside) = used/opened. Replaces manually running `bn_rename` vs `sh_rename`. Tag is a manifest/CSV field, not a filename convention (cropstudio already names files by resolved title). |
| Mismatch guard | Best-effort: when the title came from barcode (i.e. Back's barcode resolved it), also read text off the Front cover and check whether **at least one word of 4+ characters** from the resolved title (case-insensitive, stripped of punctuation) appears in the Front-cover text. No match -> **non-blocking warning banner** (matches this codebase's existing best-effort pattern in `vlm.py`) — catches a Back shot from a different physical DVD than Front/Inside. User can override and save anyway. |
| Crop auto-detect | Add a DVD-case aspect-ratio prior (reuse the ~0.711 ratio already proven in `redboxflip.scan`) to `autoDetect()`'s contour scoring, so a spurious thin high-contrast band (e.g. glare/reflection lines) can't out-score the true case boundary just by being a cleaner rectangle. |
| Crop content guard | New backstop, independent of the above: if an accepted crop's real (non-white) content occupies less than **40% of the canvas height** (non-white-row span ÷ canvas height — the exact measurement used to diagnose the Pool Boy/Sexy Beast bug, where the two bad crops measured 0.20 and 0.27 against a normal range of 0.5–0.95), block Save & Next with a clear warning instead of silently saving a near-blank crop. |
| Output layout | **Flat.** No per-DVD subfolders (this replaces the 2026-06-23 design's `processed/<run>/<Title>/` layout). Every image for a run saves directly into the run's export folder; filenames carry the resolved title exactly as read, never altered/slugified/truncated. |
| Run manifest | New `run_manifest.json` per run, appended per DVD as each stage completes: shot count/new-used tag, WB gains used (if RAW), barcode, which title source won (barcode/qwen), eBay match, crop-guard warnings, output filenames. Supplements (doesn't replace) the existing per-DVD `ebay_listing.txt/csv`. |
| Batch queue view | New UI panel: thumbnails of every DVD in the run + live per-stage status (queued/color-corrected/barcode found/priced/cropped/saved), backed by the run manifest. |
| Credentials | eBay client ID/secret + Gemini key move into this repo's own `.env` (gitignored). The WSL `.env` was exposed in this chat session and must be rotated — not reused as-is. |

Out of scope (YAGNI for this build): editing already-saved DVDs; multi-browser
concurrent batches (still a single-user local tool); the old pipeline's
rename-by-EXIF-capture-time ordering (cropstudio already orders shots via
manual slot assignment in the UI, so file-order inference isn't needed).

## 3. Architecture

Extends the existing `cropstudio/` package (FastAPI + `static/index.html`),
still importing `redboxflip` directly for barcode/vlm/naming/imaging. New
modules:

```
cropstudio/
  colorcorrect.py   NEW  RAW decode + white-balance + enhance (ported from Colorprodawgv1.py)
  ebay.py           NEW  eBay Browse API client (ported from ebay_api.py) + EbayUnavailable
  ebay_config.py    NEW  reads EBAY_CLIENT_ID/SECRET/MARKETPLACE_ID from this repo's .env
  manifest.py       NEW  run_manifest.json read/append per DVD
  service.py        EXT  resolve_title() becomes barcode-primary w/ Qwen fallback;
                          new/used tag from shot count; flat output path (no <Title>/ subfolder)
  app.py            EXT  new routes: raw-preview handling in /shot, /manifest (for queue view),
                          reuses existing /finish for the new "Finish DVD (2 shots only)" button
  static/index.html EXT  autoDetect() gets aspect-ratio prior + content-fraction guard;
                          new Add-Images/Export-Folder front page; batch queue thumbnail panel;
                          "Finish DVD (2 shots only)" button (mid-batch, not just batch-end)
  __main__pyw__.py  NEW  pywebview entry point for the PyInstaller exe build
```

### 3.1 Packaging (pywebview + PyInstaller)

`cropstudio/__main__pyw__.py` starts uvicorn on a background thread bound to
`127.0.0.1:<port>` (port picked the same way `__main__.py` does today), then
opens a `pywebview` window pointed at that URL — no console, no browser tab.
PyInstaller builds this entry point into a single `.exe`. Known packaging
risks to validate early: `rawpy` (bundles `libraw`), `rembg`/`onnxruntime`
(large, may need `--collect-all`), `pyzbar` (needs its bundled `zbar` DLL),
`zxingcpp` (compiled wheel). `python -m cropstudio` (uvicorn only, browser tab)
keeps working for dev — the pywebview entry point is additive, not a
replacement.

### 3.2 Title resolution flow (extends `service.resolve_title`)

```
Back shot uploaded
  -> barcode.decode(bgr)                              (redboxflip.barcode, existing)
  -> if barcode found:
       ebay.lookup_barcode(barcode)                    (new, best-effort, network)
       if eBay match found:
         title = match.title                           # PRIMARY, grounded
         price/specifics recorded for CSV
       else:
         title = None                                  # falls through to Qwen below
     if barcode not found or no eBay match:
       title = vlm.title_from_cover(front_bgr)          # FALLBACK (existing Qwen path, unchanged)
  -> if title resolved via barcode:
       front_text = vlm.read_text_best_effort(front_bgr)   # new, best-effort only
       if front_text doesn't loosely match title:
         emit warning (non-blocking) to the manifest + UI toast
```

`ebay.lookup_barcode` follows the existing `EbayUnavailable` pattern from the
ported `ebay_api.py`: network failure never blocks the save, it just means no
eBay-sourced title/price this DVD (falls through to Qwen for title; price/
specifics simply absent from the CSV row).

### 3.3 New/used tagging

`BatchState.finish()` (already implemented, currently used for the trailing
partial DVD at batch-end) is reused as-is for a new **mid-batch** "Finish DVD
(2 shots only)" button — no new server logic needed, just a new UI trigger
point. Whatever shot count is in the flushed buffer (2 or 3) becomes the
`new_stock` / `used_stock` tag written to the manifest and CSV.

### 3.4 Crop auto-detect improvements

In `autoDetect()` ([cropstudio/static/index.html](../../../cropstudio/static/index.html)):
add an aspect-ratio term to the existing `score = fill * (area / imgArea)`
so a candidate box whose aspect ratio is far from ~0.711 is penalized, making
it harder for a thin spurious band to win over the true (if noisier) case
boundary. Separately, add a content-fraction check at crop-accept time
(client-side, same measurement used to diagnose the Pool Boy/Sexy Beast bug:
non-white-row span ÷ canvas height) — below **40%**, block Save & Next
with a warning rather than silently accepting a near-blank crop.

### 3.5 Output layout change

Per DVD, files now save directly under the run folder:
`processed/run_<ts>/<Title> - Back Cover.jpg` (etc.) — no `<Title>/`
subfolder. `_unique_dir`'s collision handling moves to filename-level (append
` (2)` to the title portion, not the folder) since there's no longer a folder
to disambiguate.

## 4. Error handling

- eBay unreachable/rate-limited/no match → falls through to Qwen for title;
  price/specifics simply omitted from that DVD's CSV row. Never blocks a save.
- Qwen unreachable/no title → existing fallback (`Untitled DVD <n>`) unchanged.
- RAW decode failure (corrupt file, unsupported variant) → that shot is
  skipped with a clear toast, same pattern as existing image-decode failures.
- Crop content-guard trip → blocks Save & Next with a specific message
  ("crop looks mostly blank — check the corners"), not a silent save.
- Mismatch guard trip → non-blocking warning banner; user can save anyway.

## 5. Testing

TDD per this project's existing convention (`tests/test_cropstudio_*.py`
pattern) — each new module gets a failing-test-first pass:
- `colorcorrect.py`: white-balance gains from a synthetic bordered image,
  RAW decode mocked/skipped in CI (no real `.NEF` fixture needed for the
  correction-math tests).
- `ebay.py`: `EbayUnavailable` on network failure (mirrors `ebay_api.py`'s
  existing test pattern), successful match parsing from a fixture response.
- `service.resolve_title`: barcode+eBay-found → Qwen never invoked (assert
  the mock wasn't called); barcode-not-found → Qwen invoked; barcode-found
  but no eBay match → Qwen invoked; Qwen never overrides a barcode title.
- New/used tagging: 2-shot flush → `new_stock=True`; 3-shot flush →
  `used_stock=True`.
- Manifest: appends one entry per DVD, valid JSON after a full run.
- Crop guard (JS): out of scope for the Python test suite; verified manually
  per the existing manual end-to-end checklist, plus a documented repro case
  (re-run the "Pool Boy" source photo through the improved `autoDetect`).
- Manual end-to-end: real RAW batch through Add Images → Export Folder →
  crop → save, confirm flat output, manifest, and CSV all agree.

## 6. Related follow-up (not part of this build)

PyInstaller packaging risk (rawpy/rembg/onnxruntime/pyzbar/zxingcpp bundling)
may need its own investigation spike before the exe build is attempted — flag
if any of these fail to bundle cleanly, that becomes a separate task rather
than blocking this design.
