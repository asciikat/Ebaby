# Crop Studio — batch browser cropper with auto-save + Qwen naming

_Design doc. 2026-06-23._

## 1. Purpose

The slider-based browser cropper (`ebaby_gemini_fixed.html`) produces cleaner,
more controllable crops than redboxflip's automatic rembg pipeline — because a
human drags the 4 corners, it never suffers the auto-detect failure modes (e.g.
the multi-disc "inside" shot that rembg crops down to a single disc). The user
wants to keep that manual-crop quality but make it usable for a whole batch:

1. **Batch navigation** — load many images, step through them with Next/Prev.
2. **Auto-save** — write each finished crop into
   `C:\Users\mardi\Documents\Ebay code\processed` automatically, no manual
   "download" per file.
3. **Qwen naming** — read the DVD title with the local Qwen model and name the
   files/folders accordingly, reusing redboxflip's existing naming + title logic.

A browser page opened as `file://` cannot write to disk or call Ollama
(CORS). The solution is a small local server that serves the page (same origin)
and handles disk + Qwen on the user's behalf.

## 2. Scope (locked decisions)

| Decision | Choice |
|----------|--------|
| Batch input | Multi-file upload/drag (a `File[]`), not a folder path |
| Save mechanism | Local Python helper (FastAPI on `127.0.0.1:8765`) |
| Save trigger | **Auto-save on Next** (Next = save current crop, then advance) |
| Qwen scope | Group shots into DVDs of 3 (Back→Front→Inside), one title read per DVD |
| Output layout | Subfolder per DVD: `processed/<run>/<Title>/<Title> - <Face>.jpg` |
| Helper design | **Reuse `redboxflip` modules** (`naming`, `vlm`, `imaging`, `config`) |

Out of scope (YAGNI for v1): per-shot face override dropdown (faces follow the
fixed Back→Front→Inside rhythm, matching `Images in/README.md`); barcode decode;
full eBay field extraction; editing already-saved DVDs. These can be layered on
later; the main pipeline already does them.

## 3. Architecture

One new package `cropstudio/` plus a launcher. The server imports `redboxflip`
directly (same `.venv`), so naming/title logic stays in sync with the main app.

```
Browser (cropstudio/static/index.html)          Server (cropstudio/app.py, FastAPI)
─────────────────────────────────────           ───────────────────────────────────
 multi-file upload -> File[]                      GET  /            -> index.html
 crop/rotate/finish on <canvas>                   POST /shot        -> buffer + flush
 Next: canvas.toBlob(jpeg) --------------------->   (image bytes, slot 0/1/2)
        then advance to next file                  on slot==2 (or Finish):
 toast <-------- resolved title / error  <-------    title = vlm.title_from_cover(front)
                                                     dir   = processed/<run>/<safe title>/
                                                     write 3 JPEGs via imaging.save_jpeg
```

- **Single origin**: the page is served from `http://127.0.0.1:8765/`, so
  `fetch('/shot')` is same-origin — no CORS config, no `file://` limitations.
  Qwen is reached server-side via the existing `redboxflip.vlm` (`curl` →
  `127.0.0.1:11434`), so the browser never talks to Ollama directly.

### 3.1 Components

- **`cropstudio/static/index.html`** — `ebaby_gemini_fixed.html` evolved:
  - Multi-file `<input type="file" multiple>` + drag-drop onto the page.
  - A `File[]` queue with an index; position indicator `Shot 3 / 9 — Inside`
    (face derived from `index % 3` → 0 Back, 1 Front, 2 Inside).
  - Existing controls unchanged: draggable 4-corner crop, auto-detect, rotate
    90/180, studio-finish sliders (shave/radius/feather/padding), 1:1 square.
  - **Next** button: `canvas.toBlob('image/jpeg', 0.95)` → `POST /shot` with the
    slot index → advance to next file (non-blocking; a toast reports the result).
    **Prev** re-loads the previous file for re-cropping (does not un-save; see §6).
  - On the final file, Next is replaced by **Finish batch** which flushes any
    partial (1–2 shot) DVD still buffered on the server.
- **`cropstudio/app.py`** — FastAPI app:
  - `GET /` → serve `static/index.html`. `GET /static/*` for assets if split out.
  - `POST /shot` (multipart: `image` file, `slot` int) → see §4.
  - `POST /finish` → flush the current partial buffer.
  - `GET /health` → `{ok, qwen_available, output_dir}` for a startup banner.
  - A single in-process `BatchState` (the server handles one browser at a time —
    a local single-user tool).
- **`cropstudio/__main__.py`** — `python -m cropstudio` starts uvicorn on 8765,
  resolves the output dir, prints the URL.
- **`Run Crop Studio.bat`** — native Windows launcher (mirrors the fixed
  `Run DVD Flip.bat`): runs `.venv\Scripts\python.exe -m cropstudio` and opens
  the default browser to `http://127.0.0.1:8765/`.

### 3.2 Reused redboxflip functions (real signatures)

- `redboxflip.config.load_settings() -> Settings` — for `qwen_model`,
  `ollama_url`, `title_engine`, `jpeg_quality`.
- `redboxflip.vlm.available() -> bool` and
  `redboxflip.vlm.title_from_cover(bgr, settings) -> str | None` — Qwen title read
  on the Front shot (decode POSTed JPEG with `cv2.imdecode` → BGR).
- `redboxflip.titles.clean_title(raw) -> str` — normalise the title string.
- `redboxflip.naming.safe_stem(title)` and
  `redboxflip.naming.output_filename(title, face) -> "<safe> - <Label>.jpg"`
  (`FACE_FILE_LABEL`: Back Cover / Front Cover / Inside).
- `redboxflip.models.Face`, `FACE_ORDER = [BACK, FRONT, INSIDE]`.
- `redboxflip.imaging.save_jpeg(pil_image, path, quality)`.

### 3.3 Output directory resolution

`Settings.output_dir` defaults to a **WSL path** (`/mnt/c/...`) and may be
overridden in `~/.redboxflip.json`. Crop Studio runs natively on Windows, so it
does **not** trust that value. It resolves the output root as:

1. `CROPSTUDIO_OUTPUT` env var if set, else
2. `<project root>/processed` (the project root = the parent of the `cropstudio`
   package), giving `C:\Users\mardi\Documents\Ebay code\processed`.

Each launch creates a run folder `processed/run_<YYYYmmdd_HHMMSS>/` (same scheme
as the main pipeline) so repeated sessions never collide.

## 4. Data flow — `POST /shot`

The browser sends `(image_bytes, dvd_index, slot)` where `dvd_index =
floor(fileIndex / 3)` and `slot = fileIndex % 3` (0 Back, 1 Front, 2 Inside).
`BatchState` holds: `run_dir`, the `current_dvd_index`, a `buffer` keyed by slot
(`{0: bytes, 1: bytes, 2: bytes}` for the current DVD only), and a `dvd_counter`.

Keying the buffer by slot (not an append list) is deliberate: re-cropping a shot
via Prev→Next re-POSTs the same `(dvd_index, slot)` and simply **overwrites** the
slot, so navigation never duplicates or corrupts a DVD.

1. Receive `(image_bytes, dvd_index, slot)`.
2. If `dvd_index > current_dvd_index` and the buffer is non-empty, **flush the
   current DVD first** (it is now complete or partial) — see step 4 — then start
   the new DVD. Set `current_dvd_index = dvd_index`.
3. Store `buffer[slot] = image_bytes`. If the buffer does not yet hold all 3
   slots, return `{status:"buffered", have:[…]}` and stop.
4. **Flush** (buffer has all 3 slots, or forced by step 2 / `POST /finish`):
   a. Pick the title source: Front (`slot 1`) if present, else Back, else the
      remaining shot.
   b. `cv2.imdecode` → BGR; `title = clean_title(vlm.title_from_cover(bgr,
      settings))` when `vlm.available()`; else fall back (see §5).
   c. `dvd_counter += 1`; `dvd_dir = run_dir / unique(safe_stem(title))`, where
      `unique()` appends ` (2)`, ` (3)`… if the folder already exists.
   d. For each buffered slot, decode bytes → PIL, `save_jpeg` to
      `dvd_dir / output_filename(title, FACE_ORDER[slot])`.
   e. Clear the buffer; return `{status:"saved", title, dir, files:[…]}`.

The browser shows a toast from the response: `Saved "Goober and the Ghost
Chasers" (3 files)`, or the buffered/partial state. Because flushing the last
shot of a DVD also runs the ~7–9 s Qwen title read, the browser has already
advanced (non-blocking) and the toast arrives when the save completes.

## 5. Error handling

- **Qwen unreachable / times out / unusable answer** → title falls back to
  `Untitled DVD <dvd_counter>` (mirrors redboxflip's existing fallback). Files are
  still saved. The toast flags it: `Saved as "Untitled DVD 2" — Qwen gave no title`.
- **`POST /shot` fails (server down, fetch error)** → the browser shows a red
  toast with a **Retry** button and does **not** advance; the current crop stays
  on the canvas so nothing is lost.
- **Image decode failure on the server** → `400` with a clear message; that shot
  is skipped, the toast says so, the rest of the DVD still saves.
- **Disk write error** (permissions, path) → `500` with the OS error text
  surfaced in the toast.

## 6. Edge cases

- **Partial final DVD** (batch length not a multiple of 3): `Finish batch` flushes
  1 or 2 buffered shots, naming from whichever face is available (prefer Front).
- **Re-cropping with Prev within the current (un-flushed) DVD**: re-POSTs the same
  `(dvd_index, slot)` and overwrites that slot in the buffer — fully supported, no
  duplicate. **Re-cropping a DVD that has already flushed** (e.g. stepping back
  three files): the server starts it as a fresh `dvd_index` and writes a new
  folder rather than patching the old one. v1 keeps this simple; a true "re-open
  saved DVD" edit is out of scope. Documented in the page's help text.
- **Empty title from Qwen on a stylised cover**: handled by the fallback in §5.
- **Single-browser assumption**: the server holds one `BatchState`; opening a
  second tab shares it. Acceptable for a local single-user tool (documented).

## 7. Testing

- `tests/test_cropstudio_app.py` (FastAPI `TestClient`, monkeypatch
  `vlm.title_from_cover` to avoid a live Ollama call):
  - POST 3 small synthetic JPEGs (slots 0/1/2) → assert a `<title>/` folder with
    `… - Back Cover.jpg`, `… - Front Cover.jpg`, `… - Inside.jpg`.
  - Qwen-returns-None → folder named `Untitled DVD 1`, 3 files present.
  - Partial batch via `/finish` (2 shots) → 2 files saved, named from Front.
  - Duplicate title twice → second lands in `<title> (2)/`.
  - Bad image bytes → `400`, other shots in the DVD still save.
- Output dir resolution unit test: `CROPSTUDIO_OUTPUT` honored; default is
  `<project>/processed`.
- Manual verification: launch the `.bat`, run a real batch (the Goober + a closed
  DVD), confirm `processed/run_*/…` files appear with correct names while the
  server window stays open.

## 8. Related follow-up (separate task, not part of this build)

The multi-disc **inside** crop bug in `redboxflip/scan.py` (rembg segments one
disc; `_trim_empty_border` eats the rest, producing an inside shot smaller than
the covers — violating the "inside is always ~2× the cover" invariant) is tracked
as its own debugging task. Crop Studio sidesteps it by manual cropping but does
not fix it.
