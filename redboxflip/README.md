# redboxflip

Desktop app (Tkinter, runs in Ubuntu/WSLg) that turns phone photos of DVD cases
into clean 1:1 white-square eBay photos, named by title from the back-cover barcode.

## Run

    source ~/ebay-venv/venv/bin/activate
    python3 -m redboxflip                 # GUI
    python3 -m redboxflip -i IN -o OUT    # headless batch

`Run DVD Flip.bat` launches the GUI inside WSL.

## How to photograph (best results)

- **Plain white background**, case centred with **~3 cm clear white all around**.
  Keep the table edge, shadows, and clutter out of frame. No red boxes, no
  writing — the tool finds the case as the only object on the white.
- **Soft, even light** (window/indirect daylight or two lamps at 45°). Avoid
  harsh single light: it causes glare on the plastic and shadows the tool can
  mistake for content.
- **Full-resolution camera, straight overhead.** Do NOT use a downscaling
  "document scan" mode — the barcode needs ~2+ pixels per bar to read. Let the
  case fill a good part of the frame.
- One face per photo, in order: **Back, Front, Inside.** Back is first so its
  barcode names all three files. The face comes from the order, not from labels.

## Cutout engines

`geometric` (default) is best for a case on white — the convex hull of the
non-white content gives a clean, complete, artwork-intact cutout. Also available
in Settings / per-image in the editor: `grabcut`, `rembg` (AI matte), and `sam`
(MobileSAM box-prompt; set a checkpoint path). They fall back
geometric → manual if an engine is unavailable.

## Barcode → title

Auto-decode is best-effort (needs enough pixels on the barcode). When it can't
read it, type the 13 digits or the title once in the editor — it's cached and
names the files. Old red-box-template photos still process (the case is detected
by content; any stray red marker is cleaned at the cutout border).
