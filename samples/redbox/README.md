# Red-box sample scans (test fixtures + reference)

Drop the **original** phone-scanner JPGs of the OPEN WATER DVD on the red-box
template here, with these **exact filenames** (lowercase). The test suite and the
end-to-end regression in the build plan look for these names:

| Filename       | What it is                                              |
|----------------|---------------------------------------------------------|
| `back.jpg`     | Back cover on the template — **must show the barcode** (`9 325336 022306`). This is the scan the barcode reader is tuned against. |
| `front.jpg`    | Front cover on the template (OPEN WATER artwork).       |
| `inside.jpg`   | Case opened flat — disc + scene-index sleeve.           |
| `template.jpg` | The **blank** A3 red-box template (no DVD). Reference only — handy for tuning red detection. |

Matching the four images you sent:
- back cover (barcode visible)            → save as `back.jpg`
- front cover (OPEN WATER art)            → save as `front.jpg`
- blank template (Front/Back/Centre)      → save as `template.jpg`
- inside (disc + "SCENE INDEX")           → save as `inside.jpg`

Use the **full-resolution originals**, not screenshots — the barcode reader needs
the real pixels. Once these are here, run `pytest tests/test_rb_end_to_end.py -v`
and the skipped tests will run for real.
