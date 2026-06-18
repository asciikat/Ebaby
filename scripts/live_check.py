"""No-browser live check for the dvdflip pipeline.

Runs the REAL pipeline (real Ollama vision calls, no mocks) over a folder of
photos and prints the model's per-photo decisions and the drafted listing for
each grouped DVD. Use it to sanity-check the live model without launching the
review web app — handy after changing ``MODEL_TAG`` in ``dvdflip/config.py``.

Usage (from the project root, with the venv):
    .\\.venv\\Scripts\\python.exe scripts\\live_check.py [input_folder]

Defaults to the bundled ``samples/`` folder.
"""
import sys
import tempfile
from pathlib import Path

# Allow running as a plain script (add project root to sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dvdflip import vision, config
from dvdflip.webapp import build_session


def main():
    inp = (Path(sys.argv[1]) if len(sys.argv) > 1
           else Path(__file__).resolve().parent.parent / "samples")

    print(f"Model : {config.MODEL_TAG}")
    if not vision.ollama_available():
        print("Ollama is not reachable at localhost:11434 — start it first.")
        return 1

    out = Path(tempfile.mkdtemp(prefix="dvdflip_live_"))
    print(f"Input : {inp}")
    print(f"Output: {out}\n")

    session = build_session(input_dir=inp, output_base=out, progress=print)

    print(f"\n=== {len(session.groups)} DVD group(s) ===")
    for gi, g in enumerate(session.groups, 1):
        print(f"\n--- DVD {gi}: {g.title!r}  (barcode {g.barcode or '-'}) ---")
        for p in g.photos:
            print(f"  {Path(p.source_path).name:32}  side={p.side:7} "
                  f"rot={p.rotation_cw:>3}  conf={p.confidence:.2f}  "
                  f"a4={p.a4_found}  title={p.title!r}")
        print(f"  listing_title : {g.listing_title}")
        print(f"  description   : {g.description}")
        print(f"  genre/region/runtime/studio: "
              f"{g.genre} / {g.region} / {g.runtime} / {g.studio}")
        print(f"  year          : {g.year}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
