"""Entry point: GUI by default, headless batch with --input/--output."""
import argparse
import sys

from .config import load_settings
from .models import Settings


def build_parser():
    p = argparse.ArgumentParser(
        prog="redboxflip",
        description="Red-box DVD photo processor for eBay.")
    p.add_argument("-i", "--input", help="input folder of scans")
    p.add_argument("-o", "--output", help="output folder")
    p.add_argument("--engine", default=None,
                   help="cutout engine: rembg|sam|grabcut|geometric")
    p.add_argument("--no-colour", action="store_true", help="disable colour tidy")
    return p


def run_headless(input_dir, output_dir, engine=None, colour_tidy=None):
    from . import pipeline
    s = load_settings()
    s.input_dir = input_dir
    s.output_dir = output_dir
    if engine:
        s.cutout_engine = engine
    if colour_tidy is not None:
        s.colour_tidy = colour_tidy
    run_dir, groups = pipeline.run_batch(
        s, progress_cb=lambda d, t, n: print(f"[{d}/{t}] {n}"))
    print(f"Done. {len(groups)} DVD(s). Output: {run_dir}")
    return 0


def startup_report() -> str:
    from .barcode import available_decoders
    decoders = available_decoders()
    try:
        import rembg  # noqa: F401
        rembg_ok = "available"
    except Exception:
        rembg_ok = "MISSING (cutout falls back to GrabCut)"
    lines = [
        f"Barcode decoders: {', '.join(decoders) if decoders else 'NONE — install pyzbar/zxing-cpp + libzbar0'}",
        f"Cutout: rembg {rembg_ok}; GrabCut/geometric always available",
    ]
    return "\n".join(lines)


def main(argv=None):
    args = build_parser().parse_args(argv)
    print(startup_report(), file=sys.stderr)
    if args.input or args.output:
        if not (args.input and args.output):
            print("Both --input and --output are required for headless mode.",
                  file=sys.stderr)
            return 2
        return run_headless(args.input, args.output, engine=args.engine,
                            colour_tidy=(False if args.no_colour else None))
    # GUI mode
    from .gui.app import launch
    return launch()


if __name__ == "__main__":
    sys.exit(main())
