import cv2
from pathlib import Path

from redboxflip import __main__ as cli
from rb_helpers import make_redbox_image


def test_build_parser_defaults():
    parser = cli.build_parser()
    args = parser.parse_args([])
    assert args.input is None and args.output is None


def test_run_headless_processes_a_folder(tmp_path, monkeypatch):
    in_dir = tmp_path / "in"; in_dir.mkdir()
    img = make_redbox_image()
    for n in ("1.jpg", "2.jpg", "3.jpg"):
        cv2.imwrite(str(in_dir / n), img)
    out_dir = tmp_path / "out"

    from redboxflip import pipeline
    monkeypatch.setattr(pipeline, "decode_barcode", lambda bgr: ("123", "t", 0))
    monkeypatch.setattr(pipeline, "resolve_title", lambda bc, **k: ("CLI DVD", "lookup"))

    code = cli.run_headless(str(in_dir), str(out_dir),
                            engine="geometric", colour_tidy=False)
    assert code == 0
    outputs = list((out_dir).rglob("*.jpg"))
    assert len(outputs) == 3
