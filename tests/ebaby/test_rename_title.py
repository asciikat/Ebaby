from pathlib import Path

from ebaby.stages import rename_title as rt


def _touch(d, *names):
    for n in names:
        (d / n).write_bytes(b"x")


def test_plan_renames_used_stock_with_ebay_title(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _touch(src, "a_front.png", "a_back.png", "a_inside.png")
    sets = {"a": [("front", src / "a_front.png"),
                  ("back", src / "a_back.png"),
                  ("inside", src / "a_inside.png")]}
    key_to_barcode = {"a": "400638133393"}
    key_to_title = {"a": "The Matrix"}
    plan, notes = rt.plan_renames(sets, key_to_barcode, key_to_title, dst)
    names = sorted(new.name for _, new in plan)
    assert names == ["The_Matrix_back.png", "The_Matrix_front.png", "The_Matrix_inside.png"]
    assert notes == []


def test_plan_renames_new_stock_tags_new_suffix(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _touch(src, "01_front.png", "01_back.png")
    sets = {"01": [("front", src / "01_front.png"), ("back", src / "01_back.png")]}
    plan, notes = rt.plan_renames(sets, {"01": "400638133393"}, {"01": "The Matrix"}, dst)
    names = sorted(new.name for _, new in plan)
    assert names == ["The_Matrix_back_new.png", "The_Matrix_front_new.png"]


def test_plan_renames_no_ebay_match_falls_back_to_barcode(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _touch(src, "a_front.png", "a_back.png")
    sets = {"a": [("front", src / "a_front.png"), ("back", src / "a_back.png")]}
    plan, notes = rt.plan_renames(sets, {"a": "400638133393"}, {"a": ""}, dst)
    names = sorted(new.name for _, new in plan)
    assert names == ["400638133393_back.png", "400638133393_front.png"]
    assert notes == [("a", "no eBay match — named by barcode 400638133393")]


def test_plan_renames_no_barcode_uses_placeholder(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _touch(src, "a_front.png", "a_back.png")
    sets = {"a": [("front", src / "a_front.png"), ("back", src / "a_back.png")]}
    plan, notes = rt.plan_renames(sets, {"a": None}, {"a": ""}, dst)
    names = sorted(new.name for _, new in plan)
    assert names == ["NoBarcode_a_back.png", "NoBarcode_a_front.png"]
    assert notes == [("a", "no barcode found — moved with placeholder name, needs manual ID")]


def test_plan_renames_dedupes_colliding_slugs(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _touch(src, "a_front.png", "a_back.png", "b_front.png", "b_back.png")
    sets = {
        "a": [("front", src / "a_front.png"), ("back", src / "a_back.png")],
        "b": [("front", src / "b_front.png"), ("back", src / "b_back.png")],
    }
    key_to_title = {"a": "The Matrix", "b": "The Matrix"}
    plan, _ = rt.plan_renames(sets, {"a": "1", "b": "2"}, key_to_title, dst)
    names = sorted(new.name for _, new in plan)
    assert names == ["The_Matrix_2_back.png", "The_Matrix_2_front.png",
                      "The_Matrix_back.png", "The_Matrix_front.png"]


def test_apply_renames_moves_files_and_skips_existing_targets(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    _touch(src, "a_front.png")
    (dst / "The_Matrix_front.png").write_bytes(b"already there")
    plan = [(src / "a_front.png", dst / "The_Matrix_front.png")]
    done = rt.apply_renames(plan)
    assert done == 0
    assert (src / "a_front.png").exists()  # not clobbered, not moved
