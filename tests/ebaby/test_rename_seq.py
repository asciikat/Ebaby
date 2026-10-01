import pytest

from ebaby.stages import rename_seq as rs


def _touch(d, *names):
    for n in names:
        (d / n).write_bytes(b"x")


def test_group_shots_pairs_raw_and_jpeg_by_stem(tmp_path):
    _touch(tmp_path, "IMG_0001.NEF", "IMG_0001.JPG", "IMG_0002.NEF")
    files = rs.find_images(tmp_path)
    shots = rs.group_shots(files)
    assert len(shots) == 2
    assert {p.name for p in shots[0]} == {"IMG_0001.NEF", "IMG_0001.JPG"}


def test_build_plan_used_stock_three_shot_letter_scheme(tmp_path):
    _touch(tmp_path, "s1.dng", "s2.dng", "s3.dng", "s4.dng", "s5.dng", "s6.dng")
    plan = rs.build_plan(tmp_path, set_size=3, scheme="letter", labels=rs.USED_LABELS)
    names = sorted(new.name for _, new in plan)
    assert names == ["a_back.dng", "a_front.dng", "a_inside.dng",
                      "b_back.dng", "b_front.dng", "b_inside.dng"]


def test_build_plan_new_stock_two_shot_number_scheme(tmp_path):
    _touch(tmp_path, "s1.dng", "s2.dng", "s3.dng", "s4.dng")
    plan = rs.build_plan(tmp_path, set_size=2, scheme="number", labels=rs.NEW_LABELS)
    names = sorted(new.name for _, new in plan)
    assert names == ["01_back.dng", "01_front.dng", "02_back.dng", "02_front.dng"]


def test_build_plan_uneven_count_raises(tmp_path):
    _touch(tmp_path, "s1.dng", "s2.dng", "s3.dng")  # not a multiple of 2
    with pytest.raises(ValueError, match="not a whole number of sets"):
        rs.build_plan(tmp_path, set_size=2, scheme="number", labels=rs.NEW_LABELS)


def test_apply_plan_renames_and_is_collision_safe(tmp_path):
    _touch(tmp_path, "s1.dng", "s2.dng")
    plan = rs.build_plan(tmp_path, set_size=2, scheme="number", labels=rs.NEW_LABELS)
    count = rs.apply_plan(plan)
    assert count == 2
    assert (tmp_path / "01_front.dng").exists()
    assert (tmp_path / "01_back.dng").exists()
    assert not (tmp_path / "s1.dng").exists()


def test_apply_plan_swap_does_not_clobber(tmp_path):
    # A already named 01_back, B already named 01_front -> plan should SWAP
    # them, which a naive single-pass os.rename would clobber.
    (tmp_path / "01_back.dng").write_bytes(b"A")
    (tmp_path / "01_front.dng").write_bytes(b"B")
    plan = [
        (tmp_path / "01_back.dng", tmp_path / "01_front.dng"),
        (tmp_path / "01_front.dng", tmp_path / "01_back.dng"),
    ]
    rs.apply_plan(plan)
    assert (tmp_path / "01_front.dng").read_bytes() == b"A"
    assert (tmp_path / "01_back.dng").read_bytes() == b"B"
