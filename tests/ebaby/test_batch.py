import json
import pytest

from ebaby import batch


@pytest.fixture(autouse=True)
def isolated_root(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "BATCHES_ROOT", tmp_path / "Ebaby Runs")
    return tmp_path


def test_create_batch_makes_expected_subfolders():
    d = batch.create_batch("2026-07-04-run1")
    for sub in ("1_originals/used", "1_originals/new", "2_color",
                "3_barcodes", "4_renamed", "5_cropped"):
        assert (d / sub).is_dir()
    state = json.loads((d / "state.json").read_text())
    assert state["stage"] == "upload"


def test_create_batch_twice_raises():
    batch.create_batch("dup")
    with pytest.raises(batch.BatchError):
        batch.create_batch("dup")


def test_advance_stage_happy_path():
    d = batch.create_batch("run2")
    state = batch.advance_stage(d, "upload", "rename")
    assert state["stage"] == "rename"
    assert batch.read_state(d)["stage"] == "rename"


def test_advance_stage_wrong_from_stage_raises():
    d = batch.create_batch("run3")
    with pytest.raises(batch.BatchError):
        batch.advance_stage(d, "color", "barcode")  # batch is still at "upload"


def test_advance_stage_unknown_to_stage_raises():
    d = batch.create_batch("run4")
    with pytest.raises(batch.BatchError):
        batch.advance_stage(d, "upload", "not_a_real_stage")


def test_advance_stage_backward_raises():
    d = batch.create_batch("run5")
    batch.advance_stage(d, "upload", "color")
    with pytest.raises(batch.BatchError, match="backward"):
        batch.advance_stage(d, "color", "upload")


def test_read_state_missing_file_raises():
    with pytest.raises(batch.BatchError):
        batch.read_state(batch.batch_dir("never_created"))


def test_list_batches_returns_created_names_sorted():
    batch.create_batch("b_run")
    batch.create_batch("a_run")
    assert batch.list_batches() == ["a_run", "b_run"]


def test_list_batches_empty_root_returns_empty_list():
    assert batch.list_batches() == []
