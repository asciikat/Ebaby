"""Batch folder layout, state.json persistence, and stage-transition rules.

A batch is one run of the pipeline: a folder under BATCHES_ROOT holding the
originals, every intermediate stage's output, and a state.json recording
which stage the batch is currently at. Stages only ever move forward
(upload -> rename -> color -> barcode -> ebay -> crop -> done); re-running a
stage clears and regenerates only that stage's own output folder (handled by
the stage modules themselves, not here).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

STAGES = ["upload", "rename", "color", "barcode", "ebay", "crop", "done"]

BATCHES_ROOT = Path("/home/M/Ebaby Runs")

_SUBFOLDERS = (
    "1_originals/used", "1_originals/new", "2_color",
    "3_barcodes", "4_renamed", "5_cropped",
)


class BatchError(RuntimeError):
    """Raised when a stage is requested out of order or a batch is malformed."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def batch_dir(name: str) -> Path:
    return BATCHES_ROOT / name


def create_batch(name: str) -> Path:
    d = batch_dir(name)
    if d.exists():
        raise BatchError(f"batch '{name}' already exists")
    for sub in _SUBFOLDERS:
        (d / sub).mkdir(parents=True, exist_ok=True)
    write_state(d, {"stage": "upload", "sets": {}, "created": _now()})
    return d


def read_state(d: Path) -> dict:
    p = d / "state.json"
    if not p.exists():
        raise BatchError(f"no state.json in {d}")
    return json.loads(p.read_text(encoding="utf-8"))


def write_state(d: Path, state: dict) -> None:
    (d / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def advance_stage(d: Path, from_stage: str, to_stage: str) -> dict:
    state = read_state(d)
    if state["stage"] != from_stage:
        raise BatchError(
            f"batch is at stage '{state['stage']}', expected '{from_stage}'"
        )
    if to_stage not in STAGES:
        raise BatchError(f"unknown stage '{to_stage}'")
    state["stage"] = to_stage
    state["updated"] = _now()
    write_state(d, state)
    return state


def list_batches() -> list:
    if not BATCHES_ROOT.exists():
        return []
    return sorted(
        p.name for p in BATCHES_ROOT.iterdir()
        if p.is_dir() and (p / "state.json").exists()
    )
