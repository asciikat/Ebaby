"""Core data types passed between pipeline stages and the GUI."""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class Face(str, Enum):
    BACK = "back"
    FRONT = "front"
    INSIDE = "inside"


# Fixed capture order: back first (barcode), then front, then inside.
FACE_ORDER = [Face.BACK, Face.FRONT, Face.INSIDE]

FACE_FILE_LABEL = {
    Face.BACK: "Back Cover",
    Face.FRONT: "Front Cover",
    Face.INSIDE: "Inside",
}


@dataclass
class Settings:
    input_dir: str = ""
    output_dir: str = ""
    cutout_engine: str = "geometric"        # "geometric" | "grabcut" | "rembg" | "sam"
    rembg_model: str = "isnet-general-use"
    sam_checkpoint: str = ""                # path to MobileSAM weights, if using SAM
    feather_px: int = 3
    margin_pct: float = 6.0
    jpeg_quality: int = 92
    max_edge_px: int = 1600
    default_region: str = "Region 4 (PAL, Australia)"
    colour_tidy: bool = True
    title_lookup: bool = True
    auto_open_output: bool = True
    auto_open_grid: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        known = {k: d[k] for k in cls().__dict__ if k in d}
        return cls(**known)


@dataclass
class ShotResult:
    input_path: str
    face: Face
    output_path: Optional[str] = None
    barcode: Optional[str] = None
    title: Optional[str] = None
    region: Optional[str] = None
    detect_method: str = ""
    detect_conf: float = 0.0
    cutout_method: str = ""
    rotation: int = 0
    status: str = "ok"                      # "ok" | "failed" | "needs_review"
    error: str = ""
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["face"] = self.face.value
        return d


@dataclass
class DvdGroup:
    index: int
    barcode: Optional[str]
    title: str
    region: str
    shots: list = field(default_factory=list)   # list[ShotResult]

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "barcode": self.barcode,
            "title": self.title,
            "region": self.region,
            "shots": [s.to_dict() for s in self.shots],
        }
