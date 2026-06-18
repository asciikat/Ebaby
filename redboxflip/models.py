"""Core data types passed between pipeline stages and the GUI."""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class Face(str, Enum):
    BACK = "back"
    FRONT = "front"
    INSIDE = "inside"


# Confidence labels for an extracted field.
HIGH, MEDIUM, LOW, UNKNOWN = "High", "Medium", "Low", "Unknown"


@dataclass
class Field:
    """One extracted listing field with a confidence and a manual-check flag.

    `value` is "Unknown" when nothing legible was read. `needs_check` is True
    whenever the value should not be trusted blindly (unknown, low confidence,
    or a sensitive field the user told us never to guess).
    """
    value: str = "Unknown"
    confidence: str = UNKNOWN          # High | Medium | Low | Unknown
    needs_check: bool = True
    source: str = ""                   # face/decoder the value came from

    @property
    def known(self) -> bool:
        return bool(self.value) and self.value.strip().lower() != "unknown"


_SCAN_FIELDS = (
    "title", "format", "region", "pal_ntsc", "barcode", "rating",
    "release_year", "studio", "edition", "num_discs", "languages",
    "subtitles", "genre", "special_features", "notes", "condition",
)


@dataclass
class DvdScan:
    """Everything read from a DVD's photos, ready to render into a listing."""
    title: Field = field(default_factory=Field)
    format: Field = field(default_factory=Field)
    region: Field = field(default_factory=Field)
    pal_ntsc: Field = field(default_factory=Field)
    barcode: Field = field(default_factory=Field)
    rating: Field = field(default_factory=Field)
    release_year: Field = field(default_factory=Field)
    studio: Field = field(default_factory=Field)
    edition: Field = field(default_factory=Field)
    num_discs: Field = field(default_factory=Field)
    languages: Field = field(default_factory=Field)
    subtitles: Field = field(default_factory=Field)
    genre: Field = field(default_factory=Field)
    special_features: Field = field(default_factory=Field)
    notes: Field = field(default_factory=Field)
    condition: Field = field(default_factory=Field)
    suggested_title: str = ""
    included_items: str = ""
    tested: str = "Not tested - sold as is"
    raw_text: dict = field(default_factory=dict)   # face value -> all readable text

    def fields(self):
        """Yield (name, Field) for the named listing fields, in display order."""
        for name in _SCAN_FIELDS:
            yield name, getattr(self, name)

    def manual_checks(self):
        """Human-readable names of fields that still need a manual check."""
        labels = {"pal_ntsc": "PAL/NTSC", "num_discs": "number of discs",
                  "release_year": "release year",
                  "special_features": "special features"}
        return [labels.get(n, n.replace("_", " ")) for n, f in self.fields()
                if f.needs_check]

    def to_dict(self) -> dict:
        d = {n: asdict(f) for n, f in self.fields()}
        d["suggested_title"] = self.suggested_title
        d["included_items"] = self.included_items
        d["tested"] = self.tested
        d["raw_text"] = self.raw_text
        return d


# Fixed capture order: back first (barcode), then front, then inside.
FACE_ORDER = [Face.BACK, Face.FRONT, Face.INSIDE]

FACE_FILE_LABEL = {
    Face.BACK: "Back Cover",
    Face.FRONT: "Front Cover",
    Face.INSIDE: "Inside",
}


@dataclass
class Settings:
    input_dir: str = "/mnt/c/Users/mardi/Documents/Ebay code/Images in"
    output_dir: str = "/mnt/c/Users/mardi/Documents/Ebay code/processed"
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
    title_engine: str = "auto"              # "auto" (Qwen if reachable, else OCR) | "qwen" | "ocr"
    qwen_model: str = "qwen3.5:4b"
    ollama_url: str = "http://127.0.0.1:11434"
    qwen_extract: bool = True               # full Qwen field scan for the eBay listing
    auto_orient: bool = True                # auto-fix 180° flips (OSD + Qwen vote)
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
    ocr_title: Optional[str] = None
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
    scan: Optional[DvdScan] = None              # merged Qwen field extraction

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "barcode": self.barcode,
            "title": self.title,
            "region": self.region,
            "shots": [s.to_dict() for s in self.shots],
            "scan": self.scan.to_dict() if self.scan else None,
        }
