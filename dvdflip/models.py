from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

VALID_SIDES = ("front", "back", "center", "spine", "other")

@dataclass
class ClassifyResult:
    side: str
    rotation_cw: int
    title: str
    year: Optional[int]
    confidence: float

@dataclass
class Photo:
    source_path: Path
    work_image: Optional[Path] = None
    side: str = "other"
    rotation_cw: int = 0
    confidence: float = 0.0
    barcode: Optional[str] = None
    size_mm: tuple = (0.0, 0.0)
    a4_found: bool = True
    title: str = ""
    year: Optional[int] = None
    deleted: bool = False
    extra_rotation_cw: int = 0
    orient_flip: bool = False
    orient_confident: bool = False

@dataclass
class DvdGroup:
    title: str = ""
    year: Optional[int] = None
    photos: list = field(default_factory=list)
    listing_title: str = ""
    description: str = ""
    genre: str = ""
    region: str = ""
    runtime: str = ""
    studio: str = ""
    barcode: str = ""
    condition: str = ""
    price: str = ""
    approved: bool = False
