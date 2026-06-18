from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = SCRIPT_DIR / "Images in"
DEFAULT_OUTPUT = SCRIPT_DIR / "processed"

RAW_EXT = {".dng"}
PIL_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
HEIC_EXT = {".heic", ".heif"}
SUPPORTED_EXT = RAW_EXT | PIL_EXT | HEIC_EXT

A4_SHORT_MM = 210.0
A4_LONG_MM = 297.0
A4_RATIO = A4_LONG_MM / A4_SHORT_MM
PX_PER_MM = 10
CENTER_LONG_MM = 230.0  # DVD long edge >= this (mm) -> open case / center spread

AUTO_CONFIDENCE = 0.72
REVIEW_CONFIDENCE = 0.45

OLLAMA_URL = "http://localhost:11434"
MODEL_TAG = "qwen3.5:4b"
VISION_TIMEOUT = 180
# Keep the VLM's VRAM footprint small enough to fit an 8 GB GPU alongside the
# desktop. Larger values read fine print better but risk spilling to system RAM
# (catastrophically slow). 512 px / 2048 ctx runs the 4b fully on-GPU at ~2 s/image.
VISION_IMAGE_MAX_SIDE = 512
VISION_NUM_CTX = 2048

OUTPUT_MAX_DIM = 1600
JPEG_QUALITY = 92
PADDING_PCT = 7
GROUP_TIME_GAP_S = 30.0
