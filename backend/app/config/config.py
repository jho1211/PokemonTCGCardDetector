import os
import dotenv
from pathlib import Path
import contextvars

dotenv.load_dotenv()

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolo11n.pt")
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.2"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.45"))
YOLO_MAX_DETECTIONS = int(os.getenv("YOLO_MAX_DETECTIONS", "24"))
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cpu")

DEFAULT_TEMPLATE_ROOT = Path(__file__).resolve().parents[2] / "templates" / "set_symbols"
DEFAULT_METADATA_PATH = DEFAULT_TEMPLATE_ROOT / "metadata.json"
DEFAULT_SET_SYMBOL_OCR_MAP_PATH = DEFAULT_TEMPLATE_ROOT / "abbreviation_map.json"
SET_SYMBOL_TEMPLATE_DIR = Path(os.getenv("SET_SYMBOL_TEMPLATE_DIR", str(DEFAULT_TEMPLATE_ROOT)))
METADATA_PATH = Path(os.getenv("SET_SYMBOL_METADATA", str(DEFAULT_METADATA_PATH)))
SET_SYMBOL_MIN_SCORE = float(os.getenv("SET_SYMBOL_MIN_SCORE", "0.45"))
SET_SYMBOL_OCR_MAP_PATH = Path(os.getenv("SET_SYMBOL_OCR_MAP_PATH", str(DEFAULT_SET_SYMBOL_OCR_MAP_PATH)))
SET_SYMBOL_OCR_MIN_SCORE = float(os.getenv("SET_SYMBOL_OCR_MIN_SCORE", "0.78"))

CARD_RATIO_WIDTH_MM = 63.0
CARD_RATIO_HEIGHT_MM = 88.0
CARD_TARGET_WIDTH = int(os.getenv("CARD_TARGET_WIDTH", "630"))
CARD_TARGET_HEIGHT = int(round(CARD_TARGET_WIDTH * (CARD_RATIO_HEIGHT_MM / CARD_RATIO_WIDTH_MM)))