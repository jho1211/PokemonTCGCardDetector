import os
import dotenv
from pathlib import Path
import contextvars

dotenv.load_dotenv()


def _parse_bool(value: str | None, default: bool) -> bool:
	if value is None:
		return default
	return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_rotation_priority(raw: str | None) -> tuple[int, ...]:
	preferred: list[int] = []
	if raw:
		for token in raw.split(","):
			token = token.strip()
			if not token:
				continue
			try:
				turns = int(token)
			except ValueError:
				continue
			if turns < 0 or turns > 3 or turns in preferred:
				continue
			preferred.append(turns)

	for fallback in (0, 2, 1, 3):
		if fallback not in preferred:
			preferred.append(fallback)

	return tuple(preferred[:4])

PADDLEOCR_LANG = os.getenv("PADDLEOCR_LANG", "en")
PADDLEOCR_ENABLE_MKLDNN = _parse_bool(os.getenv("PADDLEOCR_ENABLE_MKLDNN"), False)
PADDLEOCR_USE_DOC_ORIENTATION_CLASSIFY = _parse_bool(os.getenv("PADDLEOCR_USE_DOC_ORIENTATION_CLASSIFY"), False)
PADDLEOCR_USE_DOC_UNWARPING = _parse_bool(os.getenv("PADDLEOCR_USE_DOC_UNWARPING"), False)
PADDLEOCR_USE_TEXTLINE_ORIENTATION = _parse_bool(os.getenv("PADDLEOCR_USE_TEXTLINE_ORIENTATION"), False)

OCR_ROTATION_PRIORITY = _parse_rotation_priority(os.getenv("OCR_ROTATION_PRIORITY", "0,2,1,3"))
OCR_COLLECTOR_EARLY_STOP_SCORE = float(os.getenv("OCR_COLLECTOR_EARLY_STOP_SCORE", "0.98"))
OCR_NAME_EARLY_STOP_SCORE = float(os.getenv("OCR_NAME_EARLY_STOP_SCORE", "0.96"))
OCR_ORIENTATION_EARLY_STOP_SCORE = float(os.getenv("OCR_ORIENTATION_EARLY_STOP_SCORE", "0.90"))

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolo11n.pt")
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.2"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.45"))
YOLO_MAX_DETECTIONS = int(os.getenv("YOLO_MAX_DETECTIONS", "24"))
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cpu")

MAX_ROTATIONS = max(1, min(4, int(os.getenv("OCR_ROTATION_STEPS", "4"))))
MAX_NUMBER_CROPS = max(1, min(4, int(os.getenv("OCR_MAX_NUMBER_CROPS", "4"))))
MAX_NAME_CROPS = max(1, min(3, int(os.getenv("OCR_MAX_NAME_CROPS", "3"))))
MAX_SYMBOL_CROPS = max(1, min(3, int(os.getenv("OCR_MAX_SYMBOL_CROPS", "3"))))
MAX_DB_RESULTS = max(5, int(os.getenv("IDENTIFY_DB_RESULT_LIMIT", "20")))
MIN_ACCEPTED_CONFIDENCE = float(os.getenv("IDENTIFY_MIN_CONFIDENCE", "0.50"))

DEFAULT_TEMPLATE_ROOT = Path(__file__).resolve().parents[2] / "templates" / "set_symbols"
DEFAULT_METADATA_PATH = DEFAULT_TEMPLATE_ROOT / "metadata.json"
SET_SYMBOL_TEMPLATE_DIR = Path(os.getenv("SET_SYMBOL_TEMPLATE_DIR", str(DEFAULT_TEMPLATE_ROOT)))
METADATA_PATH = Path(os.getenv("SET_SYMBOL_METADATA", str(DEFAULT_METADATA_PATH)))
SET_SYMBOL_MIN_SCORE = float(os.getenv("SET_SYMBOL_MIN_SCORE", "0.45"))

CARD_RATIO_WIDTH_MM = 63.0
CARD_RATIO_HEIGHT_MM = 88.0
CARD_TARGET_WIDTH = int(os.getenv("CARD_TARGET_WIDTH", "630"))
CARD_TARGET_HEIGHT = int(round(CARD_TARGET_WIDTH * (CARD_RATIO_HEIGHT_MM / CARD_RATIO_WIDTH_MM)))
MAX_DETECTED_CARDS = int(os.getenv("MAX_DETECTED_CARDS", "24"))
_DEBUG_SAVE_TRANSFORMS = os.getenv("DEBUG_SAVE_TRANSFORMS", "0").strip().lower() in {"1", "true", "yes", "on"}
_DEBUG_ROOT = Path(os.getenv("DEBUG_IMAGE_DIR", str(Path(__file__).resolve().parents[2] / "debug_outputs")))
_DEBUG_SESSION_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("debug_session_id", default=None)