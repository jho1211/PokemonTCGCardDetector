from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import cv2
import numpy as np

from app.config.config import (
    PADDLEOCR_ENABLE_MKLDNN,
    PADDLEOCR_LANG,
    PADDLEOCR_USE_DOC_ORIENTATION_CLASSIFY,
    PADDLEOCR_USE_DOC_UNWARPING,
    PADDLEOCR_USE_TEXTLINE_ORIENTATION,
)

try:
    from paddleocr import PaddleOCR  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover - dependency/runtime environment issue
    PaddleOCR = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class OCRFieldResult:
    text: str | None
    raw_text: str | None
    confidence: float


class OCRService:
    def __init__(self) -> None:
        self.lang = PADDLEOCR_LANG
        self.enable_mkldnn = PADDLEOCR_ENABLE_MKLDNN
        self.use_doc_orientation_classify = PADDLEOCR_USE_DOC_ORIENTATION_CLASSIFY
        self.use_doc_unwarping = PADDLEOCR_USE_DOC_UNWARPING
        self.use_textline_orientation = PADDLEOCR_USE_TEXTLINE_ORIENTATION
        # Legacy PaddleOCR v2 compatibility flag.
        self.use_angle_cls = self.use_textline_orientation

        self._reader: Any | None = None
        self.is_available = False
        self.last_init_error: str | None = None
        self.last_runtime_error: str | None = None
        self.last_call_backend: str | None = None
        self.last_result_format: str | None = None

        self._initialize()

    def _initialize(self) -> None:
        if PaddleOCR is None:
            self.last_init_error = "paddleocr is not installed"
            logger.warning("PaddleOCR unavailable: %s", self.last_init_error)
            return

        try:
            self._reader = self._build_reader()
            self.is_available = True
            logger.info(
                "PaddleOCR initialized (lang=%s, enable_mkldnn=%s, use_doc_orientation_classify=%s, use_doc_unwarping=%s, use_textline_orientation=%s)",
                self.lang,
                self.enable_mkldnn,
                self.use_doc_orientation_classify,
                self.use_doc_unwarping,
                self.use_textline_orientation,
            )
        except Exception as exc:  # pragma: no cover - runtime environment issue
            self.last_init_error = str(exc)
            self.is_available = False
            logger.warning("Failed to initialize PaddleOCR: %s", exc)

    def _build_reader(self) -> Any:
        try:
            return PaddleOCR(lang=self.lang, enable_mkldnn=self.enable_mkldnn)
        except TypeError as exc:
            # PaddleOCR v2 does not expose the same init kwargs as v3.
            if "enable_mkldnn" not in str(exc):
                raise
            logger.info("PaddleOCR init does not support enable_mkldnn; retrying with lang only")
            return PaddleOCR(lang=self.lang)

    def _reset_runtime_state(self) -> None:
        self.last_runtime_error = None
        self.last_call_backend = None
        self.last_result_format = None

    def extract_best_collector_number(self, crops: Sequence[np.ndarray], stop_score: float | None = None) -> OCRFieldResult:
        if not self.is_available:
            return OCRFieldResult(text=None, raw_text=None, confidence=0.0)

        stop_threshold = _resolve_stop_score(stop_score, fallback=1.0)
        self._reset_runtime_state()
        best = OCRFieldResult(text=None, raw_text=None, confidence=0.0)
        for crop in crops:
            for variant in _collector_variants(crop):
                for raw_text, conf in self._ocr_lines(variant):
                    normalized = _normalize_collector_number(raw_text)
                    if normalized is None:
                        continue

                    score = _collector_score(normalized, conf)
                    if score > best.confidence:
                        best = OCRFieldResult(text=normalized, raw_text=raw_text, confidence=score)
                        if best.confidence >= stop_threshold:
                            return best

        return best

    def extract_best_name(self, crops: Sequence[np.ndarray], stop_score: float | None = None) -> OCRFieldResult:
        if not self.is_available:
            return OCRFieldResult(text=None, raw_text=None, confidence=0.0)

        stop_threshold = _resolve_stop_score(stop_score, fallback=1.0)
        self._reset_runtime_state()
        best = OCRFieldResult(text=None, raw_text=None, confidence=0.0)
        for crop in crops:
            for variant in _name_variants(crop):
                for raw_text, conf in self._ocr_lines(variant):
                    normalized = _normalize_name(raw_text)
                    if normalized is None:
                        continue

                    score = _name_score(normalized, conf)
                    if score > best.confidence:
                        best = OCRFieldResult(text=normalized, raw_text=raw_text, confidence=score)
                        if best.confidence >= stop_threshold and not _is_low_information_name(normalized):
                            return best

        return best

    def _ocr_lines(self, image: np.ndarray) -> list[tuple[str, float]]:
        if self._reader is None:
            return []

        try:
            result = self._invoke_reader(image)
            self.last_result_format = _describe_ocr_result(result)
        except Exception as exc:  # pragma: no cover - runtime inference issue
            self.last_runtime_error = f"{type(exc).__name__}: {exc}"
            logger.debug("PaddleOCR inference failed via %s: %s", self.last_call_backend or "unknown", exc, exc_info=True)
            return []

        return _parse_ocr_output(result)

    def _invoke_reader(self, image: np.ndarray) -> Any:
        predict = getattr(self._reader, "predict", None)
        if callable(predict):
            self.last_call_backend = "predict"
            try:
                return predict(
                    image,
                    use_doc_orientation_classify=self.use_doc_orientation_classify,
                    use_doc_unwarping=self.use_doc_unwarping,
                    use_textline_orientation=self.use_textline_orientation,
                )
            except TypeError as exc:
                if "use_doc_orientation_classify" not in str(exc) and "use_doc_unwarping" not in str(exc):
                    raise
                try:
                    return predict(image, use_textline_orientation=self.use_textline_orientation)
                except TypeError as inner_exc:
                    if "use_textline_orientation" not in str(inner_exc):
                        raise
                    return predict(image)

        ocr_method = getattr(self._reader, "ocr", None)
        if callable(ocr_method):
            self.last_call_backend = "ocr"
            try:
                return ocr_method(image, cls=self.use_angle_cls)
            except TypeError as exc:
                if "cls" not in str(exc):
                    raise
                return ocr_method(image)

        raise RuntimeError("PaddleOCR reader does not expose a callable predict/ocr method")


@lru_cache(maxsize=1)
def get_ocr_service() -> OCRService:
    return OCRService()


def _parse_ocr_item(item: Any) -> tuple[str, float] | None:
    if not isinstance(item, (list, tuple)):
        return None
    if len(item) < 2:
        return None

    prediction = item[1]
    if not isinstance(prediction, (list, tuple)) or len(prediction) < 2:
        return None

    text = str(prediction[0]).strip()
    if not text:
        return None

    try:
        confidence = float(prediction[1])
    except (TypeError, ValueError):
        confidence = 0.0

    return text, confidence


def _parse_v3_result_block(block: Any) -> list[tuple[str, float]]:
    if not isinstance(block, dict):
        return []

    texts = block.get("rec_texts")
    scores = block.get("rec_scores")
    if not isinstance(texts, list):
        return []

    lines: list[tuple[str, float]] = []
    for idx, text_candidate in enumerate(texts):
        text = str(text_candidate).strip()
        if not text:
            continue

        confidence = 0.0
        if isinstance(scores, list) and idx < len(scores):
            try:
                confidence = float(scores[idx])
            except (TypeError, ValueError):
                confidence = 0.0

        lines.append((text, confidence))

    return lines


def _parse_ocr_output(result: Any) -> list[tuple[str, float]]:
    lines: list[tuple[str, float]] = []

    # PaddleOCR v3 returns a list of dicts containing rec_texts/rec_scores.
    if isinstance(result, list):
        for block in result:
            if isinstance(block, dict):
                lines.extend(_parse_v3_result_block(block))
                continue

            if isinstance(block, (list, tuple)):
                for item in block:
                    parsed = _parse_ocr_item(item)
                    if parsed is not None:
                        lines.append(parsed)

    # Some wrappers may return a single dict rather than a list.
    elif isinstance(result, dict):
        lines.extend(_parse_v3_result_block(result))

    return lines


def _describe_ocr_result(result: Any) -> str:
    if isinstance(result, list):
        if not result:
            return "list:empty"
        first = result[0]
        if isinstance(first, dict):
            return "list:dict(v3)"
        if isinstance(first, list):
            return "list:list(v2)"
        return f"list:{type(first).__name__}"

    if isinstance(result, dict):
        return "dict(v3-single)"

    return type(result).__name__


def _resolve_stop_score(value: float | None, fallback: float) -> float:
    if value is None:
        return fallback
    return float(max(0.0, min(1.0, value)))


def _collector_variants(image: np.ndarray) -> list[np.ndarray]:
    if image.size == 0:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enlarged = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(enlarged, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(thresh)

    return [
        cv2.cvtColor(enlarged, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR),
    ]


def _name_variants(image: np.ndarray) -> list[np.ndarray]:
    if image.size == 0:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enlarged = cv2.resize(gray, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.bilateralFilter(enlarged, 5, 50, 50)

    return [
        cv2.cvtColor(enlarged, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR),
    ]


def _normalize_collector_number(value: str) -> str | None:
    raw = value.upper().strip()
    if not raw:
        return None

    substitutions = {
        "O": "0",
        "I": "1",
        "L": "1",
        "|": "1",
        "\\": "/",
    }

    for old, new in substitutions.items():
        raw = raw.replace(old, new)

    raw = raw.replace(" ", "")
    raw = re.sub(r"[^A-Z0-9/]", "", raw)

    if not raw:
        return None

    if "/" in raw:
        left, right = raw.split("/", 1)
        left = re.sub(r"[^A-Z0-9]", "", left)
        right = re.sub(r"[^A-Z0-9]", "", right)
        if not left or not right:
            return None
        return f"{left}/{right}"

    # Allow promo-like IDs (e.g. SVP045) if slash is absent.
    if len(raw) >= 2:
        return raw

    return None


def _normalize_name(value: str) -> str | None:
    text = value.strip()
    if not text:
        return None

    text = re.sub(r"[^A-Za-z0-9'\- ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 2:
        return None

    return text


def _is_low_information_name(value: str) -> bool:
    canonical = re.sub(r"[^A-Z0-9]", "", value.upper())
    if not canonical:
        return True

    generic_tokens = {
        "BASIC",
        "TRAINER",
        "ENERGY",
        "POKEMON",
        "ITEM",
        "SUPPORTER",
        "STADIUM",
        "SPECIALENERGY",
        "STAGE1",
        "STAGE2",
    }
    return canonical in generic_tokens


def _collector_score(normalized: str, confidence: float) -> float:
    score = max(0.0, min(1.0, confidence))

    if "/" in normalized:
        score += 0.24
    if re.search(r"\d", normalized):
        score += 0.08
    if len(normalized) >= 4:
        score += 0.06

    return float(max(0.0, min(1.0, score)))


def _name_score(normalized: str, confidence: float) -> float:
    score = max(0.0, min(1.0, confidence))

    alpha_count = sum(1 for char in normalized if char.isalpha())
    alpha_ratio = alpha_count / max(1, len(normalized))

    score += 0.12 * alpha_ratio
    if len(normalized) >= 4:
        score += 0.06
    if _is_low_information_name(normalized):
        score -= 0.22

    return float(max(0.0, min(1.0, score)))
