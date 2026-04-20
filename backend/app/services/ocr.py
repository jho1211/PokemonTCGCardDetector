from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import cv2
import numpy as np

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
        self.lang = os.getenv("PADDLEOCR_LANG", "en")
        self.use_angle_cls = os.getenv("PADDLEOCR_USE_ANGLE_CLS", "1").strip().lower() in {"1", "true", "yes", "on"}
        self.use_gpu = os.getenv("PADDLEOCR_USE_GPU", "0").strip().lower() in {"1", "true", "yes", "on"}

        self._reader: Any | None = None
        self.is_available = False
        self.last_init_error: str | None = None
        self.last_runtime_error: str | None = None

        self._initialize()

    def _initialize(self) -> None:
        if PaddleOCR is None:
            self.last_init_error = "paddleocr is not installed"
            logger.warning("PaddleOCR unavailable: %s", self.last_init_error)
            return

        try:
            self._reader = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                show_log=False,
                use_gpu=self.use_gpu,
            )
            self.is_available = True
            logger.info("PaddleOCR initialized (lang=%s, use_gpu=%s)", self.lang, self.use_gpu)
        except Exception as exc:  # pragma: no cover - runtime environment issue
            self.last_init_error = str(exc)
            self.is_available = False
            logger.warning("Failed to initialize PaddleOCR: %s", exc)

    def extract_best_collector_number(self, crops: Sequence[np.ndarray]) -> OCRFieldResult:
        if not self.is_available:
            return OCRFieldResult(text=None, raw_text=None, confidence=0.0)

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

        return best

    def extract_best_name(self, crops: Sequence[np.ndarray]) -> OCRFieldResult:
        if not self.is_available:
            return OCRFieldResult(text=None, raw_text=None, confidence=0.0)

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

        return best

    def _ocr_lines(self, image: np.ndarray) -> list[tuple[str, float]]:
        if self._reader is None:
            return []

        try:
            result = self._reader.ocr(image, cls=self.use_angle_cls)
        except Exception as exc:  # pragma: no cover - runtime inference issue
            self.last_runtime_error = str(exc)
            logger.debug("PaddleOCR inference failed: %s", exc)
            return []

        lines: list[tuple[str, float]] = []
        if not isinstance(result, list):
            return lines

        for block in result:
            if not isinstance(block, list):
                continue
            for item in block:
                parsed = _parse_ocr_item(item)
                if parsed is None:
                    continue
                text, confidence = parsed
                lines.append((text, confidence))

        return lines


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

    return float(max(0.0, min(1.0, score)))
