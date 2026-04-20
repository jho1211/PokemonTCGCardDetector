from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
import re
from collections.abc import Iterable, Mapping
from typing import Any

import cv2
import numpy as np

from app.services.preprocess import prepare_for_ocr


SLASH_NUMBER_PATTERN = re.compile(r"[A-Z0-9]{1,8}/[A-Z0-9]{1,8}")
PREFIX_NUMBER_PATTERN = re.compile(r"[A-Z]{1,4}\s*\d{1,4}")
ANY_TEXT_PATTERN = re.compile(r"[A-Za-z0-9]{2,}")
MIN_TEXT_CONFIDENCE = 0.18

AMBIGUOUS_DIGIT_MAP = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "|": "1",
    "S": "5",
    "B": "8",
    "Z": "2",
}

NAME_BLOCKLIST = {
    "BASIC",
    "STAGE",
    "TRAINER",
    "ENERGY",
    "POKEMON",
    "HP",
}


@dataclass
class OCRFieldResult:
    text: str | None
    raw_text: str | None
    confidence: float


class OCRService:
    def __init__(self) -> None:
        self._reader = None
        self._reader_init_failed = False
        self._last_init_error: str | None = None
        self._last_runtime_error: str | None = None
        self._last_parse_hint: str | None = None

    @property
    def is_available(self) -> bool:
        return self._load_reader() is not None

    @property
    def last_init_error(self) -> str | None:
        return self._last_init_error

    @property
    def last_runtime_error(self) -> str | None:
        return self._last_runtime_error

    @property
    def last_parse_hint(self) -> str | None:
        return self._last_parse_hint

    def _load_reader(self) -> Any | None:
        if self._reader is not None:
            return self._reader
        if self._reader_init_failed:
            return None

        try:
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            from paddleocr import PaddleOCR

            det_model_dir = os.getenv("PADDLEOCR_DET_MODEL_DIR")
            rec_model_dir = os.getenv("PADDLEOCR_REC_MODEL_DIR")
            cls_model_dir = os.getenv("PADDLEOCR_CLS_MODEL_DIR")

            kwargs: dict[str, Any] = {
                "use_angle_cls": True,
                "lang": "en",
            }
            if det_model_dir:
                kwargs["det_model_dir"] = det_model_dir
            if rec_model_dir:
                kwargs["rec_model_dir"] = rec_model_dir
            if cls_model_dir:
                kwargs["cls_model_dir"] = cls_model_dir

            self._reader = PaddleOCR(**kwargs)
            return self._reader
        except Exception as exc:
            self._reader_init_failed = True
            self._last_init_error = str(exc)
            return None

    def _read_lines(self, image: np.ndarray) -> list[tuple[str, float]]:
        reader = self._load_reader()
        if reader is None:
            return []

        self._last_runtime_error = None
        self._last_parse_hint = None

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            output = reader.ocr(rgb, cls=True)
        except TypeError:
            output = reader.ocr(rgb)
        except Exception as exc:
            self._last_runtime_error = str(exc)
            return []

        lines = self._extract_lines_from_output(output)
        if lines:
            return lines

        try:
            fallback_output = reader.ocr(image)
        except Exception:
            fallback_output = None

        lines = self._extract_lines_from_output(fallback_output)
        if not lines and fallback_output is not None:
            preview = None
            if isinstance(fallback_output, list) and fallback_output:
                preview = type(fallback_output[0]).__name__
            self._last_parse_hint = (
                f"unparsed_ocr_output_type={type(fallback_output).__name__}"
                + (f", first_item_type={preview}" if preview else "")
            )

        return lines

    def _extract_lines_from_output(self, output: Any) -> list[tuple[str, float]]:
        lines: list[tuple[str, float]] = []

        def append_line(text: Any, confidence: Any) -> None:
            value = str(text).strip() if text is not None else ""
            if not value:
                return
            if not ANY_TEXT_PATTERN.search(value):
                return
            try:
                conf_value = float(confidence)
            except (TypeError, ValueError):
                conf_value = 0.0
            if conf_value < MIN_TEXT_CONFIDENCE:
                return
            lines.append((value, conf_value))

        def is_mapping_like(node: Any) -> bool:
            return isinstance(node, Mapping) or (hasattr(node, "keys") and hasattr(node, "get"))

        def iter_mapping_items(node: Any):
            if isinstance(node, Mapping):
                return node.items()
            if hasattr(node, "keys") and hasattr(node, "get"):
                return ((key, node.get(key)) for key in node.keys())
            return ()

        def walk(node: Any) -> None:
            if node is None:
                return

            if is_mapping_like(node):
                rec_text = node.get("rec_text")
                if rec_text is not None:
                    append_line(rec_text, node.get("rec_score", 0.0))

                text_value = node.get("text")
                score_value = node.get("score")
                if text_value is not None and score_value is not None:
                    append_line(text_value, score_value)

                rec_texts = node.get("rec_texts")
                if isinstance(rec_texts, list):
                    scores = node.get("rec_scores") or []
                    for idx, text in enumerate(rec_texts):
                        score = scores[idx] if idx < len(scores) else 0.0
                        append_line(text, score)

                for _, value in iter_mapping_items(node):
                    if isinstance(value, (list, tuple)) or is_mapping_like(value):
                        walk(value)
                return

            if isinstance(node, (list, tuple)):
                if len(node) >= 2 and isinstance(node[1], (list, tuple)) and len(node[1]) >= 2:
                    maybe_text, maybe_score = node[1][0], node[1][1]
                    if isinstance(maybe_text, str):
                        append_line(maybe_text, maybe_score)

                if len(node) == 2 and isinstance(node[0], str):
                    append_line(node[0], node[1])

                for child in node:
                    if isinstance(child, (list, tuple)) or is_mapping_like(child):
                        walk(child)

        walk(output)
        if not lines:
            return []

        by_text: dict[str, float] = {}
        for text, conf in lines:
            key = text.strip()
            prev = by_text.get(key)
            if prev is None or conf > prev:
                by_text[key] = conf

        return sorted(by_text.items(), key=lambda item: item[1], reverse=True)

    @staticmethod
    def _to_bgr(image: Any) -> np.ndarray | None:
        if image is None:
            return None

        arr = np.asarray(image)
        if arr.size == 0:
            return None

        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.ndim == 3 and arr.shape[2] == 3:
            return arr.copy()
        return None

    def _build_variants(self, image: Any, field: str) -> list[np.ndarray]:
        base = self._to_bgr(image)
        if base is None:
            return []

        variants: list[np.ndarray] = [base]

        h, w = base.shape[:2]
        if min(h, w) < 220:
            enlarged = cv2.resize(base, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            variants.append(enlarged)

        variants.append(prepare_for_ocr(base, invert=False, save_debug=False))
        variants.append(prepare_for_ocr(base, invert=True, save_debug=False))

        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))

        if field == "name":
            sharpened = cv2.GaussianBlur(base, (0, 0), 1.1)
            sharpened = cv2.addWeighted(base, 1.5, sharpened, -0.5, 0)
            variants.append(sharpened)

        return variants

    def _read_lines_from_variants(self, image: Any, field: str) -> list[tuple[str, float]]:
        variants = self._build_variants(image, field=field)
        if not variants:
            return []

        lines_by_text: dict[str, float] = {}
        for variant in variants:
            for text, conf in self._read_lines(variant):
                key = text.strip()
                prev = lines_by_text.get(key)
                if prev is None or conf > prev:
                    lines_by_text[key] = conf

        if not lines_by_text:
            return []

        return sorted(lines_by_text.items(), key=lambda item: item[1], reverse=True)

    @staticmethod
    def _normalize_numeric_tail(value: str) -> str:
        normalized: list[str] = []
        for ch in value.upper():
            if ch.isdigit():
                normalized.append(ch)
            else:
                mapped = AMBIGUOUS_DIGIT_MAP.get(ch)
                if mapped:
                    normalized.append(mapped)
        return "".join(normalized)

    @classmethod
    def _normalize_collector_side(cls, value: str) -> str | None:
        cleaned = re.sub(r"[^A-Z0-9|]", "", value.upper())
        if not cleaned:
            return None

        digit_like = set("0123456789OQDIL|SBZ")
        first_digit_like = next((idx for idx, ch in enumerate(cleaned) if ch in digit_like), None)
        if first_digit_like is None:
            return None

        prefix = re.sub(r"[^A-Z]", "", cleaned[:first_digit_like])[:4]
        numeric_source = cleaned[first_digit_like:]

        # OCR often appends a symbol-like letter at the end (for example "147/165O").
        if not prefix and len(numeric_source) >= 3 and numeric_source[-1] in {"O", "Q", "D"}:
            body = numeric_source[:-1]
            if sum(ch.isdigit() for ch in body) >= 2:
                numeric_source = body

        numeric_tail = cls._normalize_numeric_tail(numeric_source)[:4]
        if not prefix and len(numeric_tail) > 3:
            numeric_tail = numeric_tail[:3]

        if not numeric_tail:
            return None

        return f"{prefix}{numeric_tail}"

    @classmethod
    def _normalize_collector_token(cls, token: str) -> str | None:
        compact = re.sub(r"\s+", "", token.upper())
        compact = compact.replace("\\", "/")
        compact = compact.replace(":", "/")

        if "/" in compact:
            left_raw, right_raw = compact.split("/", 1)
            left = cls._normalize_collector_side(left_raw)
            right = cls._normalize_collector_side(right_raw)
            if not left or not right:
                return None
            return f"{left}/{right}"

        candidate = cls._normalize_collector_side(compact)
        if candidate is None:
            return None
        if not re.search(r"\d", candidate):
            return None
        return candidate

    @classmethod
    def _collector_candidates_from_text(cls, text: str) -> list[str]:
        sanitized = re.sub(r"[^A-Z0-9/| ]", " ", text.upper())
        collapsed = re.sub(r"\s+", " ", sanitized).strip()
        if not collapsed:
            return []

        dense = collapsed.replace(" ", "")
        tokens = collapsed.split()

        search_forms = [collapsed, dense]
        search_forms.extend(tokens)
        for idx in range(len(tokens) - 1):
            search_forms.append(tokens[idx] + tokens[idx + 1])

        candidates: list[str] = []
        seen: set[str] = set()

        def consider(value: str) -> None:
            normalized = cls._normalize_collector_token(value)
            if normalized is None:
                return
            if normalized in seen:
                return
            seen.add(normalized)
            candidates.append(normalized)

        for form in search_forms:
            for match in SLASH_NUMBER_PATTERN.findall(form):
                consider(match)
            for match in PREFIX_NUMBER_PATTERN.findall(form):
                consider(match)

        return candidates

    @staticmethod
    def _collector_candidate_bonus(value: str) -> float:
        bonus = 0.0
        if "/" in value:
            bonus += 0.20
        if re.fullmatch(r"\d{1,4}/\d{1,4}", value):
            bonus += 0.08
        if re.fullmatch(r"[A-Z]{1,4}\d{1,4}", value):
            bonus += 0.06
        if len(value) <= 9:
            bonus += 0.04
        return bonus

    @staticmethod
    def _name_clean(text: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9'\- ]+", " ", text).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    @staticmethod
    def _name_quality(cleaned: str) -> float:
        if not cleaned:
            return 0.0
        alpha = sum(ch.isalpha() for ch in cleaned)
        digits = sum(ch.isdigit() for ch in cleaned)
        alpha_ratio = alpha / max(len(cleaned), 1)

        quality = alpha_ratio
        if digits > 0:
            quality -= min(0.4, digits / max(len(cleaned), 1))
        return max(0.0, quality)

    def extract_collector_number(self, image: Any) -> OCRFieldResult:
        lines = self._read_lines_from_variants(image, field="number")
        if not lines:
            return OCRFieldResult(text=None, raw_text=None, confidence=0.0)

        best_text: str | None = None
        best_raw: str | None = None
        best_score = 0.0

        for raw_text, conf in lines:
            candidates = self._collector_candidates_from_text(raw_text)
            for candidate in candidates:
                score = float(np.clip(conf + self._collector_candidate_bonus(candidate), 0.0, 1.0))
                if score > best_score:
                    best_score = score
                    best_text = candidate
                    best_raw = raw_text

        if best_text is not None:
            return OCRFieldResult(text=best_text, raw_text=best_raw, confidence=best_score)

        fallback_text, fallback_conf = lines[0]
        if float(fallback_conf) < MIN_TEXT_CONFIDENCE:
            return OCRFieldResult(text=None, raw_text=None, confidence=0.0)
        return OCRFieldResult(text=None, raw_text=fallback_text, confidence=float(fallback_conf))

    def extract_best_collector_number(self, crops: Iterable[Any]) -> OCRFieldResult:
        best = OCRFieldResult(text=None, raw_text=None, confidence=0.0)
        best_score = 0.0

        for crop in crops:
            result = self.extract_collector_number(crop)
            score = result.confidence + (0.2 if result.text and "/" in result.text else 0.0)
            if result.text is None:
                score *= 0.5
            if score > best_score:
                best_score = score
                best = result

        return best

    def extract_name(self, image: Any) -> OCRFieldResult:
        lines = self._read_lines_from_variants(image, field="name")
        if not lines:
            return OCRFieldResult(text=None, raw_text=None, confidence=0.0)

        best_text: str | None = None
        best_raw: str | None = None
        best_score = 0.0

        for raw_text, conf in lines:
            cleaned = self._name_clean(raw_text)
            if len(cleaned) < 2:
                continue

            tokens_upper = {token.upper() for token in cleaned.split()}
            blocked_count = len(tokens_upper.intersection(NAME_BLOCKLIST))
            quality = self._name_quality(cleaned)

            score = float(conf)
            score += quality * 0.25
            score -= blocked_count * 0.12
            if len(cleaned) <= 20:
                score += 0.05

            if score > best_score:
                best_score = score
                best_text = cleaned
                best_raw = raw_text

        if best_text is None:
            fallback_text, fallback_conf = lines[0]
            if float(fallback_conf) < MIN_TEXT_CONFIDENCE:
                return OCRFieldResult(text=None, raw_text=None, confidence=0.0)
            return OCRFieldResult(text=None, raw_text=fallback_text, confidence=float(fallback_conf))

        return OCRFieldResult(text=best_text, raw_text=best_raw, confidence=float(np.clip(best_score, 0.0, 1.0)))

    def extract_best_name(self, crops: Iterable[Any]) -> OCRFieldResult:
        best = OCRFieldResult(text=None, raw_text=None, confidence=0.0)
        best_score = 0.0

        for crop in crops:
            result = self.extract_name(crop)
            score = result.confidence + (0.1 if result.text else 0.0)
            if score > best_score:
                best_score = score
                best = result

        return best


@lru_cache(maxsize=1)
def get_ocr_service() -> OCRService:
    return OCRService()
