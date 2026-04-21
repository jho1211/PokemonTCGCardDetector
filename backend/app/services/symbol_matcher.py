from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config.config import (
    METADATA_PATH,
    SET_SYMBOL_MIN_SCORE,
    SET_SYMBOL_OCR_MAP_PATH,
    SET_SYMBOL_OCR_MIN_SCORE,
    SET_SYMBOL_TEMPLATE_DIR,
)

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE_ROOT = Path(__file__).resolve().parents[2] / "templates" / "set_symbols"
DEFAULT_METADATA_PATH = DEFAULT_TEMPLATE_ROOT / "metadata.json"
DEFAULT_ABBREVIATION_MAP: dict[str, str] = {
    "MEW": "sv03.5",
}


@dataclass
class _TemplateEntry:
    set_id: str
    set_name: str
    image: np.ndarray


@dataclass
class SymbolMatchResult:
    set_id: str | None
    set_name: str | None
    score: float
    method: str = "none"
    token: str | None = None


class SymbolTemplateMatcher:
    def __init__(self) -> None:
        self.template_root = SET_SYMBOL_TEMPLATE_DIR
        self.metadata_path = METADATA_PATH
        self.min_match_score = SET_SYMBOL_MIN_SCORE
        self.ocr_map_path = SET_SYMBOL_OCR_MAP_PATH
        self.min_ocr_match_score = SET_SYMBOL_OCR_MIN_SCORE

        self.templates: list[_TemplateEntry] = []
        self.set_name_by_id: dict[str, str] = {}
        self.ocr_abbreviation_map: dict[str, str] = {}
        self.is_available = False
        self.last_error: str | None = None

        self._load_templates()
        self._load_ocr_abbreviation_map()

    @property
    def has_ocr_fallback(self) -> bool:
        return bool(self.ocr_abbreviation_map)

    def _load_templates(self) -> None:
        if not self.metadata_path.exists():
            self.last_error = f"Metadata file not found: {self.metadata_path}"
            logger.info("Set symbol matcher disabled: %s", self.last_error)
            return

        try:
            raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.last_error = str(exc)
            logger.warning("Failed to read set symbol metadata: %s", exc)
            return

        entries = _parse_metadata_entries(raw)
        self.set_name_by_id = {
            entry["set_id"]: entry.get("set_name", entry["set_id"]) or entry["set_id"]
            for entry in entries
            if entry.get("set_id")
        }

        loaded: list[_TemplateEntry] = []
        for entry in entries:
            set_id = entry.get("set_id", "")
            template_name = entry.get("template_file", "")
            set_name = entry.get("set_name", set_id)

            if not set_id or not template_name:
                continue

            path = self.template_root / template_name
            if not path.exists():
                continue

            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.size == 0:
                continue

            processed = _normalize_template(image)
            loaded.append(_TemplateEntry(set_id=set_id, set_name=set_name, image=processed))

        self.templates = loaded
        self.is_available = bool(loaded)

        if self.is_available:
            logger.info("Loaded %d set symbol templates from %s", len(loaded), self.template_root)
        else:
            self.last_error = "No templates loaded"
            logger.info("Set symbol matcher disabled: %s", self.last_error)

    def _load_ocr_abbreviation_map(self) -> None:
        merged: dict[str, str] = dict(DEFAULT_ABBREVIATION_MAP)

        if self.ocr_map_path.exists():
            try:
                raw = json.loads(self.ocr_map_path.read_text(encoding="utf-8"))
                merged.update(_parse_ocr_abbreviation_map(raw))
            except Exception as exc:
                logger.warning("Failed to read OCR abbreviation map %s: %s", self.ocr_map_path, exc)
        else:
            logger.info("Set abbreviation map not found at %s, using built-in defaults only", self.ocr_map_path)

        normalized: dict[str, str] = {}
        for token, set_id in merged.items():
            normalized_token = _normalize_abbreviation_token(token)
            normalized_set_id = _normalize_set_id(set_id)
            if not normalized_token or not normalized_set_id:
                continue
            normalized[normalized_token] = normalized_set_id

        self.ocr_abbreviation_map = normalized

        if self.ocr_abbreviation_map:
            logger.info("Loaded %d OCR set-abbreviation mappings", len(self.ocr_abbreviation_map))
        else:
            logger.info("No OCR set-abbreviation mappings loaded")

    def match(self, symbol_crops: list[np.ndarray]) -> SymbolMatchResult:
        if not self.is_available:
            return SymbolMatchResult(set_id=None, set_name=None, score=0.0, method="none")

        best = SymbolMatchResult(set_id=None, set_name=None, score=0.0, method="none")

        for crop in symbol_crops:
            if crop.size == 0:
                continue
            roi = _normalize_roi(crop)

            for template in self.templates:
                score = _best_template_score(roi, template.image)
                if score > best.score:
                    best = SymbolMatchResult(
                        set_id=template.set_id,
                        set_name=template.set_name,
                        score=score,
                        method="template",
                    )

        if best.score < self.min_match_score:
            return SymbolMatchResult(set_id=None, set_name=None, score=best.score, method="template_low_conf")

        return best

    def match_ocr_abbreviation(self, abbreviation: str | None, confidence: float) -> SymbolMatchResult:
        score = float(max(0.0, min(1.0, confidence)))
        token = _normalize_abbreviation_token(abbreviation)
        if not token:
            return SymbolMatchResult(set_id=None, set_name=None, score=score, method="ocr_abbreviation")

        if score < self.min_ocr_match_score:
            return SymbolMatchResult(set_id=None, set_name=None, score=score, method="ocr_abbreviation", token=token)

        set_id = self.ocr_abbreviation_map.get(token)
        if not set_id:
            return SymbolMatchResult(set_id=None, set_name=None, score=score, method="ocr_abbreviation", token=token)

        set_name = self.set_name_by_id.get(set_id, set_id)
        return SymbolMatchResult(
            set_id=set_id,
            set_name=set_name,
            score=score,
            method="ocr_abbreviation",
            token=token,
        )


@lru_cache(maxsize=1)
def get_symbol_matcher() -> SymbolTemplateMatcher:
    return SymbolTemplateMatcher()


def _parse_metadata_entries(raw: Any) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                entries.append(
                    {
                        "set_id": str(item.get("set_id", "")).strip(),
                        "set_name": str(item.get("set_name", "")).strip(),
                        "template_file": str(item.get("template_file", "")).strip(),
                    }
                )
    elif isinstance(raw, dict):
        for set_id, template_file in raw.items():
            entries.append(
                {
                    "set_id": str(set_id).strip(),
                    "set_name": str(set_id).strip(),
                    "template_file": str(template_file).strip(),
                }
            )
    return entries


def _parse_ocr_abbreviation_map(raw: Any) -> dict[str, str]:
    parsed: dict[str, str] = {}

    if isinstance(raw, dict):
        for token, set_id in raw.items():
            _append_ocr_mapping(parsed, token, set_id)
        return parsed

    if not isinstance(raw, list):
        return parsed

    for item in raw:
        if not isinstance(item, dict):
            continue

        set_id = item.get("set_id")
        aliases: list[Any] = []

        for key in ("abbreviation", "symbol", "alias"):
            value = item.get(key)
            if value is not None:
                aliases.append(value)

        for key in ("abbreviations", "symbols", "aliases"):
            value = item.get(key)
            if isinstance(value, list):
                aliases.extend(value)

        for alias in aliases:
            _append_ocr_mapping(parsed, alias, set_id)

    return parsed


def _append_ocr_mapping(mapping: dict[str, str], token: Any, set_id: Any) -> None:
    normalized_token = _normalize_abbreviation_token(token)
    normalized_set_id = _normalize_set_id(set_id)
    if not normalized_token or not normalized_set_id:
        return
    mapping[normalized_token] = normalized_set_id


def _normalize_abbreviation_token(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip().upper()
    if not text:
        return None

    text = re.sub(r"[^A-Z0-9]", "", text)
    if len(text) < 2 or len(text) > 6:
        return None

    if not re.search(r"[A-Z]", text):
        return None

    return text


def _normalize_set_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None

    if "/" in text:
        text = text.rsplit("/", 1)[-1].strip()

    sv_match = re.fullmatch(r"sv(\d)(\.[a-z0-9]+)?", text)
    if sv_match is not None:
        text = f"sv0{sv_match.group(1)}{sv_match.group(2) or ''}"

    return text


def _normalize_template(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC)
    equalized = cv2.equalizeHist(resized)
    return cv2.Canny(equalized, 50, 140)


def _normalize_roi(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enlarged = cv2.resize(gray, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(enlarged, (3, 3), 0)
    equalized = cv2.equalizeHist(blurred)
    return cv2.Canny(equalized, 40, 120)


def _best_template_score(roi: np.ndarray, template: np.ndarray) -> float:
    roi_h, roi_w = roi.shape[:2]
    if roi_h < 10 or roi_w < 10:
        return 0.0

    best = 0.0
    for scale in (0.60, 0.75, 0.90, 1.0, 1.15, 1.30):
        resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        t_h, t_w = resized_template.shape[:2]
        if t_h >= roi_h or t_w >= roi_w or t_h < 8 or t_w < 8:
            continue

        response = cv2.matchTemplate(roi, resized_template, cv2.TM_CCOEFF_NORMED)
        score = float(np.max(response)) if response.size else 0.0
        if score > best:
            best = score

    return float(max(0.0, min(1.0, best)))
