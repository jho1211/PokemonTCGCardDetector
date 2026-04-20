from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE_ROOT = Path(__file__).resolve().parents[2] / "templates" / "set_symbols"
DEFAULT_METADATA_PATH = DEFAULT_TEMPLATE_ROOT / "metadata.json"


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


class SymbolTemplateMatcher:
    def __init__(self) -> None:
        self.template_root = Path(os.getenv("SET_SYMBOL_TEMPLATE_DIR", str(DEFAULT_TEMPLATE_ROOT)))
        self.metadata_path = Path(os.getenv("SET_SYMBOL_METADATA", str(DEFAULT_METADATA_PATH)))
        self.min_match_score = float(os.getenv("SET_SYMBOL_MIN_SCORE", "0.45"))

        self.templates: list[_TemplateEntry] = []
        self.is_available = False
        self.last_error: str | None = None

        self._load_templates()

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

        entries: list[dict[str, str]] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    entries.append({
                        "set_id": str(item.get("set_id", "")).strip(),
                        "set_name": str(item.get("set_name", "")).strip(),
                        "template_file": str(item.get("template_file", "")).strip(),
                    })
        elif isinstance(raw, dict):
            for set_id, template_file in raw.items():
                entries.append(
                    {
                        "set_id": str(set_id).strip(),
                        "set_name": str(set_id).strip(),
                        "template_file": str(template_file).strip(),
                    }
                )

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

    def match(self, symbol_crops: list[np.ndarray]) -> SymbolMatchResult:
        if not self.is_available:
            return SymbolMatchResult(set_id=None, set_name=None, score=0.0)

        best = SymbolMatchResult(set_id=None, set_name=None, score=0.0)

        for crop in symbol_crops:
            if crop.size == 0:
                continue
            roi = _normalize_roi(crop)

            for template in self.templates:
                score = _best_template_score(roi, template.image)
                if score > best.score:
                    best = SymbolMatchResult(set_id=template.set_id, set_name=template.set_name, score=score)

        if best.score < self.min_match_score:
            return SymbolMatchResult(set_id=None, set_name=None, score=best.score)

        return best


@lru_cache(maxsize=1)
def get_symbol_matcher() -> SymbolTemplateMatcher:
    return SymbolTemplateMatcher()


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
