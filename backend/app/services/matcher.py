from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any


@dataclass
class CandidateScore:
    payload: Any
    confidence: float
    matched_by: str


def _normalize(value: str | None) -> str:
    if value is None:
        return ""
    return "".join(ch for ch in value.upper().strip() if ch.isalnum() or ch == "/")


def _similarity(a: str | None, b: str | None) -> float:
    left = _normalize(a)
    right = _normalize(b)
    if not left or not right:
        return 0.0
    return float(SequenceMatcher(a=left, b=right).ratio())


def _get_field(payload: Any, key: str, default: Any = None) -> Any:
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


def _extract_local_id(card: Any) -> str | None:
    local_id = _get_field(card, "localId")
    if local_id:
        return str(local_id)

    card_id = str(_get_field(card, "id", ""))
    if "-" in card_id:
        return card_id.split("-", 1)[1]
    return None


def score_candidates(
    candidates: list[Any],
    collector_number: str | None,
    card_name: str | None,
    ocr_number_confidence: float,
    ocr_name_confidence: float,
    preprocess_score: float,
) -> CandidateScore | None:
    if not candidates:
        return None

    best: CandidateScore | None = None

    for candidate in candidates:
        candidate_number = _extract_local_id(candidate)
        candidate_name = str(_get_field(candidate, "name", ""))

        number_similarity = _similarity(collector_number, candidate_number)
        name_similarity = _similarity(card_name, candidate_name)

        number_signal = number_similarity * max(0.2, ocr_number_confidence)
        name_signal = name_similarity * max(0.2, ocr_name_confidence)
        preprocess_signal = max(0.0, min(1.0, preprocess_score))

        # Heavier weight on collector number because it is usually the strongest key.
        confidence = (0.55 * number_signal) + (0.3 * name_signal) + (0.15 * preprocess_signal)

        matched_by = "name"
        if number_similarity >= max(0.8, name_similarity):
            matched_by = "collector_number"

        scored = CandidateScore(payload=candidate, confidence=float(confidence), matched_by=matched_by)
        if best is None or scored.confidence > best.confidence:
            best = scored

    return best


def to_confidence_band(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.65:
        return "medium"
    if score >= 0.4:
        return "low"
    return "none"
