from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from app.models.schemas import CardDebugInfo, CardPayload, CardResult, IdentifyCardResponse, NoMatchInfo
from app.services.ocr import OCRFieldResult, get_ocr_service
from app.services.preprocess import (
    begin_debug_image_session,
    decode_image,
    detect_and_warp_cards,
    end_debug_image_session,
    extract_regions,
    rotate_image_90,
)
from app.services.symbol_matcher import SymbolMatchResult, get_symbol_matcher
from app.services.tcgdex import map_to_frontend_fields, search_cards

from app.config.config import MAX_DB_RESULTS, MAX_NAME_CROPS, MAX_NUMBER_CROPS, MAX_ROTATIONS, MAX_SYMBOL_CROPS, MIN_ACCEPTED_CONFIDENCE

@dataclass
class _CardAnalysis:
    orientation_turns: int
    orientation_score: float
    collector_number: OCRFieldResult
    card_name: OCRFieldResult
    symbol_match: SymbolMatchResult


async def identify_card_from_image_bytes(image_bytes: bytes) -> IdentifyCardResponse:
    if not image_bytes:
        raise ValueError("Uploaded image is empty")

    image = decode_image(image_bytes)
    session = begin_debug_image_session("identify")

    try:
        detections = detect_and_warp_cards(image)

        if not detections:
            return IdentifyCardResponse(
                success=False,
                cards=[],
                confidence=0.0,
                confidence_label=_confidence_label(0.0),
                warning="No card-like object detected.",
            )

        ocr = get_ocr_service()
        symbol_matcher = get_symbol_matcher()

        card_results: list[CardResult] = []

        for detection_index, detected_card in enumerate(detections):
            analysis = _analyze_single_card(
                card_image=detected_card.image,
                ocr_available=ocr.is_available,
                symbol_available=symbol_matcher.is_available,
            )

            query_debug: str | None = None
            mapped_card: dict[str, object] | None = None

            if analysis.collector_number.text or analysis.card_name.text:
                candidates, query_debug = await search_cards(
                    collector_number=analysis.collector_number.text,
                    card_name=analysis.card_name.text,
                    set_id=analysis.symbol_match.set_id,
                    collection_name=analysis.symbol_match.set_name,
                    limit=MAX_DB_RESULTS,
                )

                if candidates:
                    mapped_card = map_to_frontend_fields(candidates[0])

            confidence = _final_confidence(
                preprocess_score=detected_card.score,
                collector_conf=analysis.collector_number.confidence,
                name_conf=analysis.card_name.confidence,
                symbol_conf=analysis.symbol_match.score,
            )

            success, warning, no_match = _resolve_match_status(
                mapped_card=mapped_card,
                confidence=confidence,
                ocr_available=ocr.is_available,
                collector_number=analysis.collector_number,
                card_name=analysis.card_name,
            )

            debug = CardDebugInfo(
                detection_confidence=detected_card.detection_confidence,
                preprocess_score=detected_card.score,
                orientation_degrees=analysis.orientation_turns * 90,
                ocr_number=analysis.collector_number.text,
                ocr_number_raw=analysis.collector_number.raw_text,
                ocr_number_confidence=analysis.collector_number.confidence,
                ocr_name=analysis.card_name.text,
                ocr_name_raw=analysis.card_name.raw_text,
                ocr_name_confidence=analysis.card_name.confidence,
                set_match=analysis.symbol_match.set_id,
                set_match_score=analysis.symbol_match.score,
                query_debug=query_debug,
            )

            card_results.append(
                CardResult(
                    detection_index=detection_index,
                    success=success,
                    confidence=confidence,
                    confidence_label=_confidence_label(confidence),
                    card=CardPayload(**mapped_card) if success and mapped_card is not None else None,
                    warning=warning,
                    no_match=no_match,
                    debug=debug,
                )
            )

        card_results.sort(key=lambda item: item.confidence, reverse=True)

        top_confidence = card_results[0].confidence if card_results else 0.0
        any_success = any(item.success for item in card_results)

        warning: str | None = None
        if not any_success:
            warning = "No confident card match found."

        return IdentifyCardResponse(
            success=any_success,
            cards=card_results,
            confidence=top_confidence,
            confidence_label=_confidence_label(top_confidence),
            warning=warning,
        )
    finally:
        end_debug_image_session(session)


def _analyze_single_card(card_image: np.ndarray, ocr_available: bool, symbol_available: bool) -> _CardAnalysis:
    ocr = get_ocr_service()
    symbol_matcher = get_symbol_matcher()

    best = _CardAnalysis(
        orientation_turns=0,
        orientation_score=-1.0,
        collector_number=OCRFieldResult(text=None, raw_text=None, confidence=0.0),
        card_name=OCRFieldResult(text=None, raw_text=None, confidence=0.0),
        symbol_match=SymbolMatchResult(set_id=None, set_name=None, score=0.0),
    )

    for turns in range(MAX_ROTATIONS):
        oriented = rotate_image_90(card_image, turns)
        regions = extract_regions(oriented)

        if ocr_available:
            number_result = ocr.extract_best_collector_number(regions["number"][:MAX_NUMBER_CROPS])
            name_result = ocr.extract_best_name(regions["name"][:MAX_NAME_CROPS])
        else:
            number_result = OCRFieldResult(text=None, raw_text=None, confidence=0.0)
            name_result = OCRFieldResult(text=None, raw_text=None, confidence=0.0)

        if symbol_available:
            symbol_result = symbol_matcher.match(regions["symbol"][:MAX_SYMBOL_CROPS])
        else:
            symbol_result = SymbolMatchResult(set_id=None, set_name=None, score=0.0)

        score = _orientation_score(number_result, name_result, symbol_result)
        if score > best.orientation_score:
            best = _CardAnalysis(
                orientation_turns=turns,
                orientation_score=score,
                collector_number=number_result,
                card_name=name_result,
                symbol_match=symbol_result,
            )

    return best


def _orientation_score(number: OCRFieldResult, name: OCRFieldResult, symbol: SymbolMatchResult) -> float:
    collector_signal = number.confidence
    if number.text and "/" in number.text:
        collector_signal = min(1.0, collector_signal + 0.12)

    name_signal = name.confidence
    symbol_signal = symbol.score

    return float((0.65 * collector_signal) + (0.20 * name_signal) + (0.15 * symbol_signal))


def _final_confidence(
    preprocess_score: float,
    collector_conf: float,
    name_conf: float,
    symbol_conf: float,
) -> float:
    weighted = (0.30 * preprocess_score) + (0.40 * collector_conf) + (0.15 * name_conf) + (0.15 * symbol_conf)
    return float(max(0.0, min(1.0, weighted)))


def _resolve_match_status(
    mapped_card: dict[str, object] | None,
    confidence: float,
    ocr_available: bool,
    collector_number: OCRFieldResult,
    card_name: OCRFieldResult,
) -> tuple[bool, str | None, NoMatchInfo | None]:
    if not ocr_available:
        return (
            False,
            "OCR service is currently unavailable.",
            NoMatchInfo(
                reason="ocr_unavailable",
                suggestions=[
                    "Ensure PaddleOCR models are accessible on this host.",
                    "Retry once OCR service reports as healthy.",
                ],
            ),
        )

    if not collector_number.text and not card_name.text:
        return (
            False,
            "Could not read collector number or card name.",
            NoMatchInfo(
                reason="ocr_fields_missing",
                suggestions=[
                    "Capture in brighter lighting.",
                    "Fill more of the frame with the card.",
                    "Avoid glare on the bottom and top text bars.",
                ],
            ),
        )

    if mapped_card is None:
        return (
            False,
            "No database match was found for extracted fields.",
            NoMatchInfo(
                reason="database_no_match",
                suggestions=[
                    "Retry with a sharper, front-facing image.",
                    "Ensure the card is from supported modern English sets.",
                ],
            ),
        )

    if confidence < MIN_ACCEPTED_CONFIDENCE:
        return (
            False,
            "Match candidate rejected because confidence is below threshold.",
            NoMatchInfo(
                reason="low_confidence",
                suggestions=[
                    "Retake image with less perspective skew.",
                    "Avoid shadowing over collector number and name.",
                ],
            ),
        )

    return True, None, None


def _confidence_label(score: float) -> str:
    if score >= 0.80:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"
