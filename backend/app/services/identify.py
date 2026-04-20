from __future__ import annotations

from dataclasses import dataclass
import os
import uuid

import numpy as np

from app.models.schemas import CardResult, DebugResult, IdentifyCardResponse, NoMatchResult
from app.services.matcher import score_candidates, to_confidence_band
from app.services.ocr import OCRFieldResult, OCRService, get_ocr_service
from app.services.preprocess import (
    begin_debug_image_session,
    decode_image,
    detect_and_warp_card,
    end_debug_image_session,
    extract_regions,
    rotate_image_90,
)
from app.services.tcgdex import map_to_frontend_fields, search_cards


LOW_CONFIDENCE_ACCEPTANCE_THRESHOLD = 0.4


@dataclass
class OrientationOCRAttempt:
    turns_clockwise: int
    number: OCRFieldResult
    name: OCRFieldResult
    score: float


def _crop(image: np.ndarray, top_ratio: float, bottom_ratio: float, left_ratio: float, right_ratio: float) -> np.ndarray:
    height, width = image.shape[:2]
    top = max(0, min(height, int(height * top_ratio)))
    bottom = max(top + 1, min(height, int(height * bottom_ratio)))
    left = max(0, min(width, int(width * left_ratio)))
    right = max(left + 1, min(width, int(width * right_ratio)))
    return image[top:bottom, left:right]


def _fallback_regions(image: np.ndarray) -> dict[str, list[np.ndarray]]:
    return {
        "number": [
            _crop(image, 0.82, 1.00, 0.00, 0.65),
            _crop(image, 0.84, 1.00, 0.00, 1.00),
            _crop(image, 0.80, 0.98, 0.04, 0.98),
        ],
        "name": [
            _crop(image, 0.01, 0.20, 0.02, 0.98),
            _crop(image, 0.04, 0.22, 0.04, 0.92),
        ],
    }


def _ocr_result_strength(result: OCRFieldResult, collector_field: bool) -> float:
    if result is None:
        return 0.0

    score = float(result.confidence)
    if result.text:
        score += 0.12
    if collector_field and result.text and "/" in result.text:
        score += 0.18
    if collector_field and result.text is None:
        score *= 0.5
    return score


def _prefer_ocr_result(primary: OCRFieldResult, secondary: OCRFieldResult, collector_field: bool) -> OCRFieldResult:
    primary_score = _ocr_result_strength(primary, collector_field=collector_field)
    secondary_score = _ocr_result_strength(secondary, collector_field=collector_field)
    if secondary_score > primary_score:
        return secondary
    return primary


def _orientation_score(number_result: OCRFieldResult, name_result: OCRFieldResult) -> float:
    number_signal = _ocr_result_strength(number_result, collector_field=True)
    name_signal = _ocr_result_strength(name_result, collector_field=False)
    return (0.75 * number_signal) + (0.25 * name_signal)


def _evaluate_orientations(card_image: np.ndarray, ocr: OCRService) -> OrientationOCRAttempt:
    best_attempt: OrientationOCRAttempt | None = None

    for turns in range(4):
        oriented = rotate_image_90(card_image, turns)
        regions = extract_regions(oriented)

        number_result = ocr.extract_best_collector_number(regions["number"])
        name_result = ocr.extract_best_name(regions["name"])
        score = _orientation_score(number_result, name_result)

        attempt = OrientationOCRAttempt(
            turns_clockwise=turns,
            number=number_result,
            name=name_result,
            score=score,
        )

        if best_attempt is None or attempt.score > best_attempt.score:
            best_attempt = attempt

    if best_attempt is None:
        return OrientationOCRAttempt(
            turns_clockwise=0,
            number=OCRFieldResult(text=None, raw_text=None, confidence=0.0),
            name=OCRFieldResult(text=None, raw_text=None, confidence=0.0),
            score=0.0,
        )

    return best_attempt


async def identify_card_from_image_bytes(image_bytes: bytes) -> IdentifyCardResponse:
    debug_token = begin_debug_image_session(f"scan_{uuid.uuid4().hex[:10]}_{os.getpid()}")

    try:
        image = decode_image(image_bytes)
        preprocessed = detect_and_warp_card(image)

        ocr = get_ocr_service()

        if not ocr.is_available:
            warning = "OCR engine is unavailable. Configure local PaddleOCR model paths or allow model download access."
            if ocr.last_init_error:
                warning = f"{warning} Init error: {ocr.last_init_error}"

            return IdentifyCardResponse(
                success=False,
                confidence=0.0,
                confidence_label="none",
                card=None,
                warning=warning,
                no_match=NoMatchResult(
                    reason="ocr_unavailable",
                    candidate_count=0,
                    suggestions=[
                        "Set PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True before starting the API.",
                        "Provide local model dirs with PADDLEOCR_DET_MODEL_DIR, PADDLEOCR_REC_MODEL_DIR, and PADDLEOCR_CLS_MODEL_DIR.",
                        "If internet is available, run once to allow PaddleOCR model download and cache.",
                    ],
                ),
                debug=DebugResult(
                    set_match=None,
                    ocr_number=None,
                    ocr_number_raw=None,
                    ocr_name=None,
                    preprocess_score=preprocessed.score,
                    contour_confidence=preprocessed.contour_confidence,
                    orientation_degrees=None,
                    ocr_number_confidence=0.0,
                    ocr_name_confidence=0.0,
                    matched_by=None,
                    tcgdex_query=None,
                    price_updated_at=None,
                ),
            )

        orientation_attempt = _evaluate_orientations(preprocessed.image, ocr)
        number_result = orientation_attempt.number
        name_result = orientation_attempt.name

        # Fallback pass on wider windows if the strict card ROIs fail.
        if not number_result.text and not name_result.text:
            fallback = _fallback_regions(preprocessed.image)
            number_fallback = ocr.extract_best_collector_number(fallback["number"])
            name_fallback = ocr.extract_best_name(fallback["name"])

            number_result = _prefer_ocr_result(number_result, number_fallback, collector_field=True)
            name_result = _prefer_ocr_result(name_result, name_fallback, collector_field=False)

        # Final pass on the original frame when contour detection was weak.
        if not number_result.text and not name_result.text and not preprocessed.detected_card:
            fallback = _fallback_regions(image)
            number_fallback = ocr.extract_best_collector_number(fallback["number"])
            name_fallback = ocr.extract_best_name(fallback["name"])

            number_result = _prefer_ocr_result(number_result, number_fallback, collector_field=True)
            name_result = _prefer_ocr_result(name_result, name_fallback, collector_field=False)

        candidates, query_debug = await search_cards(
            collector_number=number_result.text,
            card_name=name_result.text,
        )

        best_match = score_candidates(
            candidates=candidates,
            collector_number=number_result.text,
            card_name=name_result.text,
            ocr_number_confidence=number_result.confidence,
            ocr_name_confidence=name_result.confidence,
            preprocess_score=preprocessed.score,
        )

        confidence = best_match.confidence if best_match is not None else 0.0
        confidence_label = to_confidence_band(confidence)

        debug = DebugResult(
            set_match=None,
            ocr_number=number_result.text,
            ocr_number_raw=number_result.raw_text,
            ocr_name=name_result.text,
            preprocess_score=preprocessed.score,
            contour_confidence=preprocessed.contour_confidence,
            orientation_degrees=int((orientation_attempt.turns_clockwise % 4) * 90),
            ocr_number_confidence=number_result.confidence,
            ocr_name_confidence=name_result.confidence,
            matched_by=best_match.matched_by if best_match else None,
            tcgdex_query=query_debug,
            price_updated_at=None,
        )

        if best_match is None:
            no_match_reason = "no_candidates"
            warning_message = "No card candidates were found from OCR results."
            suggestions = [
                "Retake the photo with brighter, even lighting.",
                "Keep the full card frame visible and in focus.",
                "Try to keep glare away from the collector number area.",
            ]

            if not number_result.text and not name_result.text:
                no_match_reason = "ocr_empty"
                warning_message = "OCR could not extract card text from this image."
                suggestions = [
                    "Move closer so text occupies more of the frame.",
                    "Ensure the card is sharp and avoid motion blur.",
                    "Place the card on a plain background with even light.",
                ]

            if ocr.last_runtime_error:
                warning_message = f"{warning_message} OCR runtime note: {ocr.last_runtime_error}"
            elif ocr.last_parse_hint:
                warning_message = f"{warning_message} OCR parse note: {ocr.last_parse_hint}"

            return IdentifyCardResponse(
                success=False,
                confidence=0.0,
                confidence_label="none",
                card=None,
                warning=warning_message,
                no_match=NoMatchResult(
                    reason=no_match_reason,
                    candidate_count=0,
                    suggestions=suggestions,
                ),
                debug=debug,
            )

        mapped = map_to_frontend_fields(best_match.payload)
        debug.price_updated_at = mapped["price_updated_at"]

        if confidence < LOW_CONFIDENCE_ACCEPTANCE_THRESHOLD:
            return IdentifyCardResponse(
                success=False,
                confidence=confidence,
                confidence_label="none",
                card=None,
                warning="A candidate was found, but confidence is too low to auto-accept.",
                no_match=NoMatchResult(
                    reason="low_confidence",
                    candidate_count=len(candidates),
                    suggestions=[
                        "Retake the photo with the card centered and upright.",
                        "Ensure the collector number text is sharp and unobstructed.",
                        "Use neutral lighting and avoid heavy shadows.",
                    ],
                ),
                debug=debug,
            )

        warning = None
        if confidence_label == "low":
            warning = "Low confidence result. Please verify card details manually."

        return IdentifyCardResponse(
            success=True,
            confidence=confidence,
            confidence_label=confidence_label,
            warning=warning,
            card=CardResult(
                id=mapped["id"],
                name=mapped["name"],
                collection=mapped["collection"],
                collector_number=mapped["collector_number"],
                image_url=mapped["image_url"],
                market_price_usd=mapped["market_price_usd"],
                market_price_source=mapped["market_price_source"],
            ),
            no_match=None,
            debug=debug,
        )
    finally:
        end_debug_image_session(debug_token)
