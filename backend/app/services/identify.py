from __future__ import annotations
from dataclasses import dataclass

import cv2
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
    save_debug_image,
)
from app.services.symbol_matcher import SymbolMatchResult, get_symbol_matcher
from app.services.tcgdex import map_to_frontend_fields, search_cards

from app.config.config import (
    MAX_DB_RESULTS,
    MAX_NAME_CROPS,
    MAX_NUMBER_CROPS,
    MAX_ROTATIONS,
    MAX_SYMBOL_CROPS,
    MIN_ACCEPTED_CONFIDENCE,
    OCR_COLLECTOR_EARLY_STOP_SCORE,
    OCR_NAME_EARLY_STOP_SCORE,
    OCR_ORIENTATION_EARLY_STOP_SCORE,
    OCR_ROTATION_PRIORITY,
)

@dataclass
class _CardAnalysis:
    orientation_turns: int
    evaluated_rotations: int
    orientation_score: float
    collector_number: OCRFieldResult
    card_name: OCRFieldResult
    symbol_abbreviation: OCRFieldResult
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
                rotation_attempts=analysis.evaluated_rotations,
                ocr_available=ocr.is_available,
                ocr_number=analysis.collector_number.text,
                ocr_number_raw=analysis.collector_number.raw_text,
                ocr_number_confidence=analysis.collector_number.confidence,
                ocr_name=analysis.card_name.text,
                ocr_name_raw=analysis.card_name.raw_text,
                ocr_name_confidence=analysis.card_name.confidence,
                ocr_set_abbreviation=analysis.symbol_abbreviation.text,
                ocr_set_abbreviation_raw=analysis.symbol_abbreviation.raw_text,
                ocr_set_abbreviation_confidence=analysis.symbol_abbreviation.confidence,
                ocr_backend=ocr.last_call_backend,
                ocr_result_format=ocr.last_result_format,
                ocr_runtime_error=ocr.last_runtime_error,
                set_match=analysis.symbol_match.set_id,
                set_match_score=analysis.symbol_match.score,
                set_match_method=analysis.symbol_match.method,
                set_match_token=analysis.symbol_match.token,
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
        evaluated_rotations=0,
        orientation_score=-1.0,
        collector_number=OCRFieldResult(text=None, raw_text=None, confidence=0.0),
        card_name=OCRFieldResult(text=None, raw_text=None, confidence=0.0),
        symbol_abbreviation=OCRFieldResult(text=None, raw_text=None, confidence=0.0),
        symbol_match=SymbolMatchResult(set_id=None, set_name=None, score=0.0),
    )

    # Save the original card for reference
    save_debug_image("20_original_card", card_image)

    evaluated_rotations = 0
    for rotation_index, turns in enumerate(_rotation_order(MAX_ROTATIONS)):
        evaluated_rotations += 1
        oriented = rotate_image_90(card_image, turns)
        
        # Save each rotation variant
        save_debug_image(f"21_rotated_{turns * 90}deg", oriented)
        
        regions = extract_regions(oriented)

        # Save extracted region crops for debugging
        _save_debug_regions(regions, turns)

        if ocr_available:
            number_result = ocr.extract_best_collector_number(
                regions["number"][:MAX_NUMBER_CROPS],
                stop_score=OCR_COLLECTOR_EARLY_STOP_SCORE,
            )
            name_result = ocr.extract_best_name(
                regions["name"][:MAX_NAME_CROPS],
                stop_score=OCR_NAME_EARLY_STOP_SCORE,
            )
        else:
            number_result = OCRFieldResult(text=None, raw_text=None, confidence=0.0)
            name_result = OCRFieldResult(text=None, raw_text=None, confidence=0.0)

        should_evaluate_symbol = rotation_index == 0 or _is_strong_ocr_pair(number_result, name_result)
        symbol_result = SymbolMatchResult(set_id=None, set_name=None, score=0.0)
        symbol_abbreviation = OCRFieldResult(text=None, raw_text=None, confidence=0.0)

        # Set symbol handling runs on the first rotation and later only after strong OCR fields.
        if should_evaluate_symbol and symbol_available:
            symbol_result = symbol_matcher.match(regions["symbol"][:MAX_SYMBOL_CROPS])

        if should_evaluate_symbol and ocr_available and symbol_matcher.has_ocr_fallback and symbol_result.set_id is None:
            symbol_abbreviation = ocr.extract_best_set_abbreviation(
                regions["symbol"][:MAX_SYMBOL_CROPS],
                stop_score=symbol_matcher.min_ocr_match_score,
            )
            fallback_match = symbol_matcher.match_ocr_abbreviation(
                symbol_abbreviation.text,
                symbol_abbreviation.confidence,
            )
            if fallback_match.set_id is not None:
                symbol_result = fallback_match

        score = _orientation_score(number_result, name_result, symbol_result)
        
        # Log the result of this rotation attempt
        save_debug_image(
            f"22_rotation_{turns}_result_score_{score:.3f}_num_{number_result.text or 'none'}_name_{name_result.text or 'none'}",
            oriented
        )
        
        if score > best.orientation_score:
            best = _CardAnalysis(
                orientation_turns=turns,
                evaluated_rotations=evaluated_rotations,
                orientation_score=score,
                collector_number=number_result,
                card_name=name_result,
                symbol_abbreviation=symbol_abbreviation,
                symbol_match=symbol_result,
            )

        if _should_stop_rotation_search(best):
            break

    best.evaluated_rotations = evaluated_rotations
    return best


def _rotation_order(max_rotations: int) -> tuple[int, ...]:
    limit = max(1, min(4, max_rotations))
    return OCR_ROTATION_PRIORITY[:limit]


def _is_strong_ocr_pair(number: OCRFieldResult, name: OCRFieldResult) -> bool:
    if not number.text or not name.text:
        return False
    return number.confidence >= OCR_COLLECTOR_EARLY_STOP_SCORE and name.confidence >= OCR_NAME_EARLY_STOP_SCORE


def _should_stop_rotation_search(analysis: _CardAnalysis) -> bool:
    if _is_strong_ocr_pair(analysis.collector_number, analysis.card_name):
        return True

    if analysis.collector_number.text and analysis.card_name.text:
        return analysis.orientation_score >= OCR_ORIENTATION_EARLY_STOP_SCORE

    return False


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


def _save_debug_regions(regions: dict[str, list[np.ndarray]], rotation_turns: int) -> None:
    """Save extracted region crops as a composite image for visual inspection."""
    if not regions or all(not crops for crops in regions.values()):
        return

    # Create a labeled composite showing all extracted regions
    def _add_crops_to_canvas(crops: list[np.ndarray], label: str, canvas: np.ndarray, start_x: int, start_y: int) -> None:
        if not crops:
            cv2.putText(
                canvas,
                f"{label}: none",
                (start_x + 10, start_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (100, 100, 100),
                1,
            )
            return

        cv2.putText(
            canvas,
            f"{label} ({len(crops)} crops):",
            (start_x + 10, start_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        crop_y = start_y + 40
        for idx, crop in enumerate(crops[:3]):  # Show max 3 crops per region
            if crop.size == 0:
                continue
            
            # Resize crop for visibility (max height 80px)
            crop_h, crop_w = crop.shape[:2]
            if crop_h > 80:
                scale = 80 / crop_h
                crop_resized = cv2.resize(crop, (int(crop_w * scale), 80))
            else:
                crop_resized = crop
            
            # Place on canvas
            ch, cw = crop_resized.shape[:2]
            if start_x + cw < canvas.shape[1] and crop_y + ch < canvas.shape[0]:
                canvas[crop_y : crop_y + ch, start_x : start_x + cw] = crop_resized
                cv2.putText(
                    canvas,
                    f"crop{idx}",
                    (start_x + 5, crop_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                )
                start_x += cw + 10
                if start_x > 800:
                    start_x = 10
                    crop_y += ch + 15

    # Create canvas: 1200x900 pixels
    canvas = np.ones((900, 1200, 3), dtype=np.uint8) * 30  # Dark background
    
    # Add labels and crops
    _add_crops_to_canvas(regions.get("number", []), "NUMBER", canvas, 10, 10)
    _add_crops_to_canvas(regions.get("name", []), "NAME", canvas, 10, 250)
    _add_crops_to_canvas(regions.get("symbol", []), "SYMBOL", canvas, 10, 500)

    # Save the composite
    save_debug_image(f"23_regions_rotation_{rotation_turns * 90}deg", canvas)
