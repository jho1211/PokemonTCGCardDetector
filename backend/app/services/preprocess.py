from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import cv2
import numpy as np

from app.services.detector import CardDetection, get_card_detector
from app.config.config import (
    CARD_TARGET_WIDTH, 
    CARD_TARGET_HEIGHT, 
    CARD_RATIO_WIDTH_MM, 
    CARD_RATIO_HEIGHT_MM, 
    MAX_DETECTED_CARDS,
    _DEBUG_SESSION_ID,
    _DEBUG_ROOT,
    _DEBUG_SAVE_TRANSFORMS
)


@dataclass
class PreprocessedCard:
    image: np.ndarray
    score: float
    detected_card: bool
    detection_confidence: float | None = None
    source: str = "none"


def begin_debug_image_session(prefix: str) -> str:
    session_id = f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    _DEBUG_SESSION_ID.set(session_id)
    return session_id


def end_debug_image_session(session_id: str) -> None:
    current = _DEBUG_SESSION_ID.get()
    if current == session_id:
        _DEBUG_SESSION_ID.set(None)


def decode_image(image_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image data")
    return image


def detect_and_warp_card(image: np.ndarray) -> PreprocessedCard:
    cards = detect_and_warp_cards(image, max_cards=1)
    if cards:
        return cards[0]

    # Fall back to resized original image so OCR can still run in degraded mode.
    return PreprocessedCard(
        image=_resize_to_canvas(image),
        score=0.0,
        detected_card=False,
        detection_confidence=None,
        source="fallback",
    )


def detect_and_warp_cards(image: np.ndarray, max_cards: int | None = None) -> list[PreprocessedCard]:
    detector = get_card_detector()
    detection_limit = max_cards if max_cards is not None else MAX_DETECTED_CARDS

    detections = detector.detect(image)
    _save_debug_image("00_input", image)

    cards: list[PreprocessedCard] = []
    for idx, detection in enumerate(detections[:detection_limit]):
        warped, preprocess_score = _warp_detection(image, detection)
        if warped is None:
            continue

        _save_debug_image(f"10_warped_{idx}", warped)
        cards.append(
            PreprocessedCard(
                image=warped,
                score=preprocess_score,
                detected_card=True,
                detection_confidence=detection.confidence,
                source="yolo",
            )
        )

    if cards:
        cards.sort(key=lambda item: item.score, reverse=True)
        return cards

    contour = _detect_largest_quad_contour(image)
    if contour is None:
        return []

    corners, contour_score = contour
    warped = _warp_from_corners(image, corners)
    if warped is None:
        return []

    _save_debug_image("11_warped_contour", warped)
    return [
        PreprocessedCard(
            image=warped,
            score=contour_score,
            detected_card=True,
            detection_confidence=None,
            source="contour_fallback",
        )
    ]


def rotate_image_90(image: np.ndarray, turns: int) -> np.ndarray:
    normalized_turns = turns % 4
    if normalized_turns == 0:
        return image
    return np.ascontiguousarray(np.rot90(image, k=normalized_turns))


def extract_regions(card_image: np.ndarray) -> dict[str, list[np.ndarray]]:
    # Modern English layouts generally keep these fields in stable zones after rectification.
    number_regions = [
        _crop_by_norm(card_image, 0.05, 0.90, 0.35, 0.985),
        _crop_by_norm(card_image, 0.16, 0.90, 0.48, 0.985),
        _crop_by_norm(card_image, 0.62, 0.90, 0.96, 0.985),
        _crop_by_norm(card_image, 0.05, 0.87, 0.42, 0.965),
    ]

    name_regions = [
        _crop_by_norm(card_image, 0.06, 0.03, 0.70, 0.12),
        _crop_by_norm(card_image, 0.16, 0.03, 0.84, 0.12),
        _crop_by_norm(card_image, 0.07, 0.05, 0.78, 0.15),
    ]

    symbol_regions = [
        _crop_by_norm(card_image, 0.74, 0.80, 0.94, 0.92),
        _crop_by_norm(card_image, 0.70, 0.78, 0.96, 0.93),
        _crop_by_norm(card_image, 0.62, 0.80, 0.88, 0.93),
    ]

    return {
        "number": [crop for crop in number_regions if crop.size > 0],
        "name": [crop for crop in name_regions if crop.size > 0],
        "symbol": [crop for crop in symbol_regions if crop.size > 0],
    }


def _warp_detection(image: np.ndarray, detection: CardDetection) -> tuple[np.ndarray | None, float]:
    refined_corners = _refine_quad_corners_from_bbox(image, detection.bbox)
    corners = refined_corners if refined_corners is not None else detection.corners

    warped = _warp_from_corners(image, corners)
    if warped is None:
        return None, 0.0

    corner_quality = 1.0 if refined_corners is not None else 0.85
    score = (0.75 * detection.confidence) + (0.25 * corner_quality)
    return warped, float(max(0.0, min(1.0, score)))


def _warp_from_corners(image: np.ndarray, corners: np.ndarray) -> np.ndarray | None:
    if corners.shape != (4, 2):
        return None

    src = _order_quad_points(corners.astype(np.float32))
    dst = np.array(
        [
            [0, 0],
            [CARD_TARGET_WIDTH - 1, 0],
            [CARD_TARGET_WIDTH - 1, CARD_TARGET_HEIGHT - 1],
            [0, CARD_TARGET_HEIGHT - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (CARD_TARGET_WIDTH, CARD_TARGET_HEIGHT), flags=cv2.INTER_LINEAR)
    if warped is None or warped.size == 0:
        return None
    return warped


def _order_quad_points(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)

    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]

    return ordered


def _detect_largest_quad_contour(image: np.ndarray) -> tuple[np.ndarray, float] | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = float(image.shape[0] * image.shape[1])
    best_quad: np.ndarray | None = None
    best_score = 0.0

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4:
            continue

        area = cv2.contourArea(approx)
        if area < image_area * 0.05:
            continue

        corners = approx.reshape(4, 2).astype(np.float32)

        x, y, w, h = cv2.boundingRect(approx)
        ratio = w / max(1, h)
        aspect_delta = min(abs(ratio - (CARD_RATIO_WIDTH_MM / CARD_RATIO_HEIGHT_MM)), abs(ratio - (CARD_RATIO_HEIGHT_MM / CARD_RATIO_WIDTH_MM)))
        aspect_score = max(0.0, 1.0 - (aspect_delta / 0.9))

        area_score = min(1.0, area / image_area)
        score = (0.6 * area_score) + (0.4 * aspect_score)

        if score > best_score:
            best_score = score
            best_quad = corners

    if best_quad is None:
        return None

    return best_quad, float(max(0.0, min(1.0, best_score)))


def _refine_quad_corners_from_bbox(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox

    pad_x = int((x2 - x1) * 0.08)
    pad_y = int((y2 - y1) * 0.08)

    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(width, x2 + pad_x)
    bottom = min(height, y2 + pad_y)

    if right - left < 20 or bottom - top < 20:
        return None

    roi = image[top:bottom, left:right]
    contour = _detect_largest_quad_contour(roi)
    if contour is None:
        return None

    corners, _ = contour
    corners[:, 0] += left
    corners[:, 1] += top
    return corners


def _resize_to_canvas(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (CARD_TARGET_WIDTH, CARD_TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)


def _crop_by_norm(image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    h, w = image.shape[:2]

    left = max(0, min(w - 1, int(round(x1 * w))))
    top = max(0, min(h - 1, int(round(y1 * h))))
    right = max(left + 1, min(w, int(round(x2 * w))))
    bottom = max(top + 1, min(h, int(round(y2 * h))))

    return image[top:bottom, left:right]


def _save_debug_image(name: str, image: np.ndarray) -> None:
    if not _DEBUG_SAVE_TRANSFORMS:
        return

    session_id = _DEBUG_SESSION_ID.get()
    if not session_id:
        return

    output_dir = _DEBUG_ROOT / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{name}.png"
    cv2.imwrite(str(output_path), image)
