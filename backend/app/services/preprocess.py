from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np

from app.services.detector import CardDetection
from app.config.config import (
    CARD_TARGET_WIDTH, 
    CARD_TARGET_HEIGHT, 
)


@dataclass
class PreprocessedCard:
    image: np.ndarray
    detected_card: bool
    source: str = "none"


def decode_image(image_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image data")
    return image


def warp_cards(image: np.ndarray, detections: list[CardDetection]) -> list[PreprocessedCard]:
    cards: list[PreprocessedCard] = []
    for detection in detections:
        warped = _warp_detection(image, detection)
        if warped is None:
            continue
        cards.append(
            PreprocessedCard(
                image=warped,
                detected_card=True,
                source="yolo",
            )
        )
    return cards


def extract_regions(card_image: np.ndarray) -> dict[str, list[np.ndarray]]:
    # Modern English layouts generally keep these fields in stable zones after rectification.
    # Set symbol is usually adjacent to collector number; share the same ROI candidates.
    return {
        "number": _crop_by_norm(card_image, 0.05, 0.87, 0.42, 0.965),
        "name": _crop_by_norm(card_image, 0.06, 0.03, 0.70, 0.12),
    }


def _warp_detection(image: np.ndarray, detection: CardDetection) -> np.ndarray | None:
    warped = _warp_from_corners(image, detection.corners)
    if warped is None:
        return None
    return warped


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


def _crop_by_norm(image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    h, w = image.shape[:2]

    left = max(0, min(w - 1, int(round(x1 * w))))
    top = max(0, min(h - 1, int(round(y1 * h))))
    right = max(left + 1, min(w, int(round(x2 * w))))
    bottom = max(top + 1, min(h, int(round(y2 * h))))

    return image[top:bottom, left:right]