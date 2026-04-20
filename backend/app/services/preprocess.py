from __future__ import annotations

from dataclasses import dataclass
from contextvars import ContextVar, Token
from pathlib import Path
import os
import uuid
from collections.abc import Iterable

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    image: np.ndarray
    score: float
    detected_card: bool
    contour_confidence: float = 0.0


DEBUG_IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
SAVE_DEBUG_IMAGES = os.getenv("DEBUG_SAVE_TRANSFORMS", "1").lower() not in {"0", "false", "no"}
DEBUG_IMAGE_SESSION: ContextVar[str | None] = ContextVar("debug_image_session", default=None)

CARD_RATIO = 63.0 / 88.0
WARP_WIDTH = 756
WARP_HEIGHT = 1056


def _save_debug_image(prefix: str, stage: str, image: np.ndarray) -> None:
    if not SAVE_DEBUG_IMAGES or image is None or image.size == 0:
        return

    DEBUG_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_{stage}.png"
    path = DEBUG_IMAGE_DIR / filename
    cv2.imwrite(str(path), image)


def begin_debug_image_session(prefix: str) -> Token[str | None]:
    return DEBUG_IMAGE_SESSION.set(prefix)


def end_debug_image_session(token: Token[str | None]) -> None:
    DEBUG_IMAGE_SESSION.reset(token)


def _debug_prefix() -> str:
    current = DEBUG_IMAGE_SESSION.get()
    if current:
        return current
    return f"{uuid.uuid4().hex[:10]}_{os.getpid()}"


def _order_quad_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]

    top_left_index = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -top_left_index, axis=0)

    if np.cross(ordered[1] - ordered[0], ordered[2] - ordered[0]) < 0:
        ordered[[1, 3]] = ordered[[3, 1]]

    return ordered


def _warp_card(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    ordered = _order_quad_points(quad)
    width, height = WARP_WIDTH, WARP_HEIGHT
    target = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, target)
    return cv2.warpPerspective(image, matrix, (width, height))


def _extract_quad_from_contour(contour: np.ndarray) -> np.ndarray | None:
    if contour is None or len(contour) < 4:
        return None

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) == 4:
        quad = approx.reshape(4, 2).astype(np.float32)
        if cv2.isContourConvex(quad.astype(np.int32)):
            return quad

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)


def _quad_metrics(quad: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    ordered = _order_quad_points(quad)
    width_top = float(np.linalg.norm(ordered[1] - ordered[0]))
    width_bottom = float(np.linalg.norm(ordered[2] - ordered[3]))
    height_left = float(np.linalg.norm(ordered[3] - ordered[0]))
    height_right = float(np.linalg.norm(ordered[2] - ordered[1]))

    width = max(width_top, width_bottom)
    height = max(height_left, height_right)
    if width <= 1.0 or height <= 1.0:
        return 0.0, 0.0, 0.0, ordered

    longer = max(width, height)
    shorter = min(width, height)
    aspect_ratio = shorter / longer
    return width, height, aspect_ratio, ordered


def _quad_score(quad: np.ndarray, contour: np.ndarray, image_shape: tuple[int, int]) -> tuple[float, float] | None:
    image_h, image_w = image_shape
    image_area = float(image_h * image_w)

    width, height, aspect_ratio, ordered = _quad_metrics(quad)
    if width <= 1.0 or height <= 1.0:
        return None

    quad_area = float(cv2.contourArea(ordered.astype(np.float32)))
    contour_area = float(cv2.contourArea(contour.astype(np.float32)))
    if quad_area <= 1.0 or contour_area <= 1.0:
        return None

    area_ratio = quad_area / max(image_area, 1.0)
    if area_ratio < 0.1:
        return None

    rectangularity = float(np.clip(contour_area / quad_area, 0.0, 1.0))
    ratio_error = abs(aspect_ratio - CARD_RATIO)

    quad_center = ordered.mean(axis=0)
    image_center = np.array([image_w / 2.0, image_h / 2.0], dtype=np.float32)
    center_distance = float(np.linalg.norm(quad_center - image_center))
    center_distance_norm = center_distance / max(np.hypot(image_w, image_h), 1.0)

    min_x = float(np.min(ordered[:, 0]))
    min_y = float(np.min(ordered[:, 1]))
    max_x = float(np.max(ordered[:, 0]))
    max_y = float(np.max(ordered[:, 1]))
    border_margin = min(min_x, min_y, image_w - max_x, image_h - max_y)
    border_margin_norm = float(np.clip(border_margin / max(min(image_w, image_h), 1.0), 0.0, 1.0))

    score = 0.0
    score += 3.0 * area_ratio
    score += 2.0 * max(0.0, 1.0 - (ratio_error / 0.30))
    score += 1.1 * rectangularity
    score += 1.2 * max(0.0, 1.0 - (center_distance_norm * 4.0))
    score += 0.6 * border_margin_norm

    confidence = float(np.clip((score - 1.0) / 5.0, 0.0, 1.0))
    return score, confidence


def _resize_for_detection(image: np.ndarray, max_side: int = 1400) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    current_max = max(h, w)
    if current_max <= max_side:
        return image, 1.0

    scale = max_side / float(current_max)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def _build_edge_maps(gray: np.ndarray) -> list[np.ndarray]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    bilateral = cv2.bilateralFilter(gray, 7, 70, 70)

    canny_soft = cv2.Canny(blurred, 50, 150)
    canny_strong = cv2.Canny(blurred, 70, 190)
    adaptive = cv2.adaptiveThreshold(
        bilateral,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    adaptive_edges = cv2.Canny(adaptive, 30, 120)
    closed = cv2.morphologyEx(adaptive_edges, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)

    return [canny_soft, canny_strong, closed]


def _iter_candidate_contours(edge_maps: Iterable[np.ndarray]) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    for edge_map in edge_maps:
        contours, _ = cv2.findContours(edge_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        candidates.extend(sorted(contours, key=cv2.contourArea, reverse=True)[:250])

    return candidates


def _crop_ratio(image: np.ndarray, top: float, bottom: float, left: float, right: float) -> np.ndarray:
    h, w = image.shape[:2]
    y0 = int(np.clip(top * h, 0, h - 1))
    y1 = int(np.clip(bottom * h, y0 + 1, h))
    x0 = int(np.clip(left * w, 0, w - 1))
    x1 = int(np.clip(right * w, x0 + 1, w))
    return image[y0:y1, x0:x1]


def rotate_image_90(card_image: np.ndarray, turns_clockwise: int) -> np.ndarray:
    turns = turns_clockwise % 4
    if turns == 0:
        return card_image
    if turns == 1:
        return cv2.rotate(card_image, cv2.ROTATE_90_CLOCKWISE)
    if turns == 2:
        return cv2.rotate(card_image, cv2.ROTATE_180)
    return cv2.rotate(card_image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Unable to decode image bytes")

    _save_debug_image(_debug_prefix(), "01_decoded", decoded)
    return decoded


def detect_and_warp_card(image: np.ndarray) -> PreprocessResult:
    prefix = _debug_prefix()
    _save_debug_image(prefix, "01_input", image)

    resized, scale = _resize_for_detection(image)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _save_debug_image(prefix, "02_gray", gray)

    edge_maps = _build_edge_maps(gray)
    for idx, edge_map in enumerate(edge_maps, start=1):
        _save_debug_image(prefix, f"03_edges_{idx}", edge_map)

    contours = _iter_candidate_contours(edge_maps)
    if not contours:
        _save_debug_image(prefix, "04_no_contours", image)
        return PreprocessResult(image=image, score=0.2, detected_card=False, contour_confidence=0.0)

    best_quad: np.ndarray | None = None
    best_score = float("-inf")
    best_confidence = 0.0

    for contour in contours:
        if len(contour) < 4:
            continue

        candidate_quad = _extract_quad_from_contour(contour)
        if candidate_quad is None:
            continue

        result = _quad_score(candidate_quad, contour, gray.shape[:2])
        if result is None:
            continue

        score, confidence = result
        if score > best_score:
            best_score = score
            best_quad = candidate_quad
            best_confidence = confidence

    if best_quad is None:
        _save_debug_image(prefix, "04_no_quad", image)
        return PreprocessResult(image=image, score=0.25, detected_card=False, contour_confidence=0.0)

    best_quad_original = best_quad / max(scale, 1e-8)

    debug_overlay = image.copy()
    overlay_quad = best_quad_original.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(debug_overlay, [overlay_quad], isClosed=True, color=(0, 255, 0), thickness=4)
    _save_debug_image(prefix, "04_contour_overlay", debug_overlay)

    warped = _warp_card(image, best_quad_original)
    _save_debug_image(prefix, "05_warped", warped)
    score = float(np.clip(0.35 + (best_confidence * 0.65), 0.0, 1.0))

    return PreprocessResult(
        image=warped,
        score=score,
        detected_card=True,
        contour_confidence=best_confidence,
    )


def extract_regions(card_image: np.ndarray) -> dict[str, list[np.ndarray]]:
    prefix = _debug_prefix()

    number_windows = [
        (0.87, 0.975, 0.02, 0.66),
        (0.885, 0.97, 0.03, 0.58),
        (0.86, 0.965, 0.01, 0.72),
        (0.89, 0.985, 0.00, 0.50),
    ]
    name_windows = [
        (0.015, 0.11, 0.14, 0.76),
        (0.01, 0.13, 0.09, 0.82),
        (0.02, 0.12, 0.18, 0.70),
    ]
    symbol_windows = [
        (0.76, 0.91, 0.67, 0.93),
        (0.74, 0.94, 0.62, 0.96),
    ]

    number_rois = [_crop_ratio(card_image, *window) for window in number_windows]
    name_rois = [_crop_ratio(card_image, *window) for window in name_windows]
    symbol_rois = [_crop_ratio(card_image, *window) for window in symbol_windows]

    for idx, roi in enumerate(number_rois, start=1):
        _save_debug_image(prefix, f"06_number_roi_{idx}", roi)
    for idx, roi in enumerate(name_rois, start=1):
        _save_debug_image(prefix, f"07_name_roi_{idx}", roi)
    for idx, roi in enumerate(symbol_rois, start=1):
        _save_debug_image(prefix, f"08_symbol_roi_{idx}", roi)

    return {
        "number": number_rois,
        "name": name_rois,
        "symbol": symbol_rois,
    }


def prepare_for_ocr(image: np.ndarray, invert: bool = False, save_debug: bool = True) -> np.ndarray:
    prefix = _debug_prefix()
    if save_debug:
        _save_debug_image(prefix, "09_ocr_input", image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if save_debug:
        _save_debug_image(prefix, "10_ocr_gray", gray)

    h, w = gray.shape[:2]
    scale = 2.0 if min(h, w) < 180 else 1.5
    resized = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    if save_debug:
        _save_debug_image(prefix, "11_ocr_resized", resized)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(resized)
    denoised = cv2.bilateralFilter(clahe, 7, 75, 75)
    if save_debug:
        _save_debug_image(prefix, "12_ocr_denoised", denoised)

    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    if invert:
        thresh = cv2.bitwise_not(thresh)

    if save_debug:
        _save_debug_image(prefix, "13_ocr_thresh", thresh)

    prepared = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    if save_debug:
        _save_debug_image(prefix, "14_ocr_prepared", prepared)
    return prepared
