from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

try:
    from ultralytics import YOLO  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover - handled by runtime availability checks
    YOLO = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Real Pokemon card ratio in portrait orientation.
CARD_ASPECT_RATIO = 63.0 / 88.0


@dataclass
class CardDetection:
    corners: np.ndarray
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int | None
    class_name: str | None


class YoloCardDetector:
    def __init__(self) -> None:
        self.model_path = os.getenv("YOLO_MODEL_PATH", "yolo11n.pt")
        self.conf_threshold = float(os.getenv("YOLO_CONFIDENCE", "0.2"))
        self.iou_threshold = float(os.getenv("YOLO_IOU", "0.45"))
        self.max_detections = int(os.getenv("YOLO_MAX_DETECTIONS", "8"))
        self.device = os.getenv("YOLO_DEVICE", "cpu")

        self.model: Any | None = None
        self.is_available = False
        self.last_init_error: str | None = None
        self.last_runtime_error: str | None = None

        self._initialize_model()

    def _initialize_model(self) -> None:
        if YOLO is None:
            self.last_init_error = "ultralytics is not installed"
            logger.warning("YOLO unavailable: %s", self.last_init_error)
            return

        try:
            self.model = YOLO(self.model_path)
            self.is_available = True
            logger.info("YOLO model loaded: %s", self.model_path)
        except Exception as exc:  # pragma: no cover - runtime dependency/model issue
            self.last_init_error = str(exc)
            self.is_available = False
            logger.warning("Failed to load YOLO model '%s': %s", self.model_path, exc)

    def detect(self, image: np.ndarray) -> list[CardDetection]:
        if image.size == 0:
            return []
        if not self.is_available or self.model is None:
            return []

        try:
            prediction = self.model.predict(
                source=image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - runtime model execution issue
            self.last_runtime_error = str(exc)
            logger.warning("YOLO inference failed: %s", exc)
            return []

        if not prediction:
            return []

        result = prediction[0]
        detections: list[CardDetection] = []

        detections.extend(self._parse_oriented_boxes(result))
        if detections:
            detections.sort(key=lambda item: item.confidence, reverse=True)
            return detections

        detections.extend(self._parse_axis_aligned_boxes(result))
        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections

    def _parse_oriented_boxes(self, result: Any) -> list[CardDetection]:
        obb = getattr(result, "obb", None)
        if obb is None:
            return []

        points = _to_numpy(getattr(obb, "xyxyxyxy", None))
        conf_values = _to_numpy(getattr(obb, "conf", None)).reshape(-1)
        class_values = _to_numpy(getattr(obb, "cls", None)).reshape(-1)
        name_map = _resolve_name_map(result, self.model)

        if points.ndim != 3 or points.shape[1] != 4 or points.shape[2] != 2:
            return []

        output: list[CardDetection] = []
        for idx, corners in enumerate(points):
            corners = corners.astype(np.float32)
            bbox = _corners_to_bbox(corners)
            width = float(np.linalg.norm(corners[1] - corners[0]))
            height = float(np.linalg.norm(corners[2] - corners[1]))
            aspect_bonus = _aspect_fit_score(width, height)

            raw_conf = float(conf_values[idx]) if idx < len(conf_values) else 0.0
            combined_conf = (0.8 * raw_conf) + (0.2 * aspect_bonus)

            class_id = int(class_values[idx]) if idx < len(class_values) else None
            class_name = name_map.get(class_id) if class_id is not None else None

            output.append(
                CardDetection(
                    corners=corners,
                    bbox=bbox,
                    confidence=float(max(0.0, min(1.0, combined_conf))),
                    class_id=class_id,
                    class_name=class_name,
                )
            )

        return output

    def _parse_axis_aligned_boxes(self, result: Any) -> list[CardDetection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = _to_numpy(getattr(boxes, "xyxy", None))
        conf_values = _to_numpy(getattr(boxes, "conf", None)).reshape(-1)
        class_values = _to_numpy(getattr(boxes, "cls", None)).reshape(-1)
        name_map = _resolve_name_map(result, self.model)

        if xyxy.ndim != 2 or xyxy.shape[1] != 4:
            return []

        output: list[CardDetection] = []
        for idx, raw_box in enumerate(xyxy):
            x1, y1, x2, y2 = [int(round(v)) for v in raw_box.tolist()]
            width = max(1, x2 - x1)
            height = max(1, y2 - y1)

            aspect_bonus = _aspect_fit_score(width, height)
            raw_conf = float(conf_values[idx]) if idx < len(conf_values) else 0.0
            combined_conf = (0.7 * raw_conf) + (0.3 * aspect_bonus)

            # Keep broad acceptance because the default model is not card-specific.
            if combined_conf < 0.10:
                continue

            class_id = int(class_values[idx]) if idx < len(class_values) else None
            class_name = name_map.get(class_id) if class_id is not None else None

            corners = np.array(
                [
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2],
                ],
                dtype=np.float32,
            )

            output.append(
                CardDetection(
                    corners=corners,
                    bbox=(x1, y1, x2, y2),
                    confidence=float(max(0.0, min(1.0, combined_conf))),
                    class_id=class_id,
                    class_name=class_name,
                )
            )

        return output


@lru_cache(maxsize=1)
def get_card_detector() -> YoloCardDetector:
    return YoloCardDetector()


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.array([])

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()

    return np.asarray(value)


def _resolve_name_map(result: Any, model: Any) -> dict[int, str]:
    names = getattr(result, "names", None)
    if not names and model is not None:
        names = getattr(model, "names", None)

    if isinstance(names, dict):
        mapped: dict[int, str] = {}
        for key, value in names.items():
            try:
                mapped[int(key)] = str(value)
            except Exception:
                continue
        return mapped

    return {}


def _corners_to_bbox(corners: np.ndarray) -> tuple[int, int, int, int]:
    min_xy = np.min(corners, axis=0)
    max_xy = np.max(corners, axis=0)
    x1 = int(round(min_xy[0]))
    y1 = int(round(min_xy[1]))
    x2 = int(round(max_xy[0]))
    y2 = int(round(max_xy[1]))
    return x1, y1, x2, y2


def _aspect_fit_score(width: float, height: float) -> float:
    if width <= 1 or height <= 1:
        return 0.0

    ratio = width / height
    delta = min(abs(ratio - CARD_ASPECT_RATIO), abs(ratio - (1.0 / CARD_ASPECT_RATIO)))

    # Accept broad deltas while still preferring card-like aspect ratios.
    return float(max(0.0, 1.0 - (delta / 0.85)))
