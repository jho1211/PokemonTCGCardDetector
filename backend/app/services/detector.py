from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from app.config.config import YOLO_MODEL_PATH, YOLO_CONFIDENCE, YOLO_IOU
from ultralytics import YOLO

logger = logging.getLogger(__name__)

@dataclass
class CardDetection:
    bbox: tuple[int, int, int, int]
    corners: np.ndarray

class YoloCardDetector:
    def __init__(self) -> None:
        self.model = YOLO(YOLO_MODEL_PATH)


    def detect(self, image: np.ndarray) -> list[CardDetection]:
        predictions = self.model.predict(
                source=image,
                conf=YOLO_CONFIDENCE,
                iou=YOLO_IOU,
                verbose=False,
            )

        if predictions:
            return self._parse_oriented_boxes(predictions[0])


    def _parse_oriented_boxes(self, result: Any) -> list[CardDetection]:
        output: list[CardDetection] = []
        # Each set of corners represents one card
        for corners in result.obb.xyxyxyxy:
            corners = corners.cpu().numpy()  # Convert from tensor to numpy array
            bbox = _corners_to_bbox(corners)
            output.append(CardDetection(corners=corners, bbox=bbox))
        return output


@lru_cache(maxsize=1)
def get_card_detector() -> YoloCardDetector:
    return YoloCardDetector()


def _corners_to_bbox(corners: np.ndarray) -> tuple[int, int, int, int]:
    min_xy = np.min(corners, axis=0)
    max_xy = np.max(corners, axis=0)
    x1 = int(round(min_xy[0]))
    y1 = int(round(min_xy[1]))
    x2 = int(round(max_xy[0]))
    y2 = int(round(max_xy[1]))
    return x1, y1, x2, y2