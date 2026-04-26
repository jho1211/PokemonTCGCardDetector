import cv2
from pathlib import Path
import sys
import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.detector import get_card_detector
from app.services.preprocess import warp_cards, extract_regions
from app.services.ocr import predict_card
from app.services.identify import get_matching_card
from app.models.schemas import Card

import asyncio
import base64


def _as_region_list(region_value):
    if isinstance(region_value, list):
        return region_value
    return [region_value]


def build_labeled_regions_image(regions, canvas_width: int = 1000, row_height: int = 140) -> np.ndarray:
    normalized_regions: list[tuple[str, np.ndarray]] = []

    for label, region_value in regions.items():
        for region_crop in _as_region_list(region_value):
            if region_crop is None or region_crop.size == 0:
                continue
            normalized_regions.append((label, region_crop))

    if not normalized_regions:
        return np.full((120, canvas_width, 3), 255, dtype=np.uint8)

    canvas_height = 20 + len(normalized_regions) * row_height
    canvas = np.full((canvas_height, canvas_width, 3), 245, dtype=np.uint8)

    y = 10
    label_x = 16
    image_x = 220
    image_box_width = canvas_width - image_x - 20

    for label, region_crop in normalized_regions:
        cv2.rectangle(canvas, (10, y), (canvas_width - 10, y + row_height - 10), (220, 220, 220), 1)
        cv2.putText(canvas, label, (label_x, y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2, cv2.LINE_AA)

        crop_h, crop_w = region_crop.shape[:2]
        scale = min(image_box_width / crop_w, (row_height - 20) / crop_h)
        new_w = max(1, int(crop_w * scale))
        new_h = max(1, int(crop_h * scale))
        resized_crop = cv2.resize(region_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        crop_y = y + ((row_height - 10) - new_h) // 2
        canvas[crop_y:crop_y + new_h, image_x:image_x + new_w] = resized_crop
        y += row_height

    return canvas

class CardInfo:
    def __init__(self, name: str | None = None, 
                 collector_number: str | None = None, 
                 set: str | None = None):
        self.name = name
        self.collector_number = collector_number
        self.set = set
        self.image_url = None
    
    def __str__(self):
        return f"CardInfo(name={self.name}, collector_number={self.collector_number}, set={self.set}, img_url={self.image_url})"


if __name__ == "__main__":
    detector = get_card_detector()
    image = cv2.imread("data/test_card2.jpg")
    cards = detector.detect(image)
    warped_cards = warp_cards(image, cards)
    debug_dir = Path("debug_outputs")
    debug_dir.mkdir(parents=True, exist_ok=True)

    cards = []

    for idx, warped_card in enumerate(warped_cards):
        # cv2.imwrite(str(debug_dir / f"warped_{idx}.jpg"), warped_card.image)
        regions = extract_regions(warped_card.preprocessed_card.image)
        region_debug_image = build_labeled_regions_image(regions)
        # cv2.imwrite(str(debug_dir / f"regions_{idx}.jpg"), region_debug_image)
        # warped_card.preprocessed_card.image = base64.b64encode(warped_card.preprocessed_card.image.tobytes()).decode('utf-8')
        warped_card.preprocessed_card.image = "" # Clear the image data to avoid bloating the output
        predicted_card = asyncio.run(predict_card(warped_card, regions))
        print(f"Predicted card from OCR: {predicted_card}")
        card = get_matching_card(predicted_card)

        if card:
            cards.append(card)
    print(*cards, sep="\n")