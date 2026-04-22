import cv2
from pathlib import Path
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.detector import get_card_detector
from app.services.preprocess import warp_cards, extract_regions
from app.services.ocr import predict_card
from app.services.tcgdex import get_matching_card

import asyncio

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

    cards = []

    for warped_card in warped_cards:
        regions = extract_regions(warped_card.image)
        predicted_card = asyncio.run(predict_card(regions))
        card = asyncio.run(get_matching_card(predicted_card))

        if card:
            cards.append(card)
    print(*cards, sep="\n")