from app.services.detector import get_card_detector
from app.models.schemas import Card, PreprocessedCard
from app.services.preprocess import decode_image, warp_cards, extract_regions
from app.services.ocr import predict_card
from backend.app.services.identify import get_matching_card
import cv2
import base64

async def identify_cards(image_bytes: bytes) -> list[Card]:
    image = await decode_image(image_bytes)

    detector = get_card_detector()
    cards = detector.detect(image)
    warped_cards = warp_cards(image, cards)

    cards = []

    for warped_card in warped_cards:
        regions = extract_regions(warped_card.preprocessed_card.image)
        predicted_card = await predict_card(warped_card, regions)
        card = get_matching_card(predicted_card)

        warped_card.preprocessed_card.image = base64.b64encode(warped_card.preprocessed_card.image.tobytes()).decode('utf-8')
        cards.append(card)
    return cards