from app.services.detector import get_card_detector
from app.models.schemas import Card
from app.services.preprocess import decode_image, warp_cards, extract_regions
from app.services.ocr import predict_card
from app.services.tcgdex import get_matching_card, get_card_details


async def identify_cards(image_bytes: bytes) -> list[Card]:
    image = await decode_image(image_bytes)

    detector = get_card_detector()
    cards = detector.detect(image)
    warped_cards = warp_cards(image, cards)

    cards = []

    for warped_card in warped_cards:
        regions = extract_regions(warped_card.image)
        predicted_card = await predict_card(regions)
        card = await get_matching_card(predicted_card)

        if card:
            cards.append(card)
    return cards