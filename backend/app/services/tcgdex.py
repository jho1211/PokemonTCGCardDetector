from __future__ import annotations
from app.models.schemas import Card as CardResponse

from tcgdexsdk import Query, TCGdex  # type: ignore[reportMissingImports]

TCGDEX_CLIENT = TCGdex()

async def get_matching_card(card: CardResponse | None) -> CardResponse | None:
    if card is None:
        return None
    
    query = Query()

    if card.name:
        query = query.contains("name", card.name)
    if card.collector_number:
        query = query.contains("localId", int(card.collector_number))
    if card.id:
        query = query.equal("id", card.id)

    matches = await TCGDEX_CLIENT.card.list(query)
    if matches:
        main_match = await matches[0].get_full_card()
        if main_match:
            card.id = main_match.id
            card.image_url = main_match.get_image_url(quality="low", extension="webp")
            # need a different SDK or API to get the prices
            # card.market_price_source = details.market_price_source
            # card.market_price_usd = details.market_price_usd
            # card.price_updated_at = details.price_updated_at
            return card

    return None