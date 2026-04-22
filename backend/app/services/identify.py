from __future__ import annotations
import json
from app.models.schemas import Card

from tcgdexsdk import Query, TCGdex

class Identifier:
    def __init__(self):
        self.client = TCGdex()
        self.base_url = "https://tcgtracking.com/tcgapi/v1"
        self.set_db = self._load_db_as_cards("data/all_cards.json")

    def _load_db_as_cards(self, db_path: str) -> dict[str, dict[str, Card]]:
        """Load card DB JSON and validate each card payload into a Card model."""
        with open(db_path, "r") as f:
            raw_db = json.load(f)

        card_db: dict[str, dict[str, Card]] = {}
        for set_id, cards in raw_db.items():
            card_db[str(set_id)] = {
                str(card_id): Card.model_validate(card_data)
                for card_id, card_data in cards.items()
            }
        return card_db

    def tcgdex_query_card(self, card: Card | None) -> Card | None:
        """Query TCGDex for a card as a fallback if we can't find it in the TCGTracking DB"""
        if card is None:
            return None

        query = Query()
        if card.name:
            query = query.contains("name", card.name)
        if card.collector_number:
            query = query.contains("localId", int(card.collector_number))
        if card.id:
            query = query.equal("id", card.id)

        matches = self.client.card.list(query)
        if matches:
            match = matches[0]
            card.id = match.id
            card.image_url = match.get_image_url(quality="low", extension="webp")
            return card
        return None
    
    def tcgtracking_query_card(self, card: Card | None) -> Card | None:
        if card is None:
            return None

        set_id = card.set_id
        set_db = self.set_db.get(str(set_id), None)
        if set_db is None:
            return None

        for _card in set_db.values():
            if card.collector_number == _card.collector_number and card.name in _card.name:
                return _card
        return None

identifier = Identifier()

def get_matching_card(card: Card | None) -> Card | None:
    return identifier.tcgtracking_query_card(card)