import requests
import json
from pathlib import Path
import sys
from typing import Any

# Ensure backend root (the folder containing `app/`) is on sys.path
BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.models.schemas import Card

SETS = [23237]
BASE_URL = "https://tcgtracking.com/tcgapi/v1"
CARDS_DATA_PATH = "data/all_cards.json"

class TCGTrackingClient:
    def __init__(self):
        self.base_url = "https://tcgtracking.com/tcgapi/v1"
        self.sets = SETS
        self.output_path = CARDS_DATA_PATH


    def add_card_prices(self, cards: dict[int, Card], set_id: int) -> list[Card]:
        """Fetches price info for a list of cards and updates their market_price_usd and price_updated_at fields"""
        if not cards:
            return []

        response = requests.get(f"{self.base_url}/3/sets/{set_id}/pricing")
        if response.ok:
            response = response.json()
        else:
            print(f"Failed to fetch prices for set {set_id}: {response.status_code} - {response.text}")
            # If the request fails, we return the original cards without price info
            return cards
        
        prices = response.get("prices", {})

        # If there are no prices, we can return the original cards without price info
        if len(prices) == 0:
            return cards
        
        for _, card in cards.items():
            if card.id in prices:
                tcg_price = prices[card.id].get("tcg", None)
                if tcg_price and "Normal" in tcg_price:
                    card.market_price_usd = tcg_price["Normal"].get("market")
                card.price_updated_at = response.get("updated", None)
                card.market_price_source = "TCGPlayer"
        return cards


    def fetch_cards_by_set(self, set_id: int) -> dict[int, Card]:
        """Fetches all cards in a set excluding their prices"""
        url = f"{self.base_url}/3/sets/{set_id}"
        response = requests.get(url)
        if response.ok:
            response = response.json()
        else:
            print(f"Failed to fetch cards for set {set_id}: {response.status_code} - {response.text}")
            return []
        
        # map the card ID with the card itself for easy lookup later 
        # when we want to add price info to the cards
        cards = {}
        
        for card in response["products"]:
            cards[card["id"]] = Card(
                id=str(card["id"]),
                name=card["name"],
                set_id=response["set_id"],
                collector_number=card["number"],
                image_url=card["image_url"]
            )
        return cards
    

    def get_cards_by_set(self, set_id: int) -> dict[int, Card]:
        """Fetches all cards in a set including their prices"""
        card_dict = self.fetch_cards_by_set(set_id)
        return self.add_card_prices(card_dict, set_id)
    

    def get_all_cards(self) -> dict[int, Card]:
        """Fetches all cards in all sets including their prices"""
        all_cards = {}
        for set_id in SETS:
            cards = self.get_cards_by_set(set_id)
            all_cards[set_id] = cards
        return all_cards
    

    def write_cards_to_file(self, set_cards_dict: dict, filename: str):
        """Writes a list of cards to a JSON file"""
        def to_jsonable(value: Any):
            if isinstance(value, Card):
                return value.model_dump()
            if isinstance(value, dict):
                return {k: to_jsonable(v) for k, v in value.items()}
            if isinstance(value, list):
                return [to_jsonable(item) for item in value]
            if isinstance(value, tuple):
                return [to_jsonable(item) for item in value]
            return value

        serialized = to_jsonable(set_cards_dict)
        with open(filename, "w") as f:
            json.dump(serialized, f, indent=4)

if __name__ == "__main__":
    client = TCGTrackingClient()
    all_cards = client.get_all_cards()
    client.write_cards_to_file(all_cards, client.output_path)