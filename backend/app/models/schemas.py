from __future__ import annotations

from pydantic import BaseModel


class Card(BaseModel):
    id: str = ""
    name: str = ""
    set_id: str | None = None
    collector_number: str = ""
    image_url: str = ""
    market_price_usd: float | None = None
    market_price_source: str | None = None
    price_updated_at: str | None = None