from __future__ import annotations

from pydantic import BaseModel


class Card(BaseModel):
    id: str | None = None
    name: str = ""
    set_id: int | None = None
    collector_number: str | None = None
    image_url: str | None = None
    market_price_usd: float | None = None
    market_price_source: str | None = None
    price_updated_at: str | None = None