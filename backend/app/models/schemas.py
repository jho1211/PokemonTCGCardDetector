from __future__ import annotations
from pydantic import BaseModel, SkipValidation
import numpy as np


class Card(BaseModel):
    id: str | None = None
    name: str = ""
    set_id: int | None = None
    collector_number: str | None = None
    image_url: str | None = None
    market_price_usd: float | None = None
    market_price_source: str | None = None
    price_updated_at: str | None = None
    preprocessed_card: PreprocessedCard | None = None


class PreprocessedCard(BaseModel):
    image: SkipValidation[str] = ""
    detected_card: bool = True
    source: str = ""