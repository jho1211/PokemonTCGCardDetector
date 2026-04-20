from typing import Literal

from pydantic import BaseModel, Field


class CardResult(BaseModel):
    id: str
    name: str
    collection: str
    collector_number: str
    image_url: str
    market_price_usd: float | None = Field(default=None)
    market_price_source: str = Field(default="tcgplayer.normal.marketPrice")


class DebugResult(BaseModel):
    set_match: str | None = None
    ocr_number: str | None = None
    ocr_number_raw: str | None = None
    ocr_name: str | None = None
    preprocess_score: float | None = None
    contour_confidence: float | None = None
    orientation_degrees: int | None = None
    ocr_number_confidence: float | None = None
    ocr_name_confidence: float | None = None
    matched_by: str | None = None
    tcgdex_query: str | None = None
    price_updated_at: str | None = None


class NoMatchResult(BaseModel):
    reason: str
    candidate_count: int = 0
    suggestions: list[str] = Field(default_factory=list)


class IdentifyCardResponse(BaseModel):
    success: bool
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_label: Literal["high", "medium", "low", "none"]
    card: CardResult | None = None
    warning: str | None = None
    no_match: NoMatchResult | None = None
    debug: DebugResult
