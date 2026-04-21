from __future__ import annotations

from pydantic import BaseModel, Field


class NoMatchInfo(BaseModel):
    reason: str
    suggestions: list[str] = Field(default_factory=list)


class CardPayload(BaseModel):
    id: str
    name: str
    set_id: str | None = None
    collection: str
    collector_number: str
    image_url: str = ""
    market_price_usd: float | None = None
    market_price_source: str | None = None
    price_updated_at: str | None = None


class CardDebugInfo(BaseModel):
    detection_confidence: float | None = None
    preprocess_score: float | None = None
    orientation_degrees: int | None = None
    rotation_attempts: int | None = None
    ocr_available: bool | None = None
    ocr_number: str | None = None
    ocr_number_raw: str | None = None
    ocr_number_confidence: float | None = None
    ocr_name: str | None = None
    ocr_name_raw: str | None = None
    ocr_name_confidence: float | None = None
    ocr_backend: str | None = None
    ocr_result_format: str | None = None
    ocr_runtime_error: str | None = None
    set_match: str | None = None
    set_match_score: float | None = None
    query_debug: str | None = None


class CardResult(BaseModel):
    detection_index: int
    success: bool
    confidence: float
    confidence_label: str
    card: CardPayload | None = None
    warning: str | None = None
    no_match: NoMatchInfo | None = None
    debug: CardDebugInfo = Field(default_factory=CardDebugInfo)


class IdentifyCardResponse(BaseModel):
    success: bool
    cards: list[CardResult] = Field(default_factory=list)
    confidence: float
    confidence_label: str
    warning: str | None = None
