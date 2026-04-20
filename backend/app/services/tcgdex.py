from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from tcgdexsdk import Query, TCGdex  # type: ignore[reportMissingImports]


def _resolve_base_url() -> str:
    base_url = os.getenv("TCGDEX_BASE_URL", "https://api.tcgdex.net/v2")
    base_url = base_url.rstrip("/")
    if base_url.endswith("/en"):
        base_url = base_url[:-3]
    return base_url


TCGDEX_CLIENT = TCGdex()
TCGDEX_CLIENT.setEndpoint(_resolve_base_url())


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_field(payload: Any, key: str, default: Any = None) -> Any:
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


def _dedupe_cards(cards: list[Any]) -> list[Any]:
    deduped: list[Any] = []
    seen: set[str] = set()

    for card in cards:
        card_id = str(_get_field(card, "id", ""))
        if not card_id or card_id in seen:
            continue
        seen.add(card_id)
        deduped.append(card)

    return deduped


def _build_query(collector_number: str | None, card_name: str | None, limit: int) -> Query | None:
    query = Query()
    has_filters = False

    if collector_number:
        query = query.equal("localId", collector_number)
        has_filters = True
    if card_name:
        query = query.equal("name", card_name)
        has_filters = True

    if not has_filters:
        return None

    return query.paginate(page=1, itemsPerPage=limit)


def _query_debug_string(query: Query) -> str | None:
    if not query.params:
        return None
    return "&".join(f"{item['key']}={item['value']}" for item in query.params)


async def _fetch_cards(query: Query) -> list[Any]:
    return await TCGDEX_CLIENT.card.list(query)


async def _fetch_card_details(card_id: str) -> Any | None:
    return await TCGDEX_CLIENT.card.get(card_id)


async def search_cards(
    collector_number: str | None,
    card_name: str | None,
    set_id: str | None = None,
    collection_name: str | None = None,
    limit: int = 20,
) -> tuple[list[Any], str | None]:
    query_candidates: list[Query] = []

    if collector_number and card_name:
        query = _build_query(collector_number, card_name, limit)
        if query is not None:
            query_candidates.append(query)

    if collector_number:
        query = _build_query(collector_number, None, limit)
        if query is not None:
            query_candidates.append(query)

    if card_name:
        query = _build_query(None, card_name, limit)
        if query is not None:
            query_candidates.append(query)

    results: list[Any] = []
    query_debug: str | None = None

    for query in query_candidates:
        try:
            cards = await _fetch_cards(query)
        except Exception:
            continue

        if cards and query_debug is None:
            query_debug = _query_debug_string(query)
        results.extend(cards)

    deduped = _dedupe_cards(results)
    card_ids = [str(_get_field(card, "id", "")) for card in deduped if _get_field(card, "id", "")]

    if not card_ids:
        return deduped, query_debug

    fetched_cards = await asyncio.gather(*(_fetch_card_details(card_id) for card_id in card_ids))
    full_cards: list[Any] = []

    for original, fetched in zip(deduped, fetched_cards, strict=False):
        full_cards.append(fetched or original)

    filtered = _filter_cards_by_set(full_cards, set_id=set_id, collection_name=collection_name)

    if query_debug:
        set_bits: list[str] = []
        if set_id:
            set_bits.append(f"set_id={set_id}")
        if collection_name:
            set_bits.append(f"collection_name={collection_name}")
        if set_bits:
            query_debug = f"{query_debug}&{'&'.join(set_bits)}"
    elif set_id or collection_name:
        set_bits = []
        if set_id:
            set_bits.append(f"set_id={set_id}")
        if collection_name:
            set_bits.append(f"collection_name={collection_name}")
        query_debug = "&".join(set_bits)

    return filtered, query_debug


def _filter_cards_by_set(cards: list[Any], set_id: str | None, collection_name: str | None) -> list[Any]:
    normalized_set_id = _normalize_set_value(set_id)
    normalized_collection = _normalize_set_value(collection_name)

    if not normalized_set_id and not normalized_collection:
        return cards

    filtered: list[Any] = []
    for card in cards:
        set_data = _get_field(card, "set", {})
        card_set_id = _normalize_set_value(_get_field(set_data, "id"))
        card_set_name = _normalize_set_value(_get_field(set_data, "name"))

        if normalized_set_id and normalized_set_id == card_set_id:
            filtered.append(card)
            continue

        if normalized_collection and normalized_collection in card_set_name:
            filtered.append(card)

    return filtered


def _normalize_set_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    return re.sub(r"\s+", " ", text)


def map_to_frontend_fields(payload: Any) -> dict[str, Any]:
    set_data = _get_field(payload, "set", {})
    pricing = _get_field(payload, "pricing", {})
    tcgplayer = _get_field(pricing, "tcgplayer", {})
    normal = _get_field(tcgplayer, "normal", {})

    price = _safe_float(_get_field(normal, "marketPrice"))

    return {
        "id": _get_field(payload, "id", "unknown"),
        "name": _get_field(payload, "name", "Unknown Card"),
        "set_id": _get_field(set_data, "id", ""),
        "collection": _get_field(set_data, "name", "Unknown Set"),
        "collector_number": _get_field(payload, "localId", "Unknown"),
        "image_url": _get_field(payload, "image", ""),
        "market_price_usd": price,
        "market_price_source": "tcgplayer.normal.marketPrice",
        "price_updated_at": _get_field(tcgplayer, "updated"),
    }
