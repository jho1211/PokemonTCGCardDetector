from __future__ import annotations
from paddleocr import PaddleOCR
import numpy as np
import re
import json

from app.models.schemas import Card

def load_abbreviation_map() -> dict[str, str]:
    with open("templates/set_symbols/abbreviation_map.json", "r") as f:
        return json.load(f)

OCR = PaddleOCR(
            lang="en",
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
SYMBOL_MAP = load_abbreviation_map()


def parse_set_text(ocr_texts: list[str]) -> str | None:
    for text in ocr_texts:
        # remove spaces and convert to uppercase for matching
        text = text.replace(" ", "").upper()
        if text in SYMBOL_MAP:
            return SYMBOL_MAP[text]
    return None


def parse_collector_number(ocr_texts: list[str]) -> str | None:
    # if "e.g. 123/456" is in the OCR results, return it as the collector number
    for text in ocr_texts:
        if "/" in text:
            return text
    return None


def parse_card_name(ocr_texts: list[str]) -> str | None:
    if not ocr_texts:
        return None

    # use regex to find the first word that starts with a capital letter and is at least 3 characters long
    for text in ocr_texts:
        match = re.search(r'\b[A-Z][a-z]{2,}\b', text)
        if match:
            return match.group()
    return None


async def predict_card(card_regions: list[np.ndarray]) -> Card:
    card = Card()
    for region_name, region_crop in card_regions.items():
        result = OCR.predict(region_crop)
        for res in result:
            res_json = res.json['res']
            res_texts = res_json.get("rec_texts", "")

            if region_name == "number":
                card.collector_number = parse_collector_number(res_texts)
                card.set_id = parse_set_text(res_texts) # if this fails, we will try template matching later
            else:
                card.name = parse_card_name(res_texts)
    
    if card.collector_number and card.set_id:
        card.id = f"{card.set_id}-{card.collector_number}"

    return card