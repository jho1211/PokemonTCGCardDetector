import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.services.detector import get_card_detector
from app.services.ocr import get_ocr_service
from app.services.symbol_matcher import get_symbol_matcher

app = FastAPI(title="Pokemon TCG Identifier API", version="0.1.0")
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def warm_ocr_service() -> None:
    detector = get_card_detector()
    if detector.is_available:
        logger.info("YOLO detector warmed up successfully.")
    else:
        logger.warning("YOLO detector unavailable: %s", detector.last_init_error or detector.last_runtime_error or "unknown error")

    ocr_service = get_ocr_service()
    if ocr_service.is_available:
        logger.info("PaddleOCR warmed up successfully.")
    else:
        logger.warning("PaddleOCR failed to warm up: %s", ocr_service.last_init_error or ocr_service.last_runtime_error or "unknown error")

    symbol_matcher = get_symbol_matcher()
    if symbol_matcher.is_available:
        logger.info("Set symbol matcher loaded templates successfully.")
    else:
        logger.info("Set symbol matcher disabled: %s", symbol_matcher.last_error or "no templates")
