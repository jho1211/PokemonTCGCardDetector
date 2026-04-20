import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.services.ocr import get_ocr_service

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
    ocr_service = get_ocr_service()
    if ocr_service.is_available:
        logger.info("PaddleOCR warmed up successfully.")
    else:
        logger.warning("PaddleOCR failed to warm up: %s", ocr_service.last_init_error or ocr_service.last_runtime_error or "unknown error")
