from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import Card
from app.services.api import identify_cards

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/identify-card", response_model=list[Card])
async def identify_card(image: UploadFile = File(...)) -> list[Card]:
    content_type = image.content_type or ""
    if content_type not in {"image/jpeg", "image/jpg", "image/png", "image/tiff", "image/tif", "image/x-tiff"}:
        raise HTTPException(status_code=400, detail="Supported file types: jpg, jpeg, png, tif, tiff")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        return await identify_cards(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
