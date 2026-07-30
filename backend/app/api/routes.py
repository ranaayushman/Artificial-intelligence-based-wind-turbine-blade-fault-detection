from pathlib import Path
import shutil

from fastapi import APIRouter, File, UploadFile

from app.core.logger import logger
from app.schemas.response import PredictionResponse
from app.services.yolo_service import detect
from app.utils.detection_utils import summarize_detections

router = APIRouter()

UPLOAD_DIR = Path("app/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(file: UploadFile = File(...)):

    logger.info(f"Received file: {file.filename}")

    image_path = UPLOAD_DIR / file.filename

    with image_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    detections, image_url = detect(str(image_path))

    logger.info(f"Detected {len(detections)} objects.")

    summary = summarize_detections(detections)

    return {
        "success": True,
        "image": {
            "original": f"/uploads/{file.filename}",
            "annotated": image_url,
        },
        "summary": summary,
        "detections": detections,
        "report": None,
    }