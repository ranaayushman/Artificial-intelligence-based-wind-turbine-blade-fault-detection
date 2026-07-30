from pathlib import Path
import shutil

from fastapi import APIRouter, File, UploadFile, HTTPException

from app.core.logger import logger
from app.schemas.response import PredictionResponse
from app.services.yolo_service import detect
from app.services.gemini_service import generate_report
from app.utils.detection_utils import summarize_detections

router = APIRouter()

UPLOAD_DIR = Path("app/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
async def predict(file: UploadFile = File(...)):
    """
    Upload an image and receive:
    - YOLO detections
    - Detection summary
    - Annotated image
    - Gemini AI maintenance report
    """

    try:
        logger.info(f"Received file: {file.filename}")

        # Save uploaded image
        image_path = UPLOAD_DIR / file.filename

        with image_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run YOLO detection
        detections, image_url = detect(str(image_path))

        logger.info(f"Detected {len(detections)} objects.")

        # Create summary
        summary = summarize_detections(detections)

        # Generate Gemini report
        report = generate_report(summary, detections)

        return PredictionResponse(
            success=True,
            image={
                "original": f"/uploads/{file.filename}",
                "annotated": image_url,
            },
            summary=summary,
            detections=detections,
            report=report,
        )

    except Exception as e:
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )