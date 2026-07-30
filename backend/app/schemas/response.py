from typing import List, Optional

from pydantic import BaseModel


class Detection(BaseModel):
    id: int
    class_name: str
    confidence: float
    severity: str
    bbox: List[float]


class ImageResponse(BaseModel):
    original: str
    annotated: str


class Summary(BaseModel):
    total: int
    damage: int
    dirt: int


class PredictionResponse(BaseModel):
    success: bool
    image: ImageResponse
    summary: Summary
    detections: List[Detection]
    report: Optional[dict] = None