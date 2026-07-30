from typing import List

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


class Report(BaseModel):
    overall_condition: str
    risk_level: str
    inspection_summary: str
    maintenance_recommendations: List[str]


class PredictionResponse(BaseModel):
    success: bool
    image: ImageResponse
    summary: Summary
    detections: List[Detection]
    report: Report