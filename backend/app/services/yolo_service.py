from pathlib import Path
import uuid

import cv2

from app.core.model_loader import model
from app.utils.detection_utils import get_severity

PREDICTION_DIR = Path("app/static/predictions")
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


def detect(image_path: str):
    results = model.predict(
        source=image_path,
        conf=0.25,
        save=False,
        verbose=False,
    )

    result = results[0]

    detections = []

    for box in result.boxes:

        confidence = round(float(box.conf), 3)

        # Ignore weak detections
        if confidence < 0.40:
            continue

        detections.append(
            {
                "id": len(detections) + 1,
                "class_name": result.names[int(box.cls)],
                "confidence": confidence,
                "severity": get_severity(confidence),
                "bbox": [
                    round(x, 2)
                    for x in box.xyxy[0].tolist()
                ],
            }
        )

    annotated = result.plot()

    filename = f"{uuid.uuid4().hex}.jpg"

    output_path = PREDICTION_DIR / filename

    cv2.imwrite(str(output_path), annotated)

    return detections, f"/static/predictions/{filename}"