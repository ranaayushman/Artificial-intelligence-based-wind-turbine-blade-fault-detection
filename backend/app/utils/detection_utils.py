from collections import Counter


def get_severity(confidence: float) -> str:
    if confidence >= 0.80:
        return "High"
    elif confidence >= 0.60:
        return "Medium"
    return "Low"


def summarize_detections(detections: list) -> dict:
    counter = Counter(d["class_name"] for d in detections)

    return {
        "total": len(detections),
        "damage": counter.get("damage", 0),
        "dirt": counter.get("dirt", 0),
    }