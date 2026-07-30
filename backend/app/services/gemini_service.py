import json
import google.generativeai as genai

from app.core.config import settings

genai.configure(api_key=settings.GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-3.6-flash")


def generate_report(summary: dict, detections: list) -> dict:
    """
    Generate an AI maintenance report from YOLO detections.

    Args:
        summary: Detection summary
        detections: List of detected defects

    Returns:
        Structured report dictionary
    """

    prompt = f"""
You are an expert wind turbine maintenance engineer.

The following blade inspection was produced by an AI object detection model.

Detection Summary:
{json.dumps(summary, indent=2)}

Detections:
{json.dumps(detections, indent=2)}

Return ONLY valid JSON.

The JSON must follow this schema exactly:

{{
    "overall_condition": "",
    "risk_level": "",
    "inspection_summary": "",
    "maintenance_recommendations": [
        "",
        "",
        ""
    ]
}}

Do not include markdown.
Do not include explanations.
Return JSON only.
"""

    response = model.generate_content(prompt)

    text = response.text.strip()

    try:
        return json.loads(text)

    except Exception:

        return {
            "overall_condition": "Unknown",
            "risk_level": "Unknown",
            "inspection_summary": text,
            "maintenance_recommendations": [],
        }