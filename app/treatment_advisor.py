"""Confidence-aware advisory view models shared by Flask and report utilities.

Confidence is expressed as a percentage (0..100), not a fraction. It is not a
measure of disease severity or a guarantee of diagnostic correctness.
"""

from __future__ import annotations

import math
from typing import Any

from knowledge_base.kb_service import TreatmentKB

CONFIDENCE_THRESHOLD = 60.0
AMBIGUOUS_WARNING = (
    "Ambiguous symptoms detected. Verify in person before applying chemical treatments. "
    "Retake a clear photo and consult a local Agriculture Extension Officer."
)
AMBIGUOUS_WARNING_UR = (
    "علامات غیر واضح ہیں۔ کیمیائی ادویات استعمال کرنے سے پہلے موقع پر تصدیق کریں۔ "
    "صاف تصویر دوبارہ لیں اور مقامی زرعی توسیعی افسر سے مشورہ کریں۔"
)
DOSAGE_NOTICE = (
    "No universal dose is specified by this service. Use only products registered for "
    "the crop and disease; follow the local product label for concentration, rate, "
    "protective equipment, re-entry interval and pre-harvest interval. Consult a local agronomist."
)
WITHHELD = "Treatment withheld until symptoms have been verified by a local agronomist."
SAFE_ACTION = "Retake a clear leaf photo; inspect nearby plants and seek an in-person diagnosis."
SAFE_PREVENTION = "Continue routine scouting, tool sanitation and avoidance of prolonged leaf wetness."


def build_advisory(kb: TreatmentKB, class_name: str, confidence: float) -> dict[str, Any]:
    """Return panel-ready data with no actionable product advice below the threshold.

    Organic products and immediate-action text may also recommend spraying, so
    these fields are gated along with chemical control. Raw product advice is
    never included in the low-confidence API response.
    """
    if isinstance(confidence, bool) or not isinstance(confidence, (float, int)):
        raise ValueError("Confidence must be a percentage between 0 and 100")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0 <= confidence <= 100:
        raise ValueError("Confidence must be a percentage between 0 and 100")
    record = kb.get_treatment(class_name)
    low = confidence < CONFIDENCE_THRESHOLD
    healthy = record["is_healthy"]
    status = "uncertain" if low else "healthy" if healthy else "disease_detected"
    if low:
        record.update(
            chemical_treatment=WITHHELD,
            organic_treatment=WITHHELD,
            immediate_action=SAFE_ACTION,
            prevention=SAFE_PREVENTION,
            severity_guidance="Severity cannot be determined from model confidence. Seek an in-person assessment.",
        )
    elif healthy:
        record.update(
            chemical_treatment="No disease-control chemicals indicated by this prediction. Continue monitoring.",
            organic_treatment="No disease-control products indicated by this prediction. Continue cultural best practices.",
        )
    return {
        "status": status,
        "low_confidence": low,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "chemical_advice_allowed": not low and not healthy,
        "warning": AMBIGUOUS_WARNING if low else "",
        "warning_ur": AMBIGUOUS_WARNING_UR if low else "",
        "disclaimer": kb.disclaimer,
        "dosage_notice": DOSAGE_NOTICE,
        "knowledge_language": "en",
        "treatment": record,
    }
