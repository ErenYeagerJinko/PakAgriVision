"""Treatment coverage, immutable retrieval, confidence boundaries and data failures."""

import json
from pathlib import Path

import pytest

from app.treatment_advisor import build_advisory, CONFIDENCE_THRESHOLD, WITHHELD
from knowledge_base.kb_service import KnowledgeBaseError, TreatmentKB, ROOT

CLASSES = json.loads((ROOT / "splits/class_names.json").read_text())


@pytest.fixture(scope="module")
def kb():
    return TreatmentKB()


@pytest.mark.parametrize("name", CLASSES)
def test_all_38_classes_have_sourced_advice(kb, name):
    record = kb.get_treatment(name)
    assert record["sources"] and record["reviewed_by"] and record["last_reviewed"]
    assert all(record[key].strip() for key in ["chemical_treatment", "organic_treatment",
                                               "immediate_action", "prevention", "severity_guidance"])
    advisory = build_advisory(kb, name, 95)
    assert advisory["disclaimer"] == kb.disclaimer
    assert advisory["chemical_advice_allowed"] is not record["is_healthy"]
    assert "product label" in advisory["dosage_notice"]


def test_retrieval_is_an_independent_copy(kb):
    record = kb.get_treatment(CLASSES[0])
    record["sources"].clear()
    record["chemical_treatment"] = "changed"
    assert kb.get_treatment(CLASSES[0])["sources"]
    assert kb.get_treatment(CLASSES[0])["chemical_treatment"] != "changed"


@pytest.mark.parametrize("name", ["unknown", "Tomato___late_blight", "disclaimer", None])
def test_no_substitute_for_unknown_class(kb, name):
    with pytest.raises(KeyError):
        kb.get_treatment(name)


@pytest.mark.parametrize("confidence,low", [(0, True), (59.99999, True), (60, False), (100, False)])
def test_threshold_boundary(kb, confidence, low):
    advisory = build_advisory(kb, "Tomato___Late_blight", confidence)
    assert advisory["confidence_threshold"] == CONFIDENCE_THRESHOLD == 60
    assert advisory["low_confidence"] is low
    assert bool(advisory["warning"]) is low
    assert advisory["chemical_advice_allowed"] is not low
    if low:
        assert advisory["status"] == "uncertain"
        assert advisory["treatment"]["chemical_treatment"] == WITHHELD
        assert advisory["treatment"]["organic_treatment"] == WITHHELD
        assert "spray" not in advisory["treatment"]["immediate_action"].lower()


@pytest.mark.parametrize("name", CLASSES)
def test_low_confidence_never_leaks_product_advice(kb, name):
    advisory = build_advisory(kb, name, 45)
    assert advisory["status"] == "uncertain"
    assert not advisory["chemical_advice_allowed"]
    assert advisory["treatment"]["chemical_treatment"] == WITHHELD
    assert advisory["treatment"]["organic_treatment"] == WITHHELD
    assert "verify" in advisory["warning"].lower()
    # A gated lookup must not mutate subsequent confident retrievals.
    assert kb.get_treatment(name)["chemical_treatment"] != WITHHELD


@pytest.mark.parametrize("confidence", [-1, 100.01, float("nan"), float("inf"), "0.9", None, True])
def test_invalid_confidence_rejected(kb, confidence):
    with pytest.raises(ValueError, match="percentage"):
        build_advisory(kb, CLASSES[0], confidence)


def test_confident_healthy_prediction_does_not_prescribe(kb):
    advisory = build_advisory(kb, "Tomato___healthy", 90)
    assert advisory["status"] == "healthy"
    assert not advisory["chemical_advice_allowed"]
    assert "No disease-control" in advisory["treatment"]["chemical_treatment"]


def write_kb(tmp_path, mutation):
    data = json.loads((ROOT / "knowledge_base/treatments.json").read_text())
    mutation(data)
    path = tmp_path / "treatments.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("mutation", [
    lambda d: d.pop(CLASSES[0]),
    lambda d: d.update(unknown=d[CLASSES[0]]),
    lambda d: d.pop("disclaimer"),
    lambda d: d[CLASSES[0]].pop("prevention"),
    lambda d: d[CLASSES[0]].update(last_reviewed="not-a-date"),
    lambda d: d[CLASSES[0]].update(is_healthy=not d[CLASSES[0]]["is_healthy"]),
    lambda d: d[CLASSES[0]].update(sources=["no source URL"]),
    lambda d: d[CLASSES[0]].update(extra="unexpected field"),
    lambda d: d[CLASSES[0]].update(prevention="   "),
])
def test_bad_records_fail_closed(tmp_path, mutation):
    with pytest.raises(KnowledgeBaseError):
        TreatmentKB(kb_path=write_kb(tmp_path, mutation))


def test_missing_and_malformed_files(tmp_path):
    path = tmp_path / "missing.json"
    with pytest.raises(KnowledgeBaseError):
        TreatmentKB(kb_path=path)
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(KnowledgeBaseError):
        TreatmentKB(kb_path=path)


def test_invalid_schema_rejected(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text('{"type": "not-a-json-type"}')
    with pytest.raises(KnowledgeBaseError):
        TreatmentKB(schema_path=path)
