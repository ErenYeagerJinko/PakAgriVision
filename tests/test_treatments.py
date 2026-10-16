"""Unit tests for knowledge_base/treatments.json and schema."""

from pathlib import Path
import json
import pytest
from jsonschema import Draft7Validator

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "knowledge_base"
SPLITS_DIR = ROOT / "splits"
TREATMENTS_PATH = KB_DIR / "treatments.json"
SCHEMA_PATH = KB_DIR / "schema.json"
CLASS_NAMES_PATH = SPLITS_DIR / "class_names.json"


def test_schema_is_valid_draft7():
    """Verify that schema.json itself is a valid JSON schema."""
    with SCHEMA_PATH.open(encoding="utf-8") as f:
        schema = json.load(f)
    Draft7Validator.check_schema(schema)


def test_treatments_covers_all_classes():
    """Verify all 38 classes are present with zero missing and valid schema."""
    with CLASS_NAMES_PATH.open(encoding="utf-8") as f:
        class_names = json.load(f)
    with SCHEMA_PATH.open(encoding="utf-8") as f:
        schema = json.load(f)
    with TREATMENTS_PATH.open(encoding="utf-8") as f:
        treatments = json.load(f)

    assert "disclaimer" in treatments, "disclaimer key missing"
    assert len(treatments["disclaimer"].strip()) >= 20, "disclaimer too short"

    validator = Draft7Validator(schema)
    missing = []
    invalid = []

    for name in class_names:
        if name not in treatments:
            missing.append(name)
        else:
            entry = treatments[name]
            errors = list(validator.iter_errors(entry))
            if errors:
                invalid.append((name, [e.message for e in errors]))

    assert missing == [], f"Missing treatments for: {missing}"
    assert invalid == [], f"Validation errors: {invalid}"
    assert len(class_names) == 38, "Expected exactly 38 classes"
