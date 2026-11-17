"""Validated, read-only retrieval of the curated treatment knowledge base."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator, FormatChecker
from jsonschema.exceptions import SchemaError

ROOT = Path(__file__).resolve().parents[1]


class KnowledgeBaseError(ValueError):
    """Knowledge-base files cannot safely support treatment retrieval."""


class TreatmentKB:
    """Validate once at startup; return independent copies of exact class records."""

    def __init__(
        self,
        kb_path: str | Path = ROOT / "knowledge_base/treatments.json",
        schema_path: str | Path = ROOT / "knowledge_base/schema.json",
        class_names_path: str | Path = ROOT / "splits/class_names.json",
    ) -> None:
        try:
            data = json.loads(Path(kb_path).read_text(encoding="utf-8"))
            schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
            classes = json.loads(Path(class_names_path).read_text(encoding="utf-8"))
            Draft7Validator.check_schema(schema)
            if not isinstance(classes, list) or not classes or any(
                not isinstance(c, str) or not c for c in classes
            ) or len(set(classes)) != len(classes):
                raise ValueError("Class names must be a non-empty, unique list of strings")
            if not isinstance(data, dict):
                raise ValueError("Treatment data must be an object")
            if set(data) - {"disclaimer"} != set(classes):
                raise ValueError("Treatment keys must exactly match the class manifest")
            if not isinstance(data.get("disclaimer"), str) or not data["disclaimer"].strip():
                raise ValueError("A decision-support disclaimer is required")
            validator = Draft7Validator(schema, format_checker=FormatChecker())
            for name in classes:
                errors = sorted(validator.iter_errors(data[name]), key=lambda e: str(e.path))
                if errors:
                    raise ValueError(f"{name}: {errors[0].message}")
                record = data[name]
                if any(isinstance(value, str) and not value.strip() for value in record.values()):
                    raise ValueError(f"{name}: text fields must not be blank")
                if record["is_healthy"] != (name.split("___", 1)[-1].lower() == "healthy"):
                    raise ValueError(f"{name}: healthy flag disagrees with class name")
                for source in record["sources"]:
                    if "https://" not in source and "http://" not in source:
                        raise ValueError(f"{name}: source URL is missing")
            self._records = {name: data[name] for name in classes}
            self.class_names = tuple(classes)
            self.disclaimer = data["disclaimer"]
        except (OSError, ValueError, TypeError, KeyError, SchemaError) as exc:
            raise KnowledgeBaseError(f"Cannot load treatment knowledge base: {exc}") from exc

    def get_treatment(self, class_name: str) -> dict[str, Any]:
        """Retrieve an exact model class; never substitute another disease's treatment."""
        if not isinstance(class_name, str) or class_name not in self._records:
            raise KeyError(f"Unknown treatment class: {class_name!r}")
        return deepcopy(self._records[class_name])

    def __len__(self) -> int:
        return len(self._records)
