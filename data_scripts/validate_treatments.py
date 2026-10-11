"""Validate knowledge_base/treatments.json against schema and class names."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from jsonschema import Draft7Validator

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "knowledge_base"
SPLITS_DIR = ROOT / "splits"
TREATMENTS_PATH = KB_DIR / "treatments.json"
SCHEMA_PATH = KB_DIR / "schema.json"
CLASS_NAMES_PATH = SPLITS_DIR / "class_names.json"

META_KEYS = {"disclaimer"}


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if treatments.json does not cover every class in class_names.json.",
    )
    args = parser.parse_args()

    class_names = load_json(CLASS_NAMES_PATH)
    if not isinstance(class_names, list) or not class_names:
        print(f"ERROR: {CLASS_NAMES_PATH} must be a non-empty list.")
        return 1
    if len(class_names) != len(set(class_names)):
        print("ERROR: duplicate class names in class_names.json.")
        return 1

    schema = load_json(SCHEMA_PATH)
    Draft7Validator.check_schema(schema)
    entry_validator = Draft7Validator(schema)

    data = load_json(TREATMENTS_PATH)
    if not isinstance(data, dict):
        print("ERROR: treatments.json must be a JSON object.")
        return 1

    disclaimer = data.get("disclaimer")
    if not isinstance(disclaimer, str) or not disclaimer.strip():
        print("ERROR: treatments.json must include a non-empty top-level 'disclaimer' string.")
        return 1

    entries = {k: v for k, v in data.items() if k not in META_KEYS}
    expected = set(class_names)
    found = set(entries)
    missing = sorted(expected - found)
    unexpected = sorted(found - expected)

    errors = 0

    if unexpected:
        errors += 1
        print(f"ERROR: unexpected treatment keys ({len(unexpected)}):")
        for key in unexpected:
            print(f"  - {key}")

    for key, entry in sorted(entries.items()):
        if not isinstance(entry, dict):
            errors += 1
            print(f"ERROR: {key}: entry must be an object.")
            continue
        for err in sorted(entry_validator.iter_errors(entry), key=lambda e: list(e.path)):
            errors += 1
            path = ".".join(str(p) for p in err.path) or "(root)"
            print(f"ERROR: {key}.{path}: {err.message}")

        # Extra content checks beyond jsonschema
        for field in (
            "crop",
            "disease",
            "cause",
            "symptoms",
            "chemical_treatment",
            "organic_treatment",
            "immediate_action",
            "prevention",
            "severity_guidance",
            "reviewed_by",
        ):
            value = entry.get(field)
            if isinstance(value, str) and not value.strip():
                errors += 1
                print(f"ERROR: {key}.{field}: must be a non-empty string.")

        sources = entry.get("sources")
        if isinstance(sources, list):
            if any(not isinstance(s, str) or not s.strip() for s in sources):
                errors += 1
                print(f"ERROR: {key}.sources: every item must be a non-empty string.")

        if entry.get("is_healthy") is True:
            chem = entry.get("chemical_treatment", "")
            if isinstance(chem, str) and "none needed" not in chem.lower():
                print(
                    f"WARNING: {key}: healthy class should set chemical_treatment to "
                    "'none needed' (or similar)."
                )

    crops = Counter()
    for key, entry in entries.items():
        if isinstance(entry, dict) and isinstance(entry.get("crop"), str):
            crops[entry["crop"]] += 1

    print("--- Treatment KB summary ---")
    print(f"Disclaimer: present ({len(disclaimer)} chars)")
    print(f"Expected classes: {len(expected)}")
    print(f"Present entries:  {len(found)}")
    print(f"Missing entries:  {len(missing)}")
    if crops:
        print("Per-crop counts:")
        for crop, count in sorted(crops.items()):
            print(f"  {crop}: {count}")
    if missing:
        print("Missing class keys:")
        for key in missing:
            print(f"  - {key}")

    if errors:
        print(f"FAILED: {errors} error(s).")
        return 1

    if missing:
        msg = f"INCOMPLETE: {len(missing)} class(es) still need entries."
        if args.require_complete:
            print(f"FAILED: {msg}")
            return 1
        print(f"WARNING: {msg} Re-run with --require-complete before final submission.")
        return 0

    print("OK: all class keys present and every entry validates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
