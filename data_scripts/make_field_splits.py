"""Split field images into field_test (~2/3) and field_adapt (~1/3), stratified by class.

field_test.csv is never used for training, tuning or threshold selection.
Only field_adapt.csv may be used for the fine-tuning mitigation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "splits"
FIELD_DIR = ROOT / "data" / "raw" / "field"
LABELS_PATH = ROOT / "field_meta" / "field_labels.csv"
CLASS_NAMES_PATH = SPLITS_DIR / "class_names.json"
TEST_PATH = SPLITS_DIR / "field_test.csv"
ADAPT_PATH = SPLITS_DIR / "field_adapt.csv"
STATS_PATH = SPLITS_DIR / "field_split_stats.csv"

SEED = 42
ADAPT_FRACTION = 1 / 3
SPLIT_COLUMNS = ["filepath", "filename", "label", "label_idx", "source"]


def n_adapt_for(n: int) -> int:
    """Images per class for field_adapt; keeps at least one in each split when n >= 2."""
    if n < 2:
        return 0
    return min(max(1, round(n * ADAPT_FRACTION)), n - 1)


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    with CLASS_NAMES_PATH.open(encoding="utf-8") as f:
        class_names = json.load(f)
    class_idx = {name: i for i, name in enumerate(class_names)}

    with LABELS_PATH.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    errors = 0
    unknown = sorted({r["label"] for r in rows if r["label"] not in class_idx})
    for label in unknown:
        errors += 1
        print(f"ERROR: unknown label '{label}' (not in class_names.json).")
    dupes = sorted(k for k, v in Counter(r["filename"] for r in rows).items() if v > 1)
    for name in dupes:
        errors += 1
        print(f"ERROR: duplicate filename '{name}' in field_labels.csv.")
    if errors:
        print(f"FAILED: {errors} error(s).")
        return 1

    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["label"]].append(r)

    test_rows: list[dict] = []
    adapt_rows: list[dict] = []
    stats_rows: list[dict] = []
    for label in sorted(by_class, key=class_idx.get):
        items = sorted(by_class[label], key=lambda r: r["filename"])
        # Seed per class so adding images to one class never reshuffles another.
        random.Random(f"{SEED}-{label}").shuffle(items)
        k = n_adapt_for(len(items))
        for i, r in enumerate(items):
            out = {
                "filepath": f"data/raw/field/{label}/{r['filename']}",
                "filename": r["filename"],
                "label": label,
                "label_idx": class_idx[label],
                "source": r["source"],
            }
            (adapt_rows if i < k else test_rows).append(out)
        stats_rows.append(
            {
                "label": label,
                "label_idx": class_idx[label],
                "total": len(items),
                "field_test": len(items) - k,
                "field_adapt": k,
            }
        )

    test_rows.sort(key=lambda r: (r["label_idx"], r["filename"]))
    adapt_rows.sort(key=lambda r: (r["label_idx"], r["filename"]))
    stats_rows.append(
        {
            "label": "TOTAL",
            "label_idx": "",
            "total": len(rows),
            "field_test": len(test_rows),
            "field_adapt": len(adapt_rows),
        }
    )

    write_csv(TEST_PATH, SPLIT_COLUMNS, test_rows)
    write_csv(ADAPT_PATH, SPLIT_COLUMNS, adapt_rows)
    write_csv(STATS_PATH, ["label", "label_idx", "total", "field_test", "field_adapt"], stats_rows)

    print("--- Field split summary ---")
    print(f"Images:      {len(rows)}")
    print(f"Classes:     {len(by_class)} of {len(class_names)}")
    print(f"field_test:  {len(test_rows)}")
    print(f"field_adapt: {len(adapt_rows)}")
    print("Per-source counts (test / adapt):")
    test_src = Counter(r["source"] for r in test_rows)
    adapt_src = Counter(r["source"] for r in adapt_rows)
    for src in sorted(set(test_src) | set(adapt_src)):
        print(f"  {src}: {test_src[src]} / {adapt_src[src]}")
    single = [s["label"] for s in stats_rows[:-1] if s["total"] < 2]
    if single:
        print(f"WARNING: {len(single)} class(es) with one image go to field_test only:")
        for label in single:
            print(f"  - {label}")
    missing_classes = [c for c in class_names if c not in by_class]
    if missing_classes:
        print(f"Classes with no field images: {len(missing_classes)}")

    if FIELD_DIR.is_dir():
        missing = [r["filepath"] for r in test_rows + adapt_rows if not (ROOT / r["filepath"]).is_file()]
        if missing:
            print(f"WARNING: {len(missing)} listed image(s) not found under data/raw/field/.")
    else:
        print("NOTE: data/raw/field/ not found; skipped image existence check.")

    print(f"Wrote {TEST_PATH.relative_to(ROOT).as_posix()}, {ADAPT_PATH.relative_to(ROOT).as_posix()}, "
          f"{STATS_PATH.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
