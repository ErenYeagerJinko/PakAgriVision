"""Dataset quality audit, deduplication, and capping CLI script.

Performs:
1. File verification (corrupt / truncated detection via Pillow).
2. Exact duplicate clustering via MD5 byte digest.
3. Near-duplicate clustering via perceptual hashing (pHash Hamming distance <= threshold).
4. Pruning corrupted files and duplicate copies.
5. Capping each class at up to max_per_class (default: 1000) clean images.
6. Generating splits/dedupe_report.csv.

Usage:
    python data_scripts/clean_and_cap_dataset.py [--max-per-class 1000] [--num-workers 8]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import RAW_PV, SPLITS_DIR
from src.data.quality import ImageRecord, QualityChecker, scan_image_files

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clean_and_cap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean dataset for duplicates, corruption, and cap to maximum images per class."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=RAW_PV / "color",
        help="Path to dataset directory (defaults to data/raw/plantvillage/color).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=SPLITS_DIR / "dedupe_report.csv",
        help="Output CSV report path (defaults to splits/dedupe_report.csv).",
    )
    parser.add_argument(
        "--class-names",
        type=Path,
        default=SPLITS_DIR / "class_names.json",
        help="Path to class_names.json.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=1000,
        help="Maximum clean images to retain per class (default: 1000).",
    )
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=4,
        help="pHash Hamming distance threshold for near-duplicates (default: 4).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of worker threads (default: 8).",
    )
    return parser.parse_args()


def write_report_csv(records: List[ImageRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["group_id", "path", "label", "hash_type", "width", "height", "md5", "phash", "notes"]
        )
        for r in records:
            writer.writerow(
                [
                    r.group_id or "",
                    r.path,
                    r.label,
                    r.hash_type,
                    r.width,
                    r.height,
                    r.md5 or "",
                    r.phash or "",
                    r.notes,
                ]
            )


def main() -> int:
    args = parse_args()
    data_dir: Path = args.data_dir
    max_per_class: int = args.max_per_class

    if not data_dir.exists():
        logger.error(f"Dataset directory not found: {data_dir}")
        return 1

    with open(args.class_names, "r", encoding="utf-8") as f:
        official_classes = json.load(f)

    logger.info(f"Targeting {len(official_classes)} classes from {args.class_names}")
    logger.info(f"Scanning image files in {data_dir}...")

    found_files = scan_image_files(data_dir)
    # Filter to only official classes
    found_files = [f for f in found_files if f[2] in official_classes]
    total_found = len(found_files)
    logger.info(f"Discovered {total_found} image files across official classes.")

    checker = QualityChecker(phash_threshold=args.phash_threshold)
    raw_records: List[ImageRecord] = []

    logger.info(f"Verifying file integrity and computing MD5 / pHash across {args.num_workers} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(checker.inspect_file, abs_p, rel_p, lbl)
            for abs_p, rel_p, lbl in found_files
        ]
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            raw_records.append(future.result())
            completed += 1
            if completed % 5000 == 0 or completed == total_found:
                logger.info(f"Progress: {completed}/{total_found} ({completed * 100 // total_found}%)...")

    # Sort deterministically
    raw_records.sort(key=lambda r: r.path)

    logger.info("Clustering exact and near duplicates...")
    clustered_records = checker.run_deduplication(raw_records)

    # Summarize findings
    type_counts = Counter(r.hash_type for r in clustered_records)
    corrupt_count = sum(1 for r in clustered_records if not r.is_valid)

    logger.info("=== SCAN & QUALITY SUMMARY ===")
    logger.info(f"Total images scanned:       {len(clustered_records)}")
    logger.info(f"Corrupt / unreadable:       {corrupt_count}")
    logger.info(f"Unique canonical images:    {type_counts.get('unique', 0)}")
    logger.info(f"Exact duplicates:           {type_counts.get('exact_duplicate', 0)}")
    logger.info(f"Near duplicates:            {type_counts.get('near_duplicate', 0)}")
    logger.info(f"Cross-class duplicates:     {type_counts.get('cross_class_duplicate', 0)}")

    # Now group by class to select up to max_per_class clean images
    records_by_class: Dict[str, List[ImageRecord]] = defaultdict(list)
    for r in clustered_records:
        records_by_class[r.label].append(r)

    to_keep: List[ImageRecord] = []
    to_delete: List[ImageRecord] = []

    for class_name in official_classes:
        class_records = records_by_class.get(class_name, [])
        # Separate valid clean records from corrupt/duplicate records
        valid_unique = [r for r in class_records if r.is_valid and r.hash_type == "unique"]
        duplicates_or_corrupt = [r for r in class_records if not r.is_valid or r.hash_type != "unique"]

        # All duplicates or corrupt files are marked for removal
        to_delete.extend(duplicates_or_corrupt)

        # Cap valid unique images at max_per_class
        if len(valid_unique) > max_per_class:
            selected = valid_unique[:max_per_class]
            excess = valid_unique[max_per_class:]
            to_keep.extend(selected)
            for ex in excess:
                ex.notes = f"Excess beyond {max_per_class} cap; pruned"
            to_delete.extend(excess)
        else:
            to_keep.extend(valid_unique)

    logger.info(f"\nFinal selection: Retaining {len(to_keep)} clean unique images across {len(official_classes)} classes.")
    logger.info(f"Pruning {len(to_delete)} duplicate, corrupt, or excess files from disk...")

    deleted_count = 0
    for r in to_delete:
        try:
            if r.absolute_path.exists():
                r.absolute_path.unlink()
                deleted_count += 1
        except Exception as exc:
            logger.warning(f"Could not remove {r.absolute_path}: {exc}")

    logger.info(f"Successfully pruned {deleted_count} files from disk.")

    # Write dedupe report for all kept clean files
    write_report_csv(to_keep, args.output_csv)
    logger.info(f"Clean deduplication manifest written to: {args.output_csv}")

    # Log per-class summary of kept images
    kept_by_class = Counter(r.label for r in to_keep)
    logger.info("\n=== PER-CLASS FINAL COUNTS ===")
    for cls in official_classes:
        logger.info(f"  {cls}: {kept_by_class.get(cls, 0)} images")

    return 0


if __name__ == "__main__":
    sys.exit(main())
