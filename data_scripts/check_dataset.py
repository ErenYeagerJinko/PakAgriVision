"""Dataset quality audit and deduplication CLI script.

Usage:
    python data_scripts/check_dataset.py [--data-dir PATH] [--output-csv PATH] [--phash-threshold INT]

Outputs:
    splits/dedupe_report.csv
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import List

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, RAW_PV, SPLITS_DIR
from src.data.quality import ImageRecord, QualityChecker, scan_image_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan dataset images, detect corrupted files, and cluster exact/near duplicates."
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
        "--phash-threshold",
        type=int,
        default=4,
        help="Perceptual hash Hamming distance threshold for near-duplicates (default: 4).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel worker threads for image verification (default: 8).",
    )
    return parser.parse_args()


def write_report_csv(records: List[ImageRecord], output_path: Path) -> None:
    """Write deduplication report to CSV."""
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

    # Support fallback to data/raw/plantvillage if color/ is directly unpacked
    if not data_dir.exists() and (RAW_PV).exists():
        data_dir = RAW_PV

    if not data_dir.exists():
        print(f"ERROR: Dataset directory not found: {data_dir}")
        print("Please download and extract PlantVillage into data/raw/plantvillage/color/ first.")
        print("Refer to data/README.md for download instructions.")
        return 1

    print(f"Scanning image files in: {data_dir}")
    try:
        found_files = scan_image_files(data_dir)
    except Exception as exc:
        print(f"ERROR scanning directory: {exc}")
        return 1

    total_images = len(found_files)
    print(f"Found {total_images} image files. Starting integrity verification and hashing...")

    if total_images == 0:
        print("WARNING: No image files found to inspect.")
        return 0

    checker = QualityChecker(phash_threshold=args.phash_threshold)
    raw_records: List[ImageRecord] = []

    # Parallel verification and hashing
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(checker.inspect_file, abs_path, rel_path, label)
            for abs_path, rel_path, label in found_files
        ]
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            raw_records.append(future.result())
            completed += 1
            if completed % 5000 == 0 or completed == total_images:
                print(f"Progress: {completed}/{total_images} images inspected ({completed * 100 // total_images}%)...")

    # Sort deterministically by relative path
    raw_records.sort(key=lambda r: r.path)

    print("Clustering exact, near, and cross-class duplicates via perceptual hashing...")
    clustered_records = checker.run_deduplication(raw_records)

    # Statistics summary
    type_counts = Counter(r.hash_type for r in clustered_records)
    corrupt_count = sum(1 for r in clustered_records if not r.is_valid)

    print("\n--- Data Quality & Deduplication Summary ---")
    print(f"Total images scanned:     {len(clustered_records)}")
    print(f"Corrupt / unreadable:     {corrupt_count}")
    print(f"Unique clean images:      {type_counts.get('unique', 0)}")
    print(f"Exact duplicate copies:   {type_counts.get('exact_duplicate', 0)}")
    print(f"Near duplicate copies:    {type_counts.get('near_duplicate', 0)}")
    print(f"Cross-class duplicates:   {type_counts.get('cross_class_duplicate', 0)}")

    write_report_csv(clustered_records, args.output_csv)
    print(f"\nDeduplication manifest written to: {args.output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
