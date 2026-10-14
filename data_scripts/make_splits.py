"""Generate stratified train/validation/test splits and compute class weights.

Usage:
    python data_scripts/make_splits.py [--dedupe-report PATH] [--output-dir PATH] [--seed INT]

Outputs:
    splits/train.csv
    splits/val.csv
    splits/test.csv
    splits/class_weights.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import RAW_PV, SEED, SPLITS_DIR
from src.data.quality import scan_image_files
from src.data.splits import (
    compute_class_weights,
    create_stratified_group_splits,
    get_class_mappings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create stratified, group-aware dataset splits and compute training class weights."
    )
    parser.add_argument(
        "--dedupe-report",
        type=Path,
        default=SPLITS_DIR / "dedupe_report.csv",
        help="Path to deduplication report (default: splits/dedupe_report.csv).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=RAW_PV / "color",
        help="Raw dataset directory if dedupe report is absent (default: data/raw/plantvillage/color).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SPLITS_DIR,
        help="Directory to save generated CSV splits and class weights (default: splits/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for reproducibility (default: {SEED}).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Proportion of training set (default: 0.70).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of validation set (default: 0.15).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proportion of test set (default: 0.15).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names, class_to_idx = get_class_mappings(output_dir / "class_names.json")

    # Ingest data from dedupe report or raw directory
    if args.dedupe_report.exists():
        print(f"Loading deduplication report from: {args.dedupe_report}")
        df = pd.read_csv(args.dedupe_report)
    elif args.data_dir.exists():
        print(f"Dedupe report not found. Scanning raw image files directly in: {args.data_dir}")
        files = scan_image_files(args.data_dir)
        df = pd.DataFrame(
            [{"path": rel_path, "label": label, "group_id": rel_path} for _, rel_path, label in files]
        )
    else:
        print(f"ERROR: Neither dedupe report ({args.dedupe_report}) nor raw directory ({args.data_dir}) exists.")
        print("Please run data_scripts/check_dataset.py after downloading the dataset.")
        return 1

    if df.empty:
        print("ERROR: Dataset dataframe is empty.")
        return 1

    print(f"Dataset contains {len(df)} entries. Creating group-aware stratified splits...")
    try:
        train_df, val_df, test_df = create_stratified_group_splits(
            df=df,
            class_to_idx=class_to_idx,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    except Exception as exc:
        print(f"ERROR partitioning dataset: {exc}")
        return 1

    # Save splits
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nWrote splits:")
    print(f"  - Train:      {len(train_df):>6} rows -> {train_path}")
    print(f"  - Validation: {len(val_df):>6} rows -> {val_path}")
    print(f"  - Test:       {len(test_df):>6} rows -> {test_path}")
    print(f"  - Total:      {len(train_df) + len(val_df) + len(test_df):>6} rows")

    # Compute class weights strictly from the training split
    ordered_weights, weights_dict = compute_class_weights(train_df, class_to_idx)
    weights_path = output_dir / "class_weights.json"

    # Save format with both ordered weights list and per-class mapping
    weights_payload = {
        "ordered_weights": ordered_weights,
        "class_weights": weights_dict,
        "metadata": {
            "num_classes": len(class_to_idx),
            "train_samples": len(train_df),
            "formula": "w_c = N_train / (K * n_c)",
            "seed": args.seed,
        },
    }
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(weights_payload, f, indent=2)

    print(f"\nCalculated class weights for {len(ordered_weights)} classes -> {weights_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
