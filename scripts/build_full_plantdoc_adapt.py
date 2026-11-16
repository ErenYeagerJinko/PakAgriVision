"""Build a full-scale field adaptation split from the official PlantDoc training set.

Clones or downloads pratikkayal/PlantDoc-Dataset, maps source classes using
field_meta/class_mapping.csv, excludes any sample in splits/field_test.csv,
and generates splits/field_full_train.csv.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, ROOT, SPLITS_DIR
from src.data.splits import load_class_names

PLANTDOC_REPO_URL = "https://github.com/pratikkayal/PlantDoc-Dataset.git"
RAW_PLANTDOC_DIR = DATA_DIR / "raw" / "plantdoc_full"
OUTPUT_SPLIT_CSV = Path(SPLITS_DIR) / "field_full_train.csv"


def download_plantdoc_repo(dest_dir: Path) -> Path:
    """Clone or pull official PlantDoc dataset repository."""
    dest_dir = Path(dest_dir)
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    
    if (dest_dir / "train").is_dir():
        print(f"PlantDoc repo already exists at {dest_dir}")
        return dest_dir

    print(f"Cloning official PlantDoc dataset from {PLANTDOC_REPO_URL}...")
    cmd = ["git", "clone", "--depth", "1", PLANTDOC_REPO_URL, str(dest_dir)]
    subprocess.run(cmd, check=True)
    return dest_dir


def build_field_full_split() -> pd.DataFrame:
    """Map PlantDoc train images to PlantVillage classes and write split CSV."""
    download_plantdoc_repo(RAW_PLANTDOC_DIR)

    # Load class mapping
    mapping_csv = ROOT / "field_meta" / "class_mapping.csv"
    if not mapping_csv.is_file():
        raise FileNotFoundError(f"Missing class mapping: {mapping_csv}")

    mapping_df = pd.read_csv(mapping_csv)
    mapping = dict(zip(mapping_df["source_class"], mapping_df["plantvillage_class"]))

    # Load official 38 class names
    class_names = load_class_names(Path(SPLITS_DIR) / "class_names.json")
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    # Load field_test to strictly prevent test leakage
    test_csv = Path(SPLITS_DIR) / "field_test.csv"
    test_filenames = set()
    if test_csv.is_file():
        test_df = pd.read_csv(test_csv)
        col = "filename" if "filename" in test_df.columns else "path"
        test_filenames = set(test_df[col].apply(lambda p: Path(p).name))
        print(f"Loaded {len(test_filenames)} test filenames to exclude from training.")

    train_dir = RAW_PLANTDOC_DIR / "train"
    records = []
    skipped_no_match = 0
    skipped_test_leak = 0

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}

    for class_folder in sorted(train_dir.iterdir()):
        if not class_folder.is_dir():
            continue

        source_class = class_folder.name
        pv_class = mapping.get(source_class)

        if not pv_class or pd.isna(pv_class) or pv_class not in class_to_idx:
            skipped_no_match += 1
            continue

        label_idx = class_to_idx[pv_class]

        for img_file in class_folder.iterdir():
            if img_file.suffix not in valid_exts:
                continue

            # Strict anti-leak check
            if img_file.name in test_filenames:
                skipped_test_leak += 1
                continue

            rel_path = img_file.relative_to(ROOT).as_posix()
            records.append({
                "filepath": rel_path,
                "filename": img_file.name,
                "label": pv_class,
                "label_idx": label_idx,
                "source": "PlantDoc-Full",
            })

    df = pd.DataFrame(records)
    print(f"\n================ Full Field Dataset Summary ================")
    print(f"Total verified training images: {len(df)}")
    print(f"Target classes covered: {df['label'].nunique()} / 38")
    print(f"Skipped non-matching classes: {skipped_no_match}")
    print(f"Excluded test-leakage files: {skipped_test_leak}")
    print(f"Saving manifest to: {OUTPUT_SPLIT_CSV}")
    print(f"============================================================\n")

    OUTPUT_SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_SPLIT_CSV, index=False)
    return df


if __name__ == "__main__":
    build_field_full_split()
