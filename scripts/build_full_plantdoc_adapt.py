"""Build a full-scale field adaptation split from the official PlantDoc training set.

Clones or downloads pratikkayal/PlantDoc-Dataset, maps source classes using
field_meta/class_mapping.csv, rigorously filters corrupted files, removes exact
duplicates via MD5 hashing, prevents any test leakage against splits/field_test.csv,
and generates a pristine splits/field_full_train.csv.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

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


def compute_file_md5(file_path: Path) -> str:
    """Compute MD5 hash for exact image deduplication."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_image_valid(file_path: Path) -> bool:
    """Rigorously verify image can be opened, decoded, and converted to RGB."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        with Image.open(file_path) as img:
            img.convert("RGB")
        return True
    except Exception:
        return False


def build_field_full_split() -> pd.DataFrame:
    """Map PlantDoc train images to PlantVillage classes with full data cleaning."""
    download_plantdoc_repo(RAW_PLANTDOC_DIR)

    mapping_csv = ROOT / "field_meta" / "class_mapping.csv"
    if not mapping_csv.is_file():
        raise FileNotFoundError(f"Missing class mapping: {mapping_csv}")

    mapping_df = pd.read_csv(mapping_csv)
    mapping = dict(zip(mapping_df["source_class"], mapping_df["plantvillage_class"]))

    class_names = load_class_names(Path(SPLITS_DIR) / "class_names.json")
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    # Collect test filenames and MD5 hashes to guarantee zero test leakage
    test_csv = Path(SPLITS_DIR) / "field_test.csv"
    test_filenames = set()
    test_hashes = set()

    if test_csv.is_file():
        test_df = pd.read_csv(test_csv)
        path_col = "filepath" if "filepath" in test_df.columns else "path"
        name_col = "filename" if "filename" in test_df.columns else path_col
        test_filenames = set(test_df[name_col].apply(lambda p: Path(p).name))

        for p_str in test_df[path_col]:
            p = ROOT / p_str if not Path(p_str).is_absolute() else Path(p_str)
            if p.is_file():
                try:
                    test_hashes.add(compute_file_md5(p))
                except Exception:
                    pass
        print(f"Loaded {len(test_filenames)} test filenames and {len(test_hashes)} test hashes for anti-leakage.")

    train_dir = RAW_PLANTDOC_DIR / "train"
    records = []
    seen_train_hashes = set()
    skipped_no_match = 0
    skipped_corrupted = 0
    skipped_duplicates = 0
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

            # Anti-leak check by filename
            if img_file.name in test_filenames:
                skipped_test_leak += 1
                continue

            # Verify image file integrity (detect broken headers, truncated bytes)
            if not is_image_valid(img_file):
                skipped_corrupted += 1
                continue

            # Deduplication & anti-leak check by MD5 hash
            file_md5 = compute_file_md5(img_file)
            if file_md5 in test_hashes:
                skipped_test_leak += 1
                continue

            if file_md5 in seen_train_hashes:
                skipped_duplicates += 1
                continue

            seen_train_hashes.add(file_md5)
            rel_path = img_file.relative_to(ROOT).as_posix()
            records.append({
                "filepath": rel_path,
                "filename": img_file.name,
                "label": pv_class,
                "label_idx": label_idx,
                "source": "PlantDoc-Full",
            })

    df = pd.DataFrame(records)
    print(f"\n================ Data Quality & Cleaning Summary ================")
    print(f"Total verified clean training images: {len(df)}")
    print(f"Target classes covered: {df['label'].nunique()} / 38")
    print(f"Corrupted / truncated images purged: {skipped_corrupted}")
    print(f"Duplicate images removed: {skipped_duplicates}")
    print(f"Test-leakage candidates blocked: {skipped_test_leak}")
    print(f"Saving pristine manifest to: {OUTPUT_SPLIT_CSV}")
    print(f"=================================================================\n")

    OUTPUT_SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_SPLIT_CSV, index=False)
    return df


if __name__ == "__main__":
    build_field_full_split()
