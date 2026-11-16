"""Extract individual leaf crops from PlantDoc Object Detection bounding boxes.

Clones pratikkayal/PlantDoc-Object-Detection-Dataset, reads train_labels.csv,
maps classes via field_meta/class_mapping.csv, strictly excludes any image in
splits/field_test.csv, crops each leaf bounding box, and creates
splits/field_cropped_train.csv.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from PIL import Image

from src.config import DATA_DIR, ROOT, SPLITS_DIR
from src.data.splits import load_class_names

OD_REPO_URL = "https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset.git"
RAW_OD_DIR = DATA_DIR / "raw" / "plantdoc_od"
CROPPED_OUT_DIR = DATA_DIR / "raw" / "field_cropped"
OUTPUT_SPLIT_CSV = Path(SPLITS_DIR) / "field_cropped_train.csv"


def download_od_repo(dest_dir: Path) -> Path:
    """Clone or pull official PlantDoc Object Detection dataset."""
    dest_dir = Path(dest_dir)
    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    if (dest_dir / "TRAIN").is_dir() and (dest_dir / "train_labels.csv").is_file():
        print(f"PlantDoc Object Detection repo already present at {dest_dir}")
        return dest_dir

    print(f"Cloning PlantDoc Object Detection dataset from {OD_REPO_URL}...")
    cmd = ["git", "clone", "--depth", "1", OD_REPO_URL, str(dest_dir)]
    subprocess.run(cmd, check=True)
    return dest_dir


def crop_all_bounding_boxes() -> pd.DataFrame:
    """Crop all leaf bounding boxes and generate training manifest."""
    od_dir = download_od_repo(RAW_OD_DIR)
    labels_csv_path = od_dir / "train_labels.csv"
    train_images_dir = od_dir / "TRAIN"

    if not labels_csv_path.is_file():
        raise FileNotFoundError(f"Missing train_labels.csv: {labels_csv_path}")

    # Load class mapping
    mapping_csv = ROOT / "field_meta" / "class_mapping.csv"
    mapping_df = pd.read_csv(mapping_csv)
    mapping = dict(zip(mapping_df["source_class"], mapping_df["plantvillage_class"]))

    # Load 38 target class names
    class_names = load_class_names(Path(SPLITS_DIR) / "class_names.json")
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    # Load field_test to strictly prevent test data leakage
    test_csv = Path(SPLITS_DIR) / "field_test.csv"
    test_filenames = set()
    if test_csv.is_file():
        test_df = pd.read_csv(test_csv)
        col = "filename" if "filename" in test_df.columns else "path"
        test_filenames = set(test_df[col].apply(lambda p: Path(p).name.lower()))
        print(f"Loaded {len(test_filenames)} test filenames for strict anti-leakage.")

    df_boxes = pd.read_csv(labels_csv_path)
    print(f"Total bounding boxes in manifest: {len(df_boxes)}")

    CROPPED_OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    skipped_test_leak = 0
    skipped_no_class = 0
    skipped_corrupted = 0
    skipped_tiny_box = 0

    current_img_name = None
    current_img = None

    for i, row in df_boxes.iterrows():
        img_name = str(row["filename"]).strip()
        source_class = str(row["class"]).strip()

        # Anti-leak check
        if img_name.lower() in test_filenames:
            skipped_test_leak += 1
            continue

        pv_class = mapping.get(source_class)
        if not pv_class or pv_class not in class_to_idx:
            skipped_no_class += 1
            continue

        label_idx = class_to_idx[pv_class]

        # Coordinates
        try:
            xmin = max(0, int(row["xmin"]))
            ymin = max(0, int(row["ymin"]))
            xmax = int(row["xmax"])
            ymax = int(row["ymax"])
        except (ValueError, TypeError):
            continue

        if (xmax - xmin) < 20 or (ymax - ymin) < 20:
            skipped_tiny_box += 1
            continue

        img_path = train_images_dir / img_name
        if not img_path.is_file():
            continue

        # Cache open image across consecutive bounding boxes of the same image
        if img_name != current_img_name:
            if current_img is not None:
                current_img.close()
            current_img_name = img_name
            try:
                current_img = Image.open(img_path).convert("RGB")
            except Exception:
                current_img = None
                skipped_corrupted += 1
                continue

        if current_img is None:
            continue

        # Crop leaf box
        try:
            crop_box = (xmin, ymin, min(current_img.width, xmax), min(current_img.height, ymax))
            leaf_crop = current_img.crop(crop_box)
        except Exception:
            continue

        # Save cropped leaf
        class_folder = CROPPED_OUT_DIR / pv_class
        class_folder.mkdir(parents=True, exist_ok=True)
        crop_filename = f"crop_{Path(img_name).stem}_{i}.jpg"
        save_path = class_folder / crop_filename

        try:
            leaf_crop.save(save_path, "JPEG", quality=90)
        except Exception:
            continue

        rel_path = save_path.relative_to(ROOT).as_posix()
        records.append({
            "filepath": rel_path,
            "filename": crop_filename,
            "label": pv_class,
            "label_idx": label_idx,
            "source": "PlantDoc-Cropped",
        })

    if current_img is not None:
        current_img.close()

    df_out = pd.DataFrame(records)
    print(f"\n================ Leaf Bounding Box Cropping Summary ================")
    print(f"Total isolated leaf crops generated: {len(df_out)}")
    print(f"Target classes covered: {df_out['label'].nunique()} / 38")
    print(f"Excluded test-leakage boxes: {skipped_test_leak}")
    print(f"Skipped non-matching classes: {skipped_no_class}")
    print(f"Skipped micro/invalid boxes: {skipped_tiny_box}")
    print(f"Saving manifest to: {OUTPUT_SPLIT_CSV}")
    print(f"====================================================================\n")

    OUTPUT_SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_SPLIT_CSV, index=False)
    return df_out


if __name__ == "__main__":
    crop_all_bounding_boxes()
