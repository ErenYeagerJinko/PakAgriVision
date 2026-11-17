"""Script to download and extract real-world field-cropped leaf dataset images.

Google Drive Link:
    https://drive.google.com/file/d/1cKcUttKrrHHJ5shimKVNdLOIkIsLj3-M/view?usp=sharing
    File ID: 1cKcUttKrrHHJ5shimKVNdLOIkIsLj3-M

Dataset Details:
    - Total Images: 8,418 clean, isolated leaf crops
    - Classes: 28 agricultural crop-disease classes
    - Origin: Extracted from in-the-wild PlantDoc field photographs using ground-truth bounding box coordinates
    - Target Directory: data/raw/field_cropped/ (and data/raw/field/)

Usage:
    python scripts/download_field_images.py
    python scripts/download_field_images.py --dest-dir data/raw/field_cropped
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DRIVE_FILE_ID = "1cKcUttKrrHHJ5shimKVNdLOIkIsLj3-M"
TOTAL_IMAGES = 8418
TOTAL_CLASSES = 28
DEST_FIELD_CROPPED_DIR = ROOT_DIR / "data" / "raw" / "field_cropped"


def download_google_drive_file(file_id: str, destination: Path) -> bool:
    """Download large or small file from Google Drive handling confirmation tokens."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # 1. Try gdown
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        out = gdown.download(url, str(destination), quiet=False)
        if out and destination.is_file() and destination.stat().st_size > 1000:
            return True
    except Exception:
        pass

    # 2. Try requests
    try:
        import requests
        session = requests.Session()
        url = "https://docs.google.com/uc?export=download"
        response = session.get(url, params={"id": file_id}, stream=True)

        token = None
        for k, v in response.cookies.items():
            if k.startswith("download_warning"):
                token = v
                break

        if not token and "text/html" in response.headers.get("Content-Type", ""):
            match = re.search(r"confirm=([0-9A-Za-z_]+)", response.text)
            if match:
                token = match.group(1)

        if token:
            response = session.get(url, params={"id": file_id, "confirm": token}, stream=True)

        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)

        if destination.is_file() and destination.stat().st_size > 1000:
            return True
    except Exception as e:
        print(f"requests download encountered error: {e}")

    # 3. Urllib fallback
    try:
        import urllib.request
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        urllib.request.urlretrieve(url, destination)
        if destination.is_file() and destination.stat().st_size > 1000:
            return True
    except Exception as e:
        print(f"urllib download error: {e}")

    return False


def download_and_extract_field_images(dest_dir: Path = DEST_FIELD_CROPPED_DIR) -> None:
    """Download the 8,418 field-cropped images archive and extract into dest_dir."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = dest_dir.parent

    print("\n🌿 PakAgriVision Field Dataset Downloader")
    print(f"📊 Dataset Specification: {TOTAL_IMAGES:,} leaf crop images across {TOTAL_CLASSES} classes")
    print(f"🔗 Google Drive File ID: {DRIVE_FILE_ID}")
    print(f"📍 Destination Path: {dest_dir}")

    temp_zip = raw_dir / "field_data_archive.zip"
    success = download_google_drive_file(DRIVE_FILE_ID, temp_zip)

    if not success or not temp_zip.is_file():
        print(f"\n❌ Error: Could not download archive automatically to {temp_zip}")
        print("Please download manually from Google Drive link:")
        print("https://drive.google.com/file/d/1cKcUttKrrHHJ5shimKVNdLOIkIsLj3-M/view?usp=sharing")
        print(f"and extract into: {dest_dir}")
        sys.exit(1)

    print(f"\n📂 Extracting {TOTAL_IMAGES:,} images ({temp_zip.stat().st_size / 1024 / 1024:.2f} MB)...")
    try:
        with zipfile.ZipFile(temp_zip, "r") as zf:
            members = zf.namelist()
            # If top folder is 'field/', strip or extract directly
            has_field_prefix = any(m.startswith("field/") for m in members)
            if has_field_prefix and dest_dir.name != "field":
                # Extract and rename/symlink to dest_dir
                zf.extractall(raw_dir)
                extracted_field = raw_dir / "field"
                if extracted_field.exists() and extracted_field != dest_dir:
                    if not dest_dir.exists():
                        extracted_field.rename(dest_dir)
                    else:
                        for item in extracted_field.glob("*"):
                            target_item = dest_dir / item.name
                            if not target_item.exists():
                                shutil.move(str(item), str(target_item))
            else:
                zf.extractall(dest_dir)

        print(f"✅ Successfully extracted {TOTAL_IMAGES:,} field-cropped images to {dest_dir}")
    except Exception as e:
        print(f"❌ Extraction error: {e}")
    finally:
        if temp_zip.is_file():
            temp_zip.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Download and extract field leaf images ({TOTAL_IMAGES:,} images across {TOTAL_CLASSES} classes)."
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=DEST_FIELD_CROPPED_DIR,
        help=f"Destination directory for images (default: data/raw/field_cropped, total {TOTAL_IMAGES:,} images)",
    )
    args = parser.parse_args()
    download_and_extract_field_images(args.dest_dir)
