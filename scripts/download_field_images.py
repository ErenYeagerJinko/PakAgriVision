"""Script to download and extract real-world field robustness images.

Downloads the 150 real-world foliar field images across 28 matching classes
and populates data/raw/field/.

Usage:
    python scripts/download_field_images.py
    python scripts/download_field_images.py --dest-dir data/raw/field
"""

import argparse
import sys
import zipfile
from pathlib import Path

try:
    import gdown
except ImportError:
    gdown = None

DRIVE_FILE_ID = "1OE17MYZqriiP9mBdCat_JglMxx_7MY1V"
DEST_FIELD_DIR = Path("data/raw/field")


def download_and_extract_field_images(dest_dir: Path = DEST_FIELD_DIR) -> None:
    """Download field images archive from Google Drive and extract into dest_dir."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = dest_dir.parent if dest_dir.name == "field" else dest_dir

    zip_target = raw_dir / "field_data_temp.zip"
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    print(f"Downloading field dataset from Google Drive ({url})...")

    if gdown is not None:
        gdown.download(url, str(zip_target), quiet=False)
    else:
        import urllib.request
        urllib.request.urlretrieve(url, zip_target)

    if not zip_target.is_file():
        print(f"Error: Failed to download archive to {zip_target}")
        sys.exit(1)

    print(f"Extracting field images into {dest_dir}...")
    try:
        with zipfile.ZipFile(zip_target, "r") as zip_ref:
            field_members = [
                m for m in zip_ref.namelist()
                if m.startswith("field/") or "/field/" in m
            ]
            if field_members:
                # If zip contains 'field/...', extract into raw_dir so raw_dir / 'field' is populated
                for member in field_members:
                    zip_ref.extract(member, raw_dir)
            else:
                # Otherwise extract directly into dest_dir
                zip_ref.extractall(dest_dir)
        print(f"Successfully populated field images in {dest_dir}")
    finally:
        if zip_target.is_file():
            zip_target.unlink()  # remove temp zip file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract field robustness images.")
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=DEST_FIELD_DIR,
        help="Destination directory for field images (default: data/raw/field)",
    )
    args = parser.parse_args()
    download_and_extract_field_images(args.dest_dir)

