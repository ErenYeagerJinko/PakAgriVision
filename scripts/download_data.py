"""Script to download and extract the curated dataset archive from Google Drive.

Downloads the curated PlantVillage benchmark (~32k images across 38 classes)
and field robustness images (150 images across 28 classes) from Google Drive
and extracts them directly into data/raw/.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --dest-dir data/raw
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
DEST_DATA_DIR = Path("data/raw")


def download_and_extract_data(dest_dir: Path = DEST_DATA_DIR) -> None:
    """Download curated dataset archive from Google Drive and extract into dest_dir."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    zip_target = dest_dir / "data.zip"
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    print(f"Downloading curated dataset archive (data.zip) from Google Drive ({url})...")

    if gdown is not None:
        gdown.download(url, str(zip_target), quiet=False)
    else:
        import urllib.request
        urllib.request.urlretrieve(url, zip_target)

    if not zip_target.is_file():
        print(f"Error: Failed to download archive to {zip_target}")
        sys.exit(1)

    print(f"Extracting archive into {dest_dir}...")
    try:
        with zipfile.ZipFile(zip_target, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
        print(f"Successfully extracted dataset into {dest_dir}")
        print("Populated directories:")
        if (dest_dir / "plantvillage").exists():
            print(f"  - {dest_dir / 'plantvillage'}")
        if (dest_dir / "field").exists():
            print(f"  - {dest_dir / 'field'}")
    finally:
        if zip_target.is_file():
            zip_target.unlink()  # remove temp zip file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract curated dataset.")
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=DEST_DATA_DIR,
        help="Destination directory to extract dataset into (default: data/raw)",
    )
    args = parser.parse_args()
    download_and_extract_data(args.dest_dir)

