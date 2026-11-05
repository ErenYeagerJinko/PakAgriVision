"""Script to download and extract baseline run artifacts from Google Drive.

Usage:
    python scripts/download_weights.py
"""

import argparse
import sys
import zipfile
from pathlib import Path

try:
    import gdown
except ImportError:
    gdown = None

DRIVE_FILE_ID = "1RBab1_qShQGObGJtnefs0HlR-G-orM1g"
DEST_RUNS_DIR = Path("runs")


def download_and_extract_weights(runs_dir: Path = DEST_RUNS_DIR) -> None:
    """Download baseline archive from Google Drive and extract into runs/."""
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    zip_target = runs_dir / "baseline_artifacts.zip"
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    print(f"Downloading baseline artifacts archive from Google Drive ({url})...")

    if gdown is not None:
        gdown.download(url, str(zip_target), quiet=False)
    else:
        import urllib.request
        urllib.request.urlretrieve(url, zip_target)

    if not zip_target.is_file():
        print(f"Error: Failed to download archive to {zip_target}")
        sys.exit(1)

    print(f"Extracting archive into {runs_dir}...")
    try:
        with zipfile.ZipFile(zip_target, "r") as zip_ref:
            zip_ref.extractall(runs_dir)
        print(f"Successfully extracted baseline artifacts into {runs_dir / 'baseline'}")
    finally:
        if zip_target.is_file():
            zip_target.unlink()  # remove temp zip file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract baseline artifacts.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEST_RUNS_DIR,
        help="Runs directory to extract into (default: runs)",
    )
    args = parser.parse_args()
    download_and_extract_weights(args.runs_dir)
