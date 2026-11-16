"""Script to download and extract trained model weights from Google Drive.

Supported Weight Targets:
    1. baseline: ResNet-50 Day 2 baseline run artifacts (runs/baseline/)
    2. field_cropped: ResNet-50 fine-tuned on 8,418 leaf crops (runs/field_cropped/)
    3. yolo: YOLOv8n leaf detector weights (models/yolo/leaf_detector.pt)
    4. all: Download and extract all 3 sets of model weights

Google Drive Links & File IDs:
    - Baseline Weights: https://drive.google.com/file/d/1R1qFkk4v3-5dzNFdRMxG7-gZXFwyo8Cv/view?usp=sharing
      ID: 1R1qFkk4v3-5dzNFdRMxG7-gZXFwyo8Cv
    - Field Cropped Weights: https://drive.google.com/file/d/14ygSAzCwERyvAMALDSy3i-G1oXwOnNri/view?usp=sharing
      ID: 14ygSAzCwERyvAMALDSy3i-G1oXwOnNri
    - YOLOv8n Weights: https://drive.google.com/file/d/10fqb4sptHYqq5Y83Atut6qQivv-tg063/view?usp=sharing
      ID: 10fqb4sptHYqq5Y83Atut6qQivv-tg063

Usage:
    python scripts/download_weights.py --target all
    python scripts/download_weights.py --target field_cropped
    python scripts/download_weights.py --target baseline
    python scripts/download_weights.py --target yolo
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

WEIGHTS_MANIFEST = {
    "baseline": {
        "file_id": "1R1qFkk4v3-5dzNFdRMxG7-gZXFwyo8Cv",
        "name": "Baseline ResNet-50 Weights (runs/baseline/)",
        "dest_dir": ROOT_DIR / "runs",
        "zip_name": "baseline_weights.zip",
        "expected_subfolder": "baseline",
    },
    "field_cropped": {
        "file_id": "14ygSAzCwERyvAMALDSy3i-G1oXwOnNri",
        "name": "Field Cropped ResNet-50 Weights (runs/field_cropped/)",
        "dest_dir": ROOT_DIR / "runs",
        "zip_name": "field_cropped_weights.zip",
        "expected_subfolder": "field_cropped",
    },
    "yolo": {
        "file_id": "10fqb4sptHYqq5Y83Atut6qQivv-tg063",
        "name": "YOLOv8n Leaf Detector Weights (models/yolo/leaf_detector.pt)",
        "dest_dir": ROOT_DIR / "models",
        "zip_name": "yolo_weights.zip",
        "expected_subfolder": "yolo",
    },
}


def download_google_drive_file(file_id: str, destination: Path) -> bool:
    """Download large or small file from Google Drive using gdown or requests."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # 1. Try gdown if available
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        out = gdown.download(url, str(destination), quiet=False)
        if out and destination.is_file() and destination.stat().st_size > 1000:
            return True
    except Exception:
        pass

    # 2. Try requests with session and token confirmation handling
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

    # 3. Fallback to urllib
    try:
        import urllib.request
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        urllib.request.urlretrieve(url, destination)
        if destination.is_file() and destination.stat().st_size > 1000:
            return True
    except Exception as e:
        print(f"urllib download error: {e}")

    return False


def extract_weight_archive(target_key: str) -> None:
    """Download and unpack a specific weight target."""
    entry = WEIGHTS_MANIFEST[target_key]
    dest_dir = entry["dest_dir"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    zip_path = dest_dir / entry["zip_name"]
    print(f"\n📦 Fetching {entry['name']}...")
    print(f"🔗 Google Drive File ID: {entry['file_id']}")

    success = download_google_drive_file(entry["file_id"], zip_path)
    if not success or not zip_path.is_file():
        print(f"❌ Failed to download {entry['name']} to {zip_path}")
        print("Please download manually from Google Drive and place into:", dest_dir)
        return

    print(f"📂 Extracting {zip_path.name} ({zip_path.stat().st_size / 1024 / 1024:.2f} MB)...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)

        # For YOLO, ensure models/yolo/leaf_detector.pt exists
        if target_key == "yolo":
            yolo_dir = dest_dir / "yolo"
            yolo_pt = yolo_dir / "leaf_detector.pt"
            best_pt = yolo_dir / "best.pt"
            if not yolo_pt.exists() and best_pt.exists():
                shutil.copy(best_pt, yolo_pt)
            # Also mirror to runs/yolo for experiment symmetry
            runs_yolo = ROOT_DIR / "runs" / "yolo"
            runs_yolo.mkdir(parents=True, exist_ok=True)
            if yolo_pt.exists():
                shutil.copy(yolo_pt, runs_yolo / "best.pt")

        print(f"✅ Successfully unpacked into: {dest_dir / entry['expected_subfolder']}")
    except Exception as e:
        print(f"❌ Extraction error: {e}")
    finally:
        if zip_path.is_file():
            zip_path.unlink()  # remove temp zip file


def main():
    parser = argparse.ArgumentParser(description="Download model weights for PakAgriVision.")
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        choices=["all", "baseline", "field_cropped", "yolo"],
        help="Weight target to download: 'baseline', 'field_cropped', 'yolo', or 'all' (default: all)",
    )
    args = parser.parse_args()

    if args.target == "all":
        for k in ["baseline", "field_cropped", "yolo"]:
            extract_weight_archive(k)
    else:
        extract_weight_archive(args.target)

    print("\n🌿 Weights download routine complete.")


if __name__ == "__main__":
    main()
