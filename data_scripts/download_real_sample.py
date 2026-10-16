"""Download a real photographic sample (10 real images per class) directly from the official PlantVillage repository.
Verifies every image using Pillow to guarantee 0 corrupt files.
"""

import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from PIL import Image

ROOT = Path(".")
DATA_DIR = ROOT / "data" / "raw" / "plantvillage" / "color"
CLASS_NAMES_PATH = ROOT / "splits" / "class_names.json"

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

# Clear existing contents in color/
import shutil
if DATA_DIR.exists():
    shutil.rmtree(DATA_DIR)
DATA_DIR.mkdir(parents=True, exist_ok=True)

GITHUB_API_BASE = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color"
RAW_BASE = "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color"

print(f"Starting download of real photographic PlantVillage images across {len(class_names)} classes...")

total_downloaded = 0
corrupt_count = 0

for idx, class_name in enumerate(class_names, 1):
    class_folder = DATA_DIR / class_name
    class_folder.mkdir(parents=True, exist_ok=True)

    encoded_name = urllib.parse.quote(class_name)
    api_url = f"{GITHUB_API_BASE}/{encoded_name}"
    
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            files_meta = json.load(resp)
    except Exception as e:
        print(f"[{idx}/{len(class_names)}] Failed to list {class_name}: {e}")
        continue

    # Pick up to 10 images
    img_files = [f for f in files_meta if f["name"].lower().endswith((".jpg", ".jpeg", ".png"))][:10]
    
    saved_this_class = 0
    for f_info in img_files:
        dl_url = f_info["download_url"]
        target_path = class_folder / f_info["name"]
        
        # Download
        try:
            urllib.request.urlretrieve(dl_url, target_path)
            # Verify image integrity with Pillow
            with Image.open(target_path) as im:
                im.verify()
            # Re-open and convert to RGB to verify decoding
            with Image.open(target_path) as im:
                im.convert("RGB")
            saved_this_class += 1
            total_downloaded += 1
        except Exception as err:
            corrupt_count += 1
            print(f"Corrupt or failed: {target_path.name}: {err}")
            if target_path.exists():
                target_path.unlink()
                
    print(f"[{idx}/{len(class_names)}] {class_name}: saved {saved_this_class} verified real images.")
    # Brief pause to respect GitHub rate limits
    time.sleep(0.5)

print("\n--- Download & Verification Summary ---")
print(f"Total verified real images downloaded: {total_downloaded}")
print(f"Corrupt images: {corrupt_count} (all corrupt files pruned)")
print(f"Destination: {DATA_DIR}")
