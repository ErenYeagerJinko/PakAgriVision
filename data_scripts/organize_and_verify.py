"""Organize Babar's 150 field files into data/raw/field/, complete PlantVillage photos, and verify 0 corrupt files."""

import json
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path
from PIL import Image

ROOT = Path(".")
PV_DIR = ROOT / "data" / "raw" / "plantvillage" / "color"
FIELD_DIR = ROOT / "data" / "raw" / "field"
CLASS_NAMES_PATH = ROOT / "splits" / "class_names.json"

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

FIELD_DIR.mkdir(parents=True, exist_ok=True)

# 1. Move all plantdoc_* files to data/raw/field/<class_name>/
plantdoc_moved = 0
for f in list(PV_DIR.glob("*/plantdoc_*")):
    class_name = f.parent.name
    dest_dir = FIELD_DIR / class_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / f.name
    shutil.move(str(f), str(dest_file))
    plantdoc_moved += 1

print(f"Moved {plantdoc_moved} Babar field images into {FIELD_DIR}")

# 2. Verify all field images with Pillow
corrupt_field = 0
verified_field = 0
for f in FIELD_DIR.glob("*/*"):
    if f.is_file() and not f.name.startswith("."):
        try:
            with Image.open(f) as im:
                im.verify()
            with Image.open(f) as im:
                im.convert("RGB")
            verified_field += 1
        except Exception as e:
            print(f"Corrupt field image: {f}: {e}")
            corrupt_field += 1

print(f"Field dataset verification: {verified_field} valid, {corrupt_field} corrupt.")

# 3. Complete PlantVillage 10 images per class
GITHUB_API_BASE = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color"

print("\nCompleting 10 real PlantVillage images per class...")
for idx, class_name in enumerate(class_names, 1):
    class_dir = PV_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    existing_imgs = [p for p in class_dir.glob("*") if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    
    needed = 10 - len(existing_imgs)
    if needed <= 0:
        continue

    encoded_name = urllib.parse.quote(class_name)
    api_url = f"{GITHUB_API_BASE}/{encoded_name}"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            files_meta = json.load(resp)
    except Exception as e:
        print(f"Failed to query {class_name}: {e}")
        continue

    img_files = [f for f in files_meta if f["name"].lower().endswith((".jpg", ".jpeg", ".png"))]
    downloaded = 0
    for f_info in img_files:
        if downloaded >= needed:
            break
        target_path = class_dir / f_info["name"]
        if target_path.exists():
            continue
        try:
            urllib.request.urlretrieve(f_info["download_url"], target_path)
            with Image.open(target_path) as im:
                im.verify()
            with Image.open(target_path) as im:
                im.convert("RGB")
            downloaded += 1
        except Exception as err:
            print(f"Failed download {f_info['name']}: {err}")
            if target_path.exists():
                target_path.unlink()
    print(f"[{idx}/38] {class_name}: now has {len(list(class_dir.glob('*.*')))} images.")
    time.sleep(0.4)

# 4. Final verification of all PlantVillage images
verified_pv = 0
corrupt_pv = 0
classes_with_images = 0
for class_name in class_names:
    c_dir = PV_DIR / class_name
    imgs = [p for p in c_dir.glob("*.*") if p.is_file()]
    if imgs:
        classes_with_images += 1
    for img in imgs:
        try:
            with Image.open(img) as im:
                im.verify()
            with Image.open(img) as im:
                im.convert("RGB")
            verified_pv += 1
        except Exception as e:
            print(f"Corrupt PV image: {img}: {e}")
            corrupt_pv += 1

print("\n================ FINAL VERIFICATION REPORT ================")
print(f"Babar Field Images ({FIELD_DIR}):")
print(f"  - Total files: {verified_field}")
print(f"  - Corrupt: {corrupt_field}")
print(f"Leader PlantVillage Images ({PV_DIR}):")
print(f"  - Classes covered: {classes_with_images}/38")
print(f"  - Total verified real photos: {verified_pv}")
print(f"  - Corrupt: {corrupt_pv}")
print("===========================================================")
