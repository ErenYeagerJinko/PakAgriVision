"""Complete real PlantVillage download for all 38 classes via web scraping (no rate limits).
Verifies 100% of images with Pillow.
"""

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from PIL import Image

ROOT = Path(".")
PV_DIR = ROOT / "data" / "raw" / "plantvillage" / "color"
CLASS_NAMES_PATH = ROOT / "splits" / "class_names.json"

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

# Mapping from class_names.json to spMohanty repo folder name
REPO_NAME_MAP = {
    "Cherry___healthy": "Cherry_(including_sour)___healthy",
    "Cherry___Powdery_mildew": "Cherry_(including_sour)___Powdery_mildew",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust": "Corn_(maize)___Common_rust_",
    "Corn___Northern_Leaf_Blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn___healthy": "Corn_(maize)___healthy",
    "Grape___Esca_Black_Measles": "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight Isariopsis_Leaf_Spot": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing(Citrus_greening)": "Orange___Haunglongbing_(Citrus_greening)",
    "Pepper___bell_Bacterial_spot": "Pepper,_bell___Bacterial_spot",
    "Pepper___healthy": "Pepper,_bell___healthy",
}

RAW_BASE = "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color"
WEB_BASE = "https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color"

print("Checking and downloading real images for remaining PlantVillage classes...")

for idx, class_name in enumerate(class_names, 1):
    c_dir = PV_DIR / class_name
    c_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in c_dir.glob("*.*") if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    
    needed = 10 - len(existing)
    if needed <= 0:
        continue

    repo_folder = REPO_NAME_MAP.get(class_name, class_name)
    encoded_folder = urllib.parse.quote(repo_folder)
    web_url = f"{WEB_BASE}/{encoded_folder}"
    
    req = urllib.request.Request(web_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        matches = re.findall(r'"path":"raw/color/[^"]+/([^"]+\.(?:JPG|jpg|jpeg|png))"', html)
        if not matches:
            # Fallback match pattern
            matches = re.findall(r'\"name\":\"([^\"]+\.(?:JPG|jpg|jpeg|png))\"', html)
    except Exception as e:
        print(f"[{idx}/38] Error scraping {class_name}: {e}")
        continue

    downloaded = 0
    for filename in matches:
        if downloaded >= needed:
            break
        target_path = c_dir / filename
        if target_path.exists():
            continue

        raw_url = f"{RAW_BASE}/{encoded_folder}/{urllib.parse.quote(filename)}"
        try:
            urllib.request.urlretrieve(raw_url, target_path)
            with Image.open(target_path) as im:
                im.verify()
            with Image.open(target_path) as im:
                im.convert("RGB")
            downloaded += 1
        except Exception as err:
            print(f"Failed {filename}: {err}")
            if target_path.exists():
                target_path.unlink()

    total_in_class = len(list(c_dir.glob('*.*')))
    print(f"[{idx}/38] {class_name}: now has {total_in_class} images.")

# Verification
total_verified = 0
corrupt_count = 0
incomplete_classes = []

for class_name in class_names:
    c_dir = PV_DIR / class_name
    imgs = [p for p in c_dir.glob("*.*") if p.is_file()]
    if len(imgs) < 10:
        incomplete_classes.append((class_name, len(imgs)))
    for img in imgs:
        try:
            with Image.open(img) as im:
                im.verify()
            with Image.open(img) as im:
                im.convert("RGB")
            total_verified += 1
        except Exception as e:
            print(f"CORRUPT: {img}: {e}")
            corrupt_count += 1
            img.unlink()

print("\n--- Summary ---")
print(f"Total verified PlantVillage images: {total_verified}")
print(f"Total corrupt: {corrupt_count}")
print(f"Classes under 10 images: {len(incomplete_classes)}")
if incomplete_classes:
    for c, cnt in incomplete_classes:
        print(f"  - {c}: {cnt}")
