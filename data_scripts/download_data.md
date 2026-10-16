# Dataset Acquisition & Download Guide (`data_scripts/download_data.md`)

This document provides the exact automated and manual commands for acquiring the primary **PlantVillage** foliar disease benchmark and the secondary **PlantDoc** field robustness dataset as mandated by Deliverable 1 (§2 Data Collection).

---

## 1. PlantVillage Benchmark Dataset (~54k Images, 38 Classes)

- **Source**: Kaggle `abdallahalidev/plantvillage-dataset`
- **Citation**: Hughes, D.P., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics*. arXiv:1511.08060.
- **Scale**: 54,303 color images across 14 crop species and 38 classes.
- **Format**: 256×256 pixels, 3-channel RGB JPEG format.
- **License**: CC0 Public Domain.

### Option A: Kaggle API Download (Automated)
Ensure your `kaggle.json` API token is installed in `~/.kaggle/` (`%USERPROFILE%\.kaggle\kaggle.json` on Windows):

```bash
# Download and unpack directly into data/raw/
kaggle datasets download -d abdallahalidev/plantvillage-dataset -p data/raw --unzip
```

Ensure the target layout matches:
```text
data/raw/plantvillage/color/<PlantVillage_class_name>/<filename>.JPG
```

### Option B: Mendeley Direct Mirror (Fast HTTP)
Download the official non-augmented color archive (~828 MB):
```bash
# Direct mirror link:
curl -L -o data/raw/plant_village.zip "https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded"

# Extract color images into data/raw/plantvillage/color/
python -m zipfile -e data/raw/plant_village.zip data/raw/
```

### Option C: Hugging Face Mirror
```bash
curl -L -o data/raw/data.zip "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/data.zip"
```

---

## 2. Real-World Field Robustness Dataset (150 Images, 28 Classes)

- **Source**: PlantDoc Dataset (`pratikkayal/PlantDoc-Dataset`)
- **Citation**: Singh, D., et al. (2020). *PlantDoc: A dataset for visual plant disease detection*. ACM CoDS-COMAD.
- **Scale**: 150 real-world foliar photographs across 28 matching classes with complex outdoor backgrounds (soil, sunlight, weeds, hands).
- **License**: CC-BY-4.0.

### Acquisition Steps
1. Clone the PlantDoc repository or download the archive:
   ```bash
   git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git data/raw/plantdoc_repo
   ```
2. The exact 150 files and their mappings are cataloged in `field_meta/field_labels.csv` and `field_meta/class_mapping.csv`.
3. Generate the stratified field test (92 images) and adaptation (58 images) splits:
   ```bash
   python data_scripts/make_field_splits.py
   ```

---

## 3. Data Cleaning, Deduplication, and Capping Pipeline

Once raw images are populated in `data/raw/plantvillage/color/`, run the automated cleaning pipeline:

```bash
# Verify integrity (Pillow), remove exact duplicates (MD5) and near-duplicates (pHash),
# cap each class at 1,000 clean images, and prune duplicate/excess files:
python data_scripts/clean_and_cap_dataset.py --max-per-class 1000

# Generate 70/15/15 stratified, group-leakage-free train/val/test splits and class weights:
python data_scripts/make_splits.py

# Validate the 38-class treatment knowledge base:
python data_scripts/validate_treatments.py
```
