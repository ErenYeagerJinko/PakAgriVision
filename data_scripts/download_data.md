# Dataset Acquisition & Download Guide (`data_scripts/download_data.md`)

This document provides the exact automated and manual commands for acquiring the primary **PlantVillage** foliar disease benchmark and the secondary **PlantDoc** field robustness dataset as mandated by Deliverable 1 (§2 Data Collection).

---

## 1. Primary Curated Dataset (~32k Images, 38 Classes) via Google Drive

To ensure data-centric quality, reproducibility, and prevent training instability:
- **Raw Benchmark Limitations**: The original raw PlantVillage dataset on Kaggle contained ~54,303 uncurated images with corrupted JPEG files, exact duplicate photographs, near-duplicate leaf shots, and an extreme 36:1 class imbalance ratio (e.g., Orange Huanglongbing having 5,507 images while Potato healthy had only 152).
- **Curated Dataset Implementation**: All raw candidate files underwent binary integrity checking (`PIL.Image.verify()`), exact duplicate elimination via 128-bit MD5 hashes, near-duplicate clustering via 64-bit DCT perceptual hashes (`imagehash.phash`), and class capping at a ceiling of **1,000 clean images per class**.
- **Final Scale**: **31,694 verified clean unique images (~32k images)** across 38 classes, plus 150 real-world field robustness images across 28 matching classes.
- **We do NOT use raw Kaggle downloads anymore**: All team members and graders download the verified, pre-cleaned, and capped archive directly from Google Drive.

### Primary Download Instructions: Pre-Packaged `data.zip`
- **Google Drive Link**: [Download data.zip](https://drive.google.com/file/d/1OE17MYZqriiP9mBdCat_JglMxx_7MY1V/view?usp=drivesdk)
- **Archive File**: `data.zip`
- **Extraction Command**:
  ```bash
  # Extract directly into data/raw/
  python -m zipfile -e data.zip data/raw/
  ```
- **Resulting Directory Layout**:
  ```text
  data/raw/
  ├── plantvillage/
  │   └── color/
  │       ├── Apple___Apple_scab/
  │       ├── ... (38 classes, max 1,000 images each)
  └── field/
      ├── Apple___Apple_scab/
      └── ... (28 classes, 150 images total)
  ```

---

## 2. Historical Raw Ingestion Sources (Reference Only)

> **Note:** These options represent the raw initial benchmarks before deduplication, corrupt file pruning, and class capping. They are retained solely for data provenance documentation.

- **Initial Raw Benchmark**: Kaggle `abdallahalidev/plantvillage-dataset` (~54k uncurated images).
- **Historical Paper**: Hughes, D.P., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics*. arXiv:1511.08060.
- **License**: CC0 Public Domain.

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
