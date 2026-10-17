# PakAgriVision

**Team Yeagerists++**
- **Abdul Ahad** — 24L-3029
- **M. Anas** — 24L-3004
- **M. Babar Shahzad** — 24L-3047

---

## Project Overview

PakAgriVision is an AI-powered web application that detects plant diseases from leaf images and provides actionable treatment recommendations. Farmers can upload a photo of a diseased leaf and instantly receive:

1. **Predicted disease** (with confidence score)
2. **Grad-CAM heatmap** highlighting the regions the model focused on
3. **Actionable treatment recommendations** — specific fungicides, dosage, organic alternatives, and preventive measures tailored to the diagnosed disease

The project originated from the PlantVillage dataset (~54,000 raw images across 38 classes). To address severe class imbalance, corrupt files, and redundant leaf captures present in the raw benchmark, we cleaned the dataset, eliminated duplicates and corrupted images, and capped each class at 1,000 clean images. This produced our curated, balanced dataset of **~32k images (31,694 images)** across all 38 classes. We no longer use Kaggle; the cleaned dataset and real-world field set are distributed directly as `data.zip` on Google Drive.

---

## Directory Structure

The repository follows the structure defined for Deliverable 1 (Problem Definition & Data Foundations). Key directories:

```
PakAgriVision/
├── README.md                  # This file — setup + how to reproduce
├── requirements.txt           # Pinned dependencies
├── .gitignore                # Git-ignored patterns (data/models/secrets)
│
├── data_scripts/              # Phase 2 scripts mandated by Deliverable 1 spec
│   ├── README.md              # Purpose & mapping of data pipeline scripts
│   ├── download_data.md       # Exact download commands
│   ├── clean_and_cap_dataset.py # A1: dedupe, quality filter & 1,000/class cap
│   ├── check_dataset.py       # A1: quality scan → dedupe_report.csv
│   ├── make_splits.py         # A2: writes splits/*.csv + class_weights.json
│   ├── make_field_splits.py   # B4: field_test / field_adapt
│   ├── validate_treatments.py # B3: checks treatments.json
│   └── visualize_augmentations.py # A4: augmentation sanity check
│
├── data/                      # NEVER commit large data/folders
│   ├── README.md              # Dataset download info, version, licence
│   ├── .gitkeep              # Keeps folder in git
│   ├── raw/
│   │   ├── plantvillage/     # PlantVillage images (color/ only; ~54k images; IGNORED)
│   │   └── field/            # Real field images + field_labels.csv (IGNORED images)
│   └── processed/            # Cached/resized tensors (IGNORED)
│
├── splits/                    # Small text files — COMMITSED (identical splits for all)
│   ├── class_names.json       # Ordered list of 38 class names → index
│   ├── train.csv              # path,label,label_idx (70% train)
│   ├── val.csv                # path,label,label_idx (15% val)
│   ├── test.csv               # path,label,label_idx (15% test)
│   ├── field_test.csv         # Field test split (~2/3)
│   ├── field_adapt.csv        # Field adaptation split (~1/3)
│   ├── field_split_stats.csv  # Per-class field split counts
│   ├── class_weights.json     # From TRAIN split only
│   └── dedupe_report.csv      # Duplicate groups found
│
├── field_meta/                # Field-image metadata (COMMITTED; images stay in data/)
│   ├── field_labels.csv       # filename,label,source,url,licence,notes
│   └── class_mapping.csv      # source class name → PlantVillage class name
│
├── knowledge_base/            # COMMITTED
│   ├── treatments.json        # 38 entries (curated JSON for treatment info)
│   ├── schema.json            # JSON schema / required fields
│   └── SOURCES.md            # Every citation (numbered list)
│
├── src/                       # Reusable code
│   ├── __init__.py
│   ├── config.py              # Paths, seed=42, image size, ImageNet mean/std
│   └── data/
│       ├── __init__.py
│       ├── dataset.py         # PlantDataset (torch Dataset)
│       ├── transforms.py      # train / eval transforms
│       ├── splits.py          # Split logic
│       └── quality.py         # Corrupt / duplicate detection
│

├── notebooks/                 # EDA notebook (outputs stripped)
│   └── 01_eda.ipynb
│
├── reports/                   # Figures used in LaTeX report (COMMITTED)
│   └── figures/
│       ├── eda_class_distribution.png
│       ├── eda_sample_grid.png
│       ├── eda_image_sizes.png
│       ├── aug_examples.png
│       └── field_vs_lab_examples.png
│
├── tests/                     # Anas's tests
│   └── test_data.py
│
├── docs/                      # LaTeX reports (COMMITTED)
│   ├── proposal/              # Proposal .tex + compiled .pdf
│   │   ├── proposal.tex
│   │   └── proposal.pdf
│   └── report/
│       ├── main.tex           # Main report LaTeX file
│       ├── references.bib
│       └── sections/
│           └── 02_data.tex    # Phase 2 section (dataset, preprocessing, splits, field data, KB, challenges)
│
├── models/                    # Trained weights — IGNORED (hosted on Hugging Face later)
│   └── .gitkeep
│
└── app/                       # Streamlit app (Phase 5)
    └── .gitkeep
```

### Placement Rules (quick lookup)

| Thing                              | Goes in                     | In Git? |
|--------------------------------------|-----------------------------|---------|
| Raw PlantVillage images              | `data/raw/plantvillage/`    | No      |
| Field images                         | `data/raw/field/`           | No      |
| Field labels / licences              | `field_meta/field_labels.csv` | Yes    |
| Split CSVs, class weights, class names | `splits/`                  | Yes     |
| Treatment JSON + sources             | `knowledge_base/`           | Yes     |
| Reusable code                        | `src/`                      | Yes     |
| One-off / data scripts              | `data_scripts/`             | Yes     |
| Report figures (small PNG/PDF)       | `reports/figures/`          | Yes     |
| Trained weights (.pth/.pt/.ckpt)     | `models/`                   | No      |
| Compiled report/proposal PDFs        | `docs/…`                    | Yes     |
| LaTeX junk (.aux .log .out …)        | beside the `.tex`           | No      |
| Kaggle API key (no longer needed)    | Not used (dataset hosted on Drive) | N/A     |

---

## Setup & How to Run

### 1. Prerequisites

- **Python** 3.9 or 3.10 recommended
- **Git** for version control

### 2. Clone the repository

```bash
git clone https://github.com/ErenYeagerJinko/PakAgriVision.git
cd PakAgriVision
```

### 3. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

**Pinned dependencies** (from `requirements.txt`):
```text
torch>=2.0.0
torchvision>=0.15.0
pillow>=10.0.0
imagehash>=4.3.0
numpy>=2.0.0
pandas>=2.2.0
matplotlib>=3.8.0
jsonschema>=4.20.0
pytest>=8.0.0
jupyter>=1.0.0
nbstripout>=0.7.0
```

> **Windows note**: If the DataLoader hangs, use `num_workers=0`.

### 5. Download the Curated Dataset (`data.zip`)

> **Dataset Curation & Kaggle Note:** Kaggle was used solely for the initial raw image intake (~54k images). We do **not** use raw Kaggle downloads for running or reproducing the project because the uncurated benchmark suffered from corrupted files, duplicate photographs, and severe class imbalance (over 5,500 images in dominant classes vs. 152 in minority classes). 
> 
> We ran our automated quality and deduplication pipeline (`src/data/quality.py`, `data_scripts/clean_and_cap_dataset.py`), pruned corrupted files, removed exact duplicates (MD5) and near-duplicates (pHash), and capped dominant classes at a maximum of 1,000 clean images per class. This yielded our balanced, high-integrity dataset of **~32k images (31,694 verified clean images)** across all 38 classes.

**Download our pre-packaged `data.zip` archive directly from Google Drive:**
- **Google Drive Link**: [Download data.zip](https://drive.google.com/file/d/1OE17MYZqriiP9mBdCat_JglMxx_7MY1V/view?usp=drivesdk)
- **Archive Contents**:
  - `data/raw/plantvillage/color/`: 31,694 clean, balanced images across all 38 classes (max 1,000 images/class).
  - `data/raw/field/`: 150 real-world field robustness images across 28 matching classes.
- **Extraction**: Extract `data.zip` into `data/raw/` so that `data/raw/plantvillage/color/` and `data/raw/field/` are populated.

### 6. Verify the dataset

```bash
# Check that data/ structure is correct
git status --short   # data/ and models/ must NOT appear

# Verify class names
python -c "import json; print(json.load(open('splits/class_names.json'))[:5])"
```

### 7. Run the EDA notebook (optional)

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 8. Validate the treatment knowledge base

```bash
python data_scripts/validate_treatments.py
```

### 9. Build the field splits

```bash
python data_scripts/make_field_splits.py
```

Reads `field_meta/field_labels.csv` and writes `splits/field_test.csv` (~2/3),
`splits/field_adapt.csv` (~1/3) and `splits/field_split_stats.csv`, stratified by
class with seed 42. The committed CSVs are the fixed split; re-run only if the
labels change.

> **Rule:** `field_test.csv` is never used for training, tuning or threshold
> selection. Only `field_adapt.csv` may be used for the fine-tuning mitigation.

### 10. Run the tests

```bash
pytest -q
```

### 11. Compile the LaTeX report

```bash
# From the docs/report/ directory
cd docs/report
pdflatex main.tex
# or use overleaf/xelatex as preferred
```

### 12. Launch the Streamlit app (Phase 5)

> **Note:** Streamlit is reserved for Phase 5 (Interactive Application Deployment) and is not required for Deliverable 1.

```bash
# Optional (Phase 5 only):
# pip install streamlit
streamlit run app/
```

---

## Team Contributions & Task Distribution

| Member        | ID          | Primary Responsibilities                                                                 |
|---------------|-------------|------------------------------------------------------------------------------------------|
| **Abdul Ahad**   | 24L-3029   | L1: Repo setup, environment, README, .gitignore; L2: Download & verify PlantVillage; L4: Integration, final checklist, compile report |
| **M. Anas**     | 24L-3004    | A1: Data quality checks (corrupt, duplicates, near-duplicates); A2: Stratified split + CSVs + class weights; A3: Dataset class, transforms, DataLoader; A4: Unit tests, augmentation visual sanity check |
| **M. Babar Shahzad** | 24L-3047 | B1: Field-image collection (≥150) + labelling + licence tracking; B2: Class-name mapping; B3: Treatment KB schema + validation; B4: Field-set split + statistics |

---

## Phase-by-Phase Status

| Phase | Task                                                                                     | Status |
|-------|------------------------------------------------------------------------------------------|--------|
| 1     | Setup, dataset download, EDA                                                             | In progress |
| 2     | Preprocessing, augmentation pipeline                                                     | Planned |
| 3     | Baseline ResNet50 training                                                               | Planned |
| 4     | Fine-tuning + hyperparameter search                                                    | Planned |
| 5     | Grad-CAM integration                                                                     | Planned |
| 6     | Treatment engine (JSON + UI cards)                                                       | Planned |
| 7     | Robustness testing on field images                                                     | Planned |
| 8     | Streamlit UI + deployment                                                                | Planned |
| 9     | Evaluation, bias check, LaTeX report                                                     | Planned |
| —     | Buffer for debugging                                                                     | Planned |

---

## Python / Environment Requirements

- **PyTorch** & **torchvision** (CPU or GPU install — check [pytorch.org](https://pytorch.org/) for the correct install command for your setup)
- **Pillow** for image loading and validation
- **imagehash** for perceptual hashing (duplicate detection)
- **NumPy** & **Pandas** for numerical computing and stratified splits
- **Matplotlib** for publication-quality EDA visualisations
- **jsonschema** for treatment KB validation
- **pytest** for automated unit test suites
- **jupyter** for exploratory data analysis
- **nbstripout** to strip notebook outputs before committing

All versions are pinned in `requirements.txt`. After `pip install -r requirements.txt`, run `pip freeze > requirements.txt` to capture the exact installed versions.

---

## Key Project Goals (from index.html)

1. **Classifier**: Fine-tuned ResNet50 / EfficientNet-B0 on PlantVillage (38 disease classes)
2. **Explainability**: Grad-CAM heatmaps integrated into Streamlit UI
3. **Treatment Engine**: Curated JSON mapping each of 38 classes to chemical/organic treatment, immediate action, and prevention measures
4. **Robustness Testing**: Lab vs. field image performance gap analysis (≥150 real field images)
5. **Deployment**: Streamlit Community Cloud + model on Hugging Face Hub

---
