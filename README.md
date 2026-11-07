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

The repository follows the structure defined for Deliverable 1 and Deliverable 2. Key directories:

```
PakAgriVision/
├── README.md                  # This file — setup + how to reproduce
├── requirements.txt           # Pinned dependencies
├── .gitignore                # Git-ignored patterns (data/models/runs/secrets)
│
├── configs/                   # Experiment YAML configurations (COMMITTED)
│   ├── base.yaml              # Base hyperparameter definitions
│   └── baseline.yaml          # Day 2 baseline configuration
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
├── scripts/                   # Utility and download automation scripts
│   └── download_weights.py    # Downloads & extracts baseline folder to runs/
│
├── data/                      # NEVER commit large data/folders
│   ├── README.md              # Dataset download info, version, licence
│   ├── .gitkeep              # Keeps folder in git
│   ├── raw/
│   │   ├── plantvillage/     # PlantVillage images (color/ only; ~32k curated images; IGNORED)
│   │   └── field/            # Real field images + field_labels.csv (IGNORED images)
│   └── processed/            # Cached/resized tensors (IGNORED)
│
├── splits/                    # Small text files — COMMITTED (identical splits for all)
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
│   ├── losses.py              # Cross-entropy with class weights & label smoothing
│   ├── optim.py               # AdamW / SGD + cosine warmup scheduler
│   ├── models/                # Model architectures (ResNet-50 / EfficientNet-B0)
│   │   ├── __init__.py
│   │   └── classifier.py
│   └── data/
│       ├── __init__.py
│       ├── dataset.py         # PlantDataset (torch Dataset)
│       ├── transforms.py      # train / eval transforms
│       ├── splits.py          # Split logic
│       └── quality.py         # Corrupt / duplicate detection
│
├── runs/                      # Training checkpoints & full logs — IGNORED (hosted on Google Drive)
│   └── baseline/              # best.pt, last.pt, history.csv, summary.json
│
├── results/                   # Lightweight metrics & curves — COMMITTED
│   └── baseline/              # history.csv, summary.json, config.yaml
│
├── notebooks/                 # Notebooks & cloud guides
│   ├── 01_eda.ipynb
│   └── cloud_training_guide.md
│
├── reports/                   # Figures used in LaTeX report (COMMITTED)
│   └── figures/
│
├── tests/                     # Unit test suites
│   ├── test_data.py
│   ├── test_dataset.py
│   ├── test_data_quality.py
│   ├── test_splits.py
│   ├── test_treatments.py
│   └── test_model.py
│
├── docs/                      # LaTeX reports (COMMITTED)
│   ├── proposal/              # Proposal .tex + compiled .pdf
│   └── report/                # Main report LaTeX file
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
| Trained weights (.pt/.pth)           | `runs/`                     | No (Hosted on Google Drive) |
| Metrics summaries & search tables    | `results/`                  | Yes     |
| Compiled report/proposal PDFs        | `docs/…`                    | Yes     |
| LaTeX junk (.aux .log .out …)        | beside the `.tex`           | No      |
| Kaggle API key (no longer needed)    | Not used (dataset hosted on Drive) | N/A     |

---

## Setup & How to Run

### 1. Prerequisites

- **Python** 3.9, 3.10, or 3.11+
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

### 5. Download the Curated Dataset (`data.zip`)

> **Dataset Curation & Kaggle Note:** Kaggle was used solely for the initial raw image intake (~54k images). We do **not** use raw Kaggle downloads for running or reproducing the project because the uncurated benchmark suffered from corrupted files, duplicate photographs, and severe class imbalance. 
> 
> We ran our automated quality and deduplication pipeline (`src/data/quality.py`, `data_scripts/clean_and_cap_dataset.py`), pruned corrupted files, removed exact duplicates (MD5) and near-duplicates (pHash), and capped dominant classes at a maximum of 1,000 clean images per class. This yielded our balanced, high-integrity dataset of **~32k images (31,694 verified clean images)** across all 38 classes.

**Download both archives from Google Drive:**
- **PlantVillage (curated)**: [Download data.zip](https://drive.google.com/file/d/1OE17MYZqriiP9mBdCat_JglMxx_7MY1V/view?usp=drivesdk) (~1.0 GB): 31,694 clean, balanced images across all 38 classes (max 1,000 images/class). The archive's top-level folder is `color/`, so extract it into `data/raw/plantvillage/` to get `data/raw/plantvillage/color/<class>/`.
- **Field set**: [Download field_images_v2.zip](https://drive.google.com/file/d/1d_OAuW6XipfGwEzTa9-4zm6nSrnJMKad/view?usp=sharing) (72 MB): 150 real-world PlantDoc images across 27 classes, exactly the images listed in `field_meta/field_labels.csv` (11 invalid images were replaced in label review). The top-level folder is `field/`, so extract it into `data/raw/` to get `data/raw/field/<class>/`.
- **Check**: every path in `splits/val.csv`, `splits/test.csv` and `splits/field_test.csv` should then exist.

### 6. Download Baseline Artifacts & Weights (`runs/baseline`)

The complete baseline run artifacts (weights, checkpoints, metrics, and configs) are packaged directly as `baseline/` in Google Drive (never committed to Git):

- **Google Drive Link**: [Download Baseline Artifacts Archive](https://drive.google.com/file/d/1RBab1_qShQGObGJtnefs0HlR-G-orM1g/view?usp=drive_link)
- **Archive Contents**:
  - `baseline/best.pt`: Winning ResNet-50 weights (~270 MB, 99.64% val accuracy)
  - `baseline/last.pt`: Final epoch checkpoint (~270 MB)
  - `baseline/history.csv`: Full epoch-by-epoch loss, accuracy, and F1 logs
  - `baseline/summary.json` & `baseline/config.yaml`: Run summary & config snapshot
- **Extraction**: Extract the downloaded archive directly into `runs/` so that `runs/baseline/` is populated.

**To download and extract automatically:**
```bash
python scripts/download_weights.py
```
*(Or download manually from the link above and extract the zip archive directly into `runs/`).*

---

## Baseline Model Performance (ResNet-50)

| Metric | Baseline Result |
|---|---|
| **Architecture** | ResNet-50 (ImageNet pretrained + Dropout 0.3 Head) |
| **Best Val Accuracy** | **99.64%** (Epoch 14) |
| **Best Val Macro-F1** | **0.9961** |
| **Best Val Loss** | **0.0169** |
| **Final Train Accuracy** | **99.81%** (Epoch 15) |
| **Optimizer & Schedule** | AdamW (LR=3e-4, WD=1e-4) + Cosine Annealing (1 epoch warm-up) |
| **Weights Location** | `runs/baseline/best.pt` |

---

### 7. Verify the dataset & environment

```bash
# Check that data/ and runs/ structures are correct and ignored
git status --short   # data/ and runs/ must NOT appear

# Verify class names
python -c "import json; print(json.load(open('splits/class_names.json'))[:5])"
```

### 8. Run the EDA notebook (optional)

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 9. Validate the treatment knowledge base

```bash
python data_scripts/validate_treatments.py
```

### 10. Build the field splits

```bash
python data_scripts/make_field_splits.py
```

### 11. Run the test suite

```bash
pytest -v
```

### 12. Compile the LaTeX report

```bash
# From the docs/report/ directory
cd docs/report
pdflatex main.tex
```

---

## Team Contributions & Task Distribution

| Member | ID | Primary Responsibilities |
|---|---|---|
| **Abdul Ahad** | 24L-3029 | Model architectures (`src/models/`), loss (`src/losses.py`), optimizer (`src/optim.py`), experiment configs (`configs/`), baseline training, weights hosting, `scripts/download_weights.py`, README |
| **M. Anas** | 24L-3004 | Training engine (`src/engine.py`), CLI orchestrator (`src/train.py`), hyperparameter tuning (`src/tune.py`), plotting (`src/plotting.py`), training curve analysis |
| **M. Babar Shahzad** | 24L-3047 | Evaluation metrics (`src/metrics.py`), evaluation script (`src/evaluate.py`), error analysis, bias check, test set evaluation |

---

## Key Project Goals

1. **Classifier**: Fine-tuned ResNet-50 / EfficientNet-B0 on PlantVillage (38 disease classes)
2. **Explainability**: Grad-CAM heatmaps integrated into Streamlit UI
3. **Treatment Engine**: Curated JSON mapping each of 38 classes to chemical/organic treatment, immediate action, and prevention measures
4. **Robustness Testing**: Lab vs. field image performance gap analysis (≥150 real field images)

---