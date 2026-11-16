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
│   ├── download_weights.py    # Downloads & extracts baseline folder to runs/
│   ├── download_data.py       # Downloads & extracts curated dataset to data/raw/
│   └── download_field_images.py # Downloads & extracts field images to data/raw/field/
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
│   └── report/                # Final Deliverable 3 LaTeX report & compiled main.pdf
│
└── app/                       # Phase 5 Web Application & REST API
    ├── app.py                 # Flask server & Two-Stage inference engine
    └── templates/             # Jinja2 dashboard UI (index.html)
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

### 5. Download Datasets

> **Dataset Curation & Kaggle Note:** Kaggle was used solely for the initial raw image intake (~54k images). We do **not** use raw Kaggle downloads for running or reproducing the project because the uncurated benchmark suffered from corrupted files, duplicate photographs, and severe class imbalance. 
> 
> We ran our automated quality and deduplication pipeline (`src/data/quality.py`, `data_scripts/clean_and_cap_dataset.py`), pruned corrupted files, removed exact duplicates (MD5) and near-duplicates (pHash), and capped dominant classes at a maximum of 1,000 clean images per class. This yielded our balanced, high-integrity dataset of **~32k images (31,694 verified clean images)** across all 38 classes.

**Download dataset archives from Google Drive:**
- **PlantVillage (Curated Lab Benchmark)**: [Download data.zip](https://drive.google.com/file/d/1OE17MYZqriiP9mBdCat_JglMxx_7MY1V/view?usp=drivesdk) (~1.0 GB): 31,694 clean, balanced images across all 38 classes (max 1,000 images/class). Extracted into `data/raw/plantvillage/color/<class>/`.
- **Field Images (Real-World Foliar Crops — 8,418 Images)**: [Download Field Images Archive](https://drive.google.com/file/d/1cKcUttKrrHHJ5shimKVNdLOIkIsLj3-M/view?usp=sharing) (~410 MB): **8,418 high-resolution leaf crop images across 28 agricultural classes**, curated directly from in-the-wild PlantDoc field captures to train and benchmark field robustness. Extracted into `data/raw/field_cropped/` and `data/raw/field/`.

**To download datasets automatically via scripts:**
```bash
# Download and extract the 8,418 real-world field leaf images:
python scripts/download_field_images.py

# Download the curated PlantVillage lab dataset (~32k images):
python scripts/download_data.py
```

---

### 6. Download Model Weights & Checkpoints

All model checkpoints and run artifacts are hosted on Google Drive (never committed directly to Git):

- **Baseline Weights (ResNet-50 Day 2)**: [Download Baseline Weights](https://drive.google.com/file/d/1R1qFkk4v3-5dzNFdRMxG7-gZXFwyo8Cv/view?usp=sharing)
  - *Contents*: `runs/baseline/` (`best.pt`, `last.pt`, `history.csv`, `summary.json`, `config.yaml`).
- **Field Cropped Weights (ResNet-50 8.4k Leaves)**: [Download Field Cropped Weights](https://drive.google.com/file/d/14ygSAzCwERyvAMALDSy3i-G1oXwOnNri/view?usp=sharing)
  - *Contents*: `runs/field_cropped/` (`best.pt`, `last.pt`, `history.csv`, `summary.json`, `config.yaml`).
- **YOLOv8n Leaf Detector Weights**: [Download YOLO Weights](https://drive.google.com/file/d/10fqb4sptHYqq5Y83Atut6qQivv-tg063/view?usp=sharing)
  - *Contents*: `models/yolo/leaf_detector.pt` (~6.5 MB), trained on PlantDoc bounding boxes for autonomous leaf localization.

**To download and extract weights automatically via script:**
```bash
# Download ALL 3 model weights at once (baseline, field_cropped, and yolo):
python scripts/download_weights.py --target all

# Or download individual weights:
python scripts/download_weights.py --target field_cropped
python scripts/download_weights.py --target baseline
python scripts/download_weights.py --target yolo
```

---

## Model Performance & Domain Adaptation Progression

### 1. Laboratory Benchmark (PlantVillage 38 Classes)
| Metric | Baseline Result |
|---|---|
| **Architecture** | ResNet-50 (ImageNet pretrained + Dropout 0.3 Head) |
| **Best Val Accuracy** | **99.64%** (Epoch 14) |
| **Best Val Macro-F1** | **0.9961** |
| **Best Val Loss** | **0.0169** |
| **Final Test Accuracy** | **99.71%** (4,768 held-out lab test images) |
| **Weights Location** | `runs/baseline/best.pt` |

### 2. In-the-Wild Field Robustness Progression (PlantDoc Field Test Images)
| Model / Pipeline Stage | Training Domain | Top-1 Acc | Top-3 Acc | Macro Prec | Macro F1 | Macro AUC |
|---|---|---|---|---|---|---|
| **1. Day 2 Baseline** | 32k Lab-only (PlantVillage) | 24.73% | 46.24% | 0.2204 | 0.1970 | 0.8001 |
| **2. Few-Shot Adapted** | 57 Curated Field Samples | 33.33% | 55.91% | 0.3424 | 0.2938 | 0.8558 |
| **3. Full Field Scale** | ~2,000 Uncropped Field Images | 54.84% | 82.80% | 0.5829 | 0.5355 | 0.9524 |
| **4. Leaf-Cropped ResNet-50** | 8,418 Bounding-Box Leaf Crops | 53.76% | 78.49% | 0.6105 | 0.5025 | 0.9459 |
| **5. Two-Stage Cascade** | YOLOv8n Crop $\rightarrow$ ResNet-50 Cropped | 67.74% | 90.32% | 0.5980 | 0.5420 | 0.9610 |
| **6. Two-Stage + Crop Prior** | **YOLO Crop + Active Crop Context** | **80.65%** | **96.77%** | **0.8240** | **0.7910** | **0.9850** |

---

### 7. Launch Phase 5 Web Application

Launch the full-featured Flask AI web application:
```bash
python app/app.py
```
Open **`http://127.0.0.1:5000`** in your browser. The application features:
- **Two-Stage Cascade Inference**: YOLOv8n automatically detects and crops the leaf to eliminate background clutter, then ResNet-50 classifies pathology.
- **Explainability**: Interactive Layer 4 Grad-CAM activation heatmap overlay with live alpha blending slider.
- **Bilingual Actionable Advisories**: Complete organic and chemical treatment advice translated into Urdu and English.
- **Crop Context Filtering**: Optional crop filter toggle chips to refine diagnostic priors.
- **1-Click Viva Presets**: Instant testing on pre-loaded diseased and healthy field specimens.

### 8. Run the test suite

```bash
pytest -v
```

### 9. Compile the LaTeX report

```bash
python scripts/compile_report.py
# Or compile manually:
cd docs/report && pdflatex main.tex
```

---

## Team Contributions & Task Distribution (Phases 1 through 6)

| Member | ID | Primary Responsibilities across Deliverables 1, 2, and 3 |
|---|---|---|
| **Abdul Ahad** (Lead) | 24L-3029 | **D3:** Web application backend & REST API (`app/app.py`), single-page responsive UI (`app/templates/index.html`), Two-Stage Cascade integration with YOLO leaf cropping and crop-filtering, Grad-CAM interactive visualizer, app tests (`tests/test_app.py`), Deliverable 3 report compilation. <br>**D1-D2:** Model architectures (`src/models/`), losses (`src/losses.py`), optimizers (`src/optim.py`), experiment configs, baseline training, weights hosting. |
| **M. Anas** | 24L-3004 | **D3:** ResNet-50 field-cropped model fine-tuning (`runs/field_cropped/best.pt`), dataset manifest generation for 8,418 leaf crops (`splits/field_cropped_train.csv`), test suite verification. <br>**D1-D2:** Training engine (`src/engine.py`), CLI orchestrator (`src/train.py`), hyperparameter tuning (`src/tune.py`), plotting (`src/plotting.py`), dataset loader & transforms (`src/data/`). |
| **M. Babar Shahzad** | 24L-3047 | **D3:** YOLOv8n leaf detector training on PlantDoc annotations (`models/yolo/leaf_detector.pt`), domain adaptation benchmarking across all 6 model iterations on 93 field test images, latency profiling, bilingual Urdu translation dictionary. <br>**D1-D2:** Evaluation metrics suite (`src/metrics.py`), evaluation script (`src/evaluate.py`), error analysis, systematic bias check, field data collection, treatment knowledge base (`knowledge_base/`). |

---

## Key Project Deliverables Summary

1. **Two-Stage Cascade Classifier**: YOLOv8n leaf detector (100% field detection) + ResNet-50 cropped classifier achieving **80.65% Top-1** and **96.77% Top-3** field accuracy.
2. **Phase 5 Web Application**: Production-ready Flask REST API and responsive agricultural dashboard at `http://127.0.0.1:5000`.
3. **Explainability**: Real-time Layer 4 Grad-CAM activation overlays with interactive transparency control.
4. **Treatment Engine**: Curated 38-class treatment advisory in English and Urdu covering biological controls, chemical remedies, dosages, and safety withdrawal periods.
5. **Technical Report**: Comprehensive Deliverable 3 LaTeX report (`docs/report/main.pdf`) documenting the entire experimental progression and viva defense talking points.

---