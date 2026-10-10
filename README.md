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

The project builds on the [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) (~54,000 labeled leaf images across 38 disease classes) and incorporates explainability and a treatment engine as key differentiators.

---

## Directory Structure

The repository follows the structure defined in `phase2_plan.html`. Key directories:

```
PakAgriVision/
├── README.md                  # This file — setup + how to reproduce
├── requirements.txt           # Pinned dependencies
├── .gitignore                # Git-ignored patterns (data/models/secrets)
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
├── scripts/                   # One-off scripts
│   ├── download_data.md       # Exact download commands
│   ├── check_dataset.py       # A1: quality scan → dedupe_report.csv
│   ├── make_splits.py         # A2: writes splits/*.csv + class_weights.json
│   ├── make_field_splits.py   # B4: field_test / field_adapt
│   └── validate_treatments.py # B3: checks treatments.json
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
| One-off scripts                      | `scripts/`                  | Yes     |
| Report figures (small PNG/PDF)       | `reports/figures/`          | Yes     |
| Trained weights (.pth/.pt/.ckpt)     | `models/`                   | No      |
| Compiled report/proposal PDFs        | `docs/…`                    | Yes     |
| LaTeX junk (.aux .log .out …)        | beside the `.tex`           | No      |
| Kaggle API key (.env, kaggle.json)   | `~/.kaggle/` / project root | **NEVER** |

---

## Setup & How to Run

### 1. Prerequisites

- **Python** 3.9 or 3.10 recommended
- **Git** for version control

### 2. Clone the repository

```bash
git clone https://github.com/your-username/PakAgriVision.git
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
```
torch
torchvision
numpy
pandas
scikit-learn
matplotlib
seaborn
pillow
imagehash
tqdm
pyyaml
jsonschema
jupyter
pytest
nbstripout
kaggle
```

> **Windows note**: If the DataLoader hangs, use `num_workers=0`.

### 5. Download the dataset

> You need a Kaggle API token to download PlantVillage automatically. If you don't have one:

1. Go to [Kaggle Account → Settings → Create New API Token](https://www.kaggle.com/settings/token)
2. This creates `~/.kaggle/kaggle.json` (never commit this file — it's in `.gitignore`)
3. Run the download command:

```bash
kaggle datasets download -d abdallahalidev/plantvillage-dataset -p data/raw --unzip
```

If Kaggle fails, manually download from the URL above and unzip into `data/raw/`.

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
python scripts/validate_treatments.py
```

### 9. Run the tests

```bash
pytest -q
```

### 10. Compile the LaTeX report

```bash
# From the docs/report/ directory
cd docs/report
pdflatex main.tex
# or use overleaf/xelatex as preferred
```

### 11. Launch the Streamlit app (Phase 5)

```bash
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

- **PyTorch** (CPU or GPU install — check [pytorch.org](https://pytorch.org/) for the correct install command for your setup)
- **torchvision** (comes with PyTorch)
- **NumPy** & **Pandas** for data manipulation
- **scikit-learn** for stratified splits and class weights
- **Matplotlib** & **Seaborn** for EDA visualisations
- **Pillow** for image handling
- **imagehash** for perceptual hashing (duplicate detection)
- **tqdm** for progress bars
- **PyYAML** for config file handling
- **jsonschema** for treatment KB validation
- **pytest** for unit tests
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
