# Phase 2: Data Scripts (`data_scripts/`)

This directory is designated for all Phase 2 data processing, acquisition, cleaning, and preparation scripts as required by the **Deliverable 1 specification** (*"A folder named /data scripts containing all Phase 2 code"*).

## Phase 2 Scripts Mapping

| Script | Purpose | Deliverable 1 Requirement |
| :--- | :--- | :--- |
| `download_data.md` | Download documentation & automated retrieval commands | § 2 Data Collection |
| `clean_and_cap_dataset.py` | Binary corrupt image filtering, MD5 exact dedupe, pHash near-dedupe & 1,000/class cap | § 2 Data Cleaning |
| `check_dataset.py` | Quality inspection CLI → `splits/dedupe_report.csv` | § 2 Data Cleaning |
| `make_splits.py` | Stratified 70/15/15 train/val/test splitting & class weights computation | § 2 Splits & Leak Prevention |
| `make_field_splits.py` | Real-world field image partition (`field_test` vs `field_adapt`) | § 2 Out-of-Domain Robustness |
| `validate_treatments.py` | JSON Schema validation for 38-class treatment knowledge base | § 2 Knowledge Base Quality |
| `visualize_augmentations.py` | Preprocessing and augmentation visual sanity check grid | § 2 Feature Strategy |
