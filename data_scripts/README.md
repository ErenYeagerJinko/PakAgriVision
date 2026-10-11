# Phase 2: Data Scripts (`data_scripts/`)

This directory is designated for all Phase 2 data processing, acquisition, cleaning, and preparation scripts as required by the **Deliverable 1 specification** (*"A folder named /data scripts containing all Phase 2 code"*).

## Planned Scripts

| Script | Purpose | Deliverable 1 Requirement |
| :--- | :--- | :--- |
| `01_download_dataset.py` | Automated dataset retrieval & unzipping | § 2 Data Collection |
| `02_clean_and_verify.py` | Corrupt image filtering, resolution & duplicate verification | § 2 Data Cleaning |
| `03_create_splits.py` | Stratified train/val/test splitting & class weights computation | § 2 Splits & Leak Prevention |
| `04_validate_treatments.py` | JSON Schema validation for treatment knowledge base | § 2 Knowledge Base Quality |
