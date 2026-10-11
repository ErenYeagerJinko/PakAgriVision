PlantVillage Dataset
====================

- **Source**: Kaggle dataset `abdallahalidev/plantvillage-dataset`
- **Kaggle URL**: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
- **Original Paper**: Hughes, D.J., Salathé, M.: An open access repository of images
  depicting plant diseases and healthy plant leaves to facilitate accurate
  automated image-based plant disease diagnostics. arXiv:1511.08060 (2015)
- **License**: The dataset images are in the public domain (CC0) or under
  attribution requirements per the original paper. Check individual class folders
  for details.
- **Version Used**: Original images (not pre-augmented). Pre-augmented copies
  (~87k images with their own train/valid split) are avoided because augmented
  duplicates can leak across splits.
- **Expected Folder Layout**: `data/raw/plantvillage/color/` (only `color/` used
  for training; other subfolders `grayscale`, `segmented` are ignored)
- **Class Count**: 38 classes (see `splits/class_names.json`)
- **Image Count**: ~54,300 images in `color/` folder (original count)
- **Download Command**:
  ```
  kaggle datasets download -d abdallahalidev/plantvillage-dataset -p data/raw --unzip
  ```
  If Kaggle fails, download manually from the URL above and unzip into
  `data/raw/`.

## Class Names (ordered)
The ordered list of 38 class names is saved in `splits/class_names.json`. This
file fixes label index order for the entire project. All CSVs and model outputs
must reference classes by their index in this file.

## Field Images (robustness set)

Field images are **not** committed to Git (see `.gitignore`). Metadata is committed
under `field_meta/`.

### Goals
- Collect **≥150** real-world leaf images total (~100 for `field_test`, ~50 for `field_adapt`).
- Cover as many of the 38 PlantVillage classes as practical (aim for ≥3–5 images per class when available).
- Use only classes that exist in `splits/class_names.json`.

### Allowed sources
1. **PlantDoc** — https://github.com/pratikkayal/PlantDoc-Dataset (check repo licence/terms; commonly cited as CC-BY-4.0)
2. **iNaturalist** — CC0 or CC-BY observations only; record the observation URL
3. **Wikimedia Commons** — public domain / CC0 preferred; record the file URL
4. **Own phone photos** — preferred when possible (`source=own_photo`, `licence=own`)

### Image rules
- One clearly visible leaf (diseased or healthy) with a **real field background** (soil, other leaves, hands, sky).
- Minimum ~224 px on the short side.
- No watermarks, no near-duplicates, no lab-style plain backgrounds.
- Prefer natural lighting; avoid heavy filters.

### Folder layout (local only)
```
data/raw/field/
  <PlantVillage_class_name>/
    <descriptive_filename>.jpg
```
Example:
```
data/raw/field/Tomato___Early_blight/plantdoc_tomato_early_blight_001.jpg
```

### Metadata (`field_meta/field_labels.csv`)
Committed CSV with one row per image. Columns:

| Column   | Meaning |
|----------|---------|
| filename | Basename only, prefixed with the source (e.g. `plantdoc_tomato_early_blight_001.jpg`) |
| label    | Exact PlantVillage class string from `splits/class_names.json` |
| source   | `PlantDoc` / `iNaturalist` / `Wikimedia` / `own_photo` |
| url      | Page or file URL; use `local` for own photos |
| licence  | e.g. `CC-BY-4.0`, `CC0`, `own` |
| notes    | Short free text (location, doubt, crop stage, etc.) |

Class-name mapping from external datasets → PlantVillage names lives in
`field_meta/class_mapping.csv`. Classes with no PV counterpart are excluded from
the lab-vs-field comparison (documented in that file).

### Label QA
A second team member must double-check every label before the field split is
finalized. Doubtful images are removed, not forced into a class.

### Google Drive link (zip of `data/raw/field/`)
Current set: **150 images across 28 classes**, all from PlantDoc (66 MB zip).

```
FIELD_IMAGES_DRIVE_URL=https://drive.google.com/file/d/1lNE3mqhb3yxPV_i5Sz3lptrWv-bkW9eo/view?usp=sharing
```

To restore locally, download `field_images.zip` and extract it into `data/raw/field/`
so that each class folder sits directly under it. File names must match
`field_meta/field_labels.csv`.

The zip itself is **not** committed.
