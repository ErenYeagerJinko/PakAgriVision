Curated PlantVillage Benchmark & Field Dataset
==============================================

- **Provenance**: Derived from the public PlantVillage benchmark (Hughes & Salathé, 2015).
  Kaggle was utilized solely for the initial raw data ingestion (~54k candidate images).
- **Why Kaggle is No Longer Used**: Raw benchmark downloads contained corrupted JPEG bitstreams,
  exact byte duplicate photos, near-duplicate captures of the same leaf under slight camera rotations,
  and severe class imbalance (36:1 ratio, up to 5,507 images in dominant classes vs. 152 in minority classes).
- **Curated Dataset Scale**: **31,694 clean unique images (~32k images)** across 38 classes.
  Dominant classes are capped at a maximum ceiling of **1,000 images per class**, ensuring high data
  quality and a balanced optimization landscape.
- **License**: CC0 Public Domain / Open Educational License.
- **Expected Folder Layout**:
  - `data/raw/plantvillage/color/<class_name>/<filename>.JPG` (31,694 curated images)
  - `data/raw/field/<class_name>/<filename>.jpg` (150 real-world field robustness images)
- **Class Count**: 38 classes (see `splits/class_names.json`)
- **Primary Download**: Hosted directly via Google Drive as `data.zip`:
  ```text
  https://drive.google.com/file/d/1OE17MYZqriiP9mBdCat_JglMxx_7MY1V/view?usp=drivesdk
  ```
  Extract directly into `data/raw/` to populate both `data/raw/plantvillage/color/` and `data/raw/field/`.

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

### Google Drive Link (Unified Dataset Archive)
The unified dataset archive contains both the verified PlantVillage color images (all 38 classes) and the real-world field robustness images (28 classes).

```
DATASET_DRIVE_URL=https://drive.google.com/file/d/1OE17MYZqriiP9mBdCat_JglMxx_7MY1V/view?usp=drivesdk
```

To restore locally:
1. Download the archive from the link above.
2. Extract the archive directly into `data/raw/` so that `data/raw/plantvillage/color/` and `data/raw/field/` are populated.
3. Class names and file mappings strictly match `splits/class_names.json` and `field_meta/field_labels.csv`.

The zip archive itself is **not** committed to Git.
