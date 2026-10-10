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

## Field Images
- **Location**: `data/raw/field/`
- **Rule**: One clearly visible diseased leaf, real background (soil, other
  leaves, hands), minimum ~224 px on the short side, no watermarks, no duplicates,
  no lab-style photos.
- **Count**: ≥150 images total (100 for testing + 50 for mitigation experiment)
- **Labels**: Recorded in `field_meta/field_labels.csv`
- **Google Drive Link**: Link to zipped field data will be shared in this README
  (zip itself not committed).