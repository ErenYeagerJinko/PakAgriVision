# Cloud GPU Training Guide (Google Colab & Kaggle)

This guide documents how to train the PakAgriVision models (Baseline, Tuning Trials, and Final Model) using free cloud GPU instances (Google Colab or Kaggle).

---

## 1. Environment & GPU Selection

* **Google Colab:**
  * Go to `Runtime` -> `Change runtime type` -> select `T4 GPU`.
  * Free instances provide ~15 GB VRAM and a 12-hour session window.
* **Kaggle Notebooks:**
  * Under notebook `Settings` (right sidebar), set `Accelerator` to `GPU T4 x2` or `GPU P100`.
  * Ensure `Internet` is toggled **ON** in the notebook settings.

---

## 2. Dataset Setup (via Google Drive)

We do **not** download raw Kaggle archives during training. The project uses our curated and balanced dataset (~32k images across 38 classes) distributed as `data.zip` on Google Drive.

In your notebook cell:

```bash
# 1. Clone repository
git clone https://github.com/ErenYeagerJinko/PakAgriVision.git
cd PakAgriVision

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data.zip directly from Google Drive using gdown
pip install gdown
gdown https://drive.google.com/uc?id=1OE17MYZqriiP9mBdCat_JglMxx_7MY1V -O data.zip

# 4. Extract into data/raw/
mkdir -p data/raw
unzip -q data.zip -d data/raw/
rm data.zip
```

Verify directory structure:
* `data/raw/plantvillage/color/` (31,694 images)
* `data/raw/field/` (150 images)

---

## 3. Running Training

### Baseline Training (Day 2)
```bash
python -m src.train --config configs/baseline.yaml
```

### Resuming After a Disconnect / Session Timeout
Free cloud instances may timeout after idle periods or reach their 12-hour maximum runtime. The training script automatically saves `best.pt` and `last.pt` at each epoch in the run directory.

If your session is interrupted, re-launch the notebook and resume from the latest saved state:
```bash
python -m src.train --config configs/baseline.yaml --resume runs/baseline/last.pt
```

---

## 4. Exporting Weights & Artifacts to Google Drive

Never commit model weights (`.pt`/`.pth`) to Git. After training concludes:

1. Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```
2. Copy the checkpoint state dictionary (~90 MB) to your Drive folder:
```bash
cp runs/baseline/best.pt /content/drive/MyDrive/PakAgriVision_Weights/resnet50_baseline.pt
```
3. Calculate the SHA-256 checksum for the README:
```bash
sha256sum runs/baseline/best.pt
```

