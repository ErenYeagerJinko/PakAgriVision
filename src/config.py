from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_PV = DATA_DIR / "raw" / "plantvillage"
RAW_FIELD = DATA_DIR / "raw" / "field"
SPLITS_DIR = ROOT / "splits"
SEED = 42
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]