"""PakAgriVision Flask dashboard and confidence-aware treatment REST API."""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import io
import json
import logging
import math
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from app.model_engine import ModelEngine, ModelUnavailable
from app.treatment_advisor import build_advisory
from knowledge_base.kb_service import KnowledgeBaseError, TreatmentKB

SPLITS_DIR = ROOT_DIR / "splits"
CLASS_NAMES_PATH = SPLITS_DIR / "class_names.json"
KB_PATH = ROOT_DIR / "knowledge_base/treatments.json"
CHECKPOINT_PATH = ROOT_DIR / "runs/baseline/best.pt"
RESULTS_DIR = ROOT_DIR / "results/baseline"
HISTORY_PATH = RESULTS_DIR / "history.csv"
TEST_METRICS_PATH = RESULTS_DIR / "test/metrics.json"
TEST_PER_CLASS_PATH = RESULTS_DIR / "test/per_class.csv"
SAMPLE_DIRS = [ROOT_DIR / "data/raw/field", ROOT_DIR / "data/raw/plantvillage/color"]

app = Flask(__name__, template_folder=str(ROOT_DIR / "app/templates"))
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


def load_history_csv():
    if not HISTORY_PATH.is_file():
        return []
    with HISTORY_PATH.open(encoding="utf-8") as f:
        return [{k: int(v) if k == "epoch" else float(v) for k, v in row.items()}
                for row in csv.DictReader(f)]


def load_test_metrics():
    # Missing benchmark files must not be replaced with invented measurements.
    if not TEST_METRICS_PATH.is_file():
        raise FileNotFoundError("Committed test metrics are missing")
    return json.loads(TEST_METRICS_PATH.read_text(encoding="utf-8"))


def format_class_name(raw_name):
    crop, _, disease = raw_name.partition("___")
    crop, disease = crop.replace("_", " "), disease.replace("_", " ")
    return crop, "Healthy Leaf (Normal)" if disease.lower() == "healthy" else disease


def is_healthy(raw_name):
    return raw_name.endswith("___healthy")


def load_per_class_csv():
    if not TEST_PER_CLASS_PATH.is_file():
        return []
    with TEST_PER_CLASS_PATH.open(encoding="utf-8") as f:
        records = []
        for row in csv.DictReader(f):
            crop, disease = format_class_name(row["class_name"])
            records.append({
                **row, "crop": crop, "disease": disease,
                **{k: int(row[k]) for k in ("label_idx", "support", "n_predicted")},
                **{k: float(row[k]) for k in ("precision", "recall", "f1", "auc")},
            })
        return records


CLASS_NAMES = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
HISTORY_DATA = load_history_csv()
TEST_METRICS = load_test_metrics()
PER_CLASS_DATA = load_per_class_csv()
KB_ERROR = ""
try:
    TREATMENT_KB = TreatmentKB()
except KnowledgeBaseError as exc:
    TREATMENT_KB = None
    KB_ERROR = str(exc)
    logging.error("Treatment service unavailable: %s", exc)

URDU_DISEASE_MAP = {
    "Tomato___Late_blight": "ٹماٹر کا پچھیتا جھلساؤ",
    "Tomato___Early_blight": "ٹماٹر کا اگیتا جھلساؤ",
    "Tomato___healthy": "تندرست ٹماٹر (کوئی بیماری نہیں)",
    "Tomato___Bacterial_spot": "ٹماٹر کا بیکٹیریل داغ",
    "Tomato___Leaf_Mold": "ٹماٹر کی پھپھوندی (لیف مولڈ)",
    "Tomato___Septoria_leaf_spot": "ٹماٹر کا سیپٹوریا پتا داغ",
    "Tomato___Spider_mites Two-spotted_spider_mite": "ٹماٹر کی لال مکڑی (مائٹس)",
    "Tomato___Target_Spot": "ٹماٹر کا ہدف نما داغ (ٹارگٹ سپاٹ)",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "ٹماٹر کا پیلا پتہ مروڑ وائرس",
    "Tomato___Tomato_mosaic_virus": "ٹماٹر کا موزیک وائرس",
    "Potato___Late_blight": "آلو کا پچھیتا جھلساؤ",
    "Potato___Early_blight": "آلو کا اگیتا جھلساؤ",
    "Potato___healthy": "تندرست آلو (کوئی بیماری نہیں)",
    "Corn___Common_rust": "مکئی کا بھورا یا زرد زنگ",
    "Corn___Northern_Leaf_Blight": "مکئی کا لمبوترا جھلساؤ",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": "مکئی کا سرمئی پتا داغ",
    "Corn___healthy": "تندرست مکئی",
    "Pepper___bell_Bacterial_spot": "مرچ کا بیکٹیریل داغ",
    "Pepper___healthy": "تندرست مرچ",
    "Apple___Apple_scab": "سیب کا کھرنڈ یا خارش (ایپل سکیب)",
    "Apple___Black_rot": "سیب کا کالا گلن سڑن (بلیک روٹ)",
    "Apple___Cedar_apple_rust": "سیب کا سڈار رسٹ زنگ",
    "Apple___healthy": "تندرست سیب",
    "Grape___Black_rot": "انگور کا سیاہ سڑن",
    "Grape___healthy": "تندرست انگور",
    "Orange___Haunglongbing(Citrus_greening)": "سٹرس گریننگ (ہوانگ لونگ بنگ)",
}
URDU_CROP_MAP = {
    "Tomato": "ٹماٹر", "Potato": "آلو", "Corn": "مکئی", "Pepper": "مرچ / شملہ مرچ",
    "Apple": "سیب", "Grape": "انگور", "Orange": "مالٹا / ترشاوہ", "Peach": "آڑو",
    "Strawberry": "سٹرابیری", "Soybean": "سویا بین", "Blueberry": "بلیو بیری",
    "Cherry": "چیری", "Raspberry": "رس بھری", "Squash": "حلوہ کدو",
}


def pil_to_base64(pil_img, format="JPEG"):
    buffered = io.BytesIO()
    pil_img.convert("RGB").save(buffered, format=format)
    return f"data:image/{format.lower()};base64," + base64.b64encode(buffered.getvalue()).decode("ascii")


def decode_image(raw):
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Use a JPEG, PNG or WebP leaf image")
            if min(image.size) < 32 or image.width * image.height > 20_000_000:
                raise ValueError("Image must be at least 32 pixels per side and at most 20 megapixels")
            image.load()
            result = image.convert("RGB")
        result.thumbnail((2048, 2048))
        return result
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid or corrupted image") from exc


def base64_to_pil(value):
    if not isinstance(value, str):
        raise ValueError("Image must be a base64 string")
    try:
        return decode_image(base64.b64decode(value.split(",", 1)[-1], validate=True))
    except (binascii.Error, UnicodeError) as exc:
        raise ValueError("Invalid base64 image") from exc


def parse_alpha(value):
    if isinstance(value, bool):
        raise ValueError("Alpha must be between 0 and 1")
    try:
        alpha = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Alpha must be between 0 and 1") from exc
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1")
    return alpha


def json_object():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    return data


ENGINE = ModelEngine(CHECKPOINT_PATH, CLASS_NAMES)
if not ENGINE.ready:
    app.logger.error("Model unavailable: %s", ENGINE.error)


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="Upload exceeds the 10 MB request limit"), 413


@app.errorhandler(ValueError)
def bad_input(error):
    return jsonify(error=str(error)), 400


@app.errorhandler(ModelUnavailable)
def unavailable(error):
    return jsonify(error=str(error)), 503


@app.route("/")
def index():
    key_names = ["Tomato___Late_blight", "Tomato___Early_blight", "Potato___Late_blight",
                 "Corn___Northern_Leaf_Blight", "Corn___Cercospora_leaf_spot Gray_leaf_spot",
                 "Pepper___bell_Bacterial_spot", "Apple___Apple_scab"]
    key_classes = [c for c in PER_CLASS_DATA if c["class_name"] in key_names]
    return render_template("index.html", history=HISTORY_DATA, test_metrics=TEST_METRICS,
                           per_class=key_classes or PER_CLASS_DATA[:7], all_per_class=PER_CLASS_DATA)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if request.is_json:
        data = json_object()
        if not data.get("image"):
            raise ValueError("No image provided")
        image = base64_to_pil(data["image"])
    elif "image" in request.files:
        data = request.form
        image = decode_image(request.files["image"].read())
    else:
        raise ValueError("No image provided")
    alpha = parse_alpha(data.get("alpha", 0.55))
    crop = data.get("crop")
    if crop is not None and (not isinstance(crop, str) or crop.casefold() not in
                            {"all", *(format_class_name(c)[0].casefold() for c in CLASS_NAMES)}):
        raise ValueError("Unknown crop filter")
    if TREATMENT_KB is None:
        raise ModelUnavailable("Treatment knowledge base is unavailable; restart after repairing its files")
    try:
        diagnosis = ENGINE.predict(image, filter_crop=crop)
        advisory = build_advisory(TREATMENT_KB, diagnosis["predicted_class"], diagnosis["confidence"])
    except KeyError as exc:
        raise ModelUnavailable("Prediction has no verified treatment mapping") from exc
    except (ModelUnavailable, ValueError):
        raise
    except Exception:
        app.logger.exception("Inference failed")
        raise ModelUnavailable("Inference failed; no treatment recommendation was issued") from None

    for candidate in diagnosis["top_3"]:
        candidate["disease_ur"] = URDU_DISEASE_MAP.get(candidate["class"], candidate["disease"])
    diagnosis["crop_ur"] = URDU_CROP_MAP.get(diagnosis["crop"], diagnosis["crop"])
    diagnosis["disease_ur"] = URDU_DISEASE_MAP.get(diagnosis["predicted_class"], diagnosis["disease"])
    cam_base64, cam_error = None, None
    try:
        cam_base64 = pil_to_base64(ENGINE.generate_gradcam(image, diagnosis["predicted_class"], alpha))
    except Exception:
        app.logger.exception("Grad-CAM unavailable")
        cam_error = "Grad-CAM is unavailable for this prediction; no substitute heatmap was generated."
    return jsonify({**diagnosis, "advisory": advisory, "treatment": advisory["treatment"],
                    "gradcam_base64": cam_base64, "gradcam_error": cam_error})


@app.route("/api/gradcam", methods=["POST"])
def api_gradcam():
    data = json_object()
    if not data.get("image"):
        raise ValueError("Image required")
    image = base64_to_pil(data["image"])
    alpha = parse_alpha(data.get("alpha", 0.55))
    target = data.get("target_class")
    if target is not None and target not in CLASS_NAMES:
        raise ValueError("Unknown Grad-CAM target class")
    try:
        return jsonify(gradcam_base64=pil_to_base64(ENGINE.generate_gradcam(image, target, alpha)))
    except (ModelUnavailable, ValueError):
        raise
    except Exception:
        app.logger.exception("Grad-CAM failed")
        raise ModelUnavailable("Grad-CAM failed; no substitute heatmap was generated") from None


@app.route("/api/history")
def api_history():
    return jsonify(HISTORY_DATA)


@app.route("/api/per-class")
def api_per_class():
    return jsonify(PER_CLASS_DATA)


@app.route("/api/metrics")
def api_metrics():
    return jsonify(TEST_METRICS)


@app.route("/api/viva-sample/<class_name>")
def api_viva_sample(class_name):
    if class_name not in CLASS_NAMES:
        raise ValueError("Unknown sample class")
    for base in SAMPLE_DIRS:
        folder = base / class_name
        if folder.is_dir():
            for file in sorted(folder.iterdir()):
                if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    try:
                        image = decode_image(file.read_bytes())
                    except ValueError:
                        continue
                    return jsonify(class_name=class_name, image_base64=pil_to_base64(image), source=file.name)
    return jsonify(error="No real sample is installed for this class; download the dataset"), 404


@app.route("/api/health")
def api_health():
    ready = ENGINE.ready and TREATMENT_KB is not None
    return jsonify(status="healthy" if ready else "unavailable", app="PakAgriVision",
                   framework="Flask 3.x", model="ResNet-50", model_ready=ENGINE.ready,
                   kb_ready=TREATMENT_KB is not None, num_classes=len(CLASS_NAMES),
                   device=ENGINE.device, epochs_recorded=len(HISTORY_DATA)), 200 if ready else 503


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    app.run(host="127.0.0.1", port=args.port, debug=False)
