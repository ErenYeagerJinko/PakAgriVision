"""HTTP advisory integration tests; model stubs isolate safety/error boundaries.

A separate checkpoint smoke test exercises the real classifier and Grad-CAM when
local weights are present. Stub images/results are test fixtures, never app fallbacks.
"""

import base64
import importlib
import io
from pathlib import Path

from PIL import Image
import pytest

from app.model_engine import ModelEngine, ModelUnavailable

web = importlib.import_module("app.app")


class StubEngine:
    ready = True
    device = "cpu"
    error = ""

    def __init__(self, label="Tomato___Late_blight", confidence=92):
        self.label, self.confidence = label, confidence

    def predict(self, image, filter_crop=None):
        assert image.mode == "RGB"
        crop, disease = web.format_class_name(self.label)
        return {"predicted_class": self.label, "crop": crop, "disease": disease,
                "confidence": self.confidence, "is_healthy": web.is_healthy(self.label),
                "top_3": [{"class": self.label, "crop": crop, "disease": disease,
                           "confidence": self.confidence}]}

    def generate_gradcam(self, image, target_class=None, alpha=0.55):
        return image


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(web, "ENGINE", StubEngine())
    web.app.config["TESTING"] = True
    with web.app.test_client() as client:
        yield client


@pytest.fixture
def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGBA", (100, 80), (20, 120, 40, 120)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def payload(image_bytes):
    return {"image": "data:image/png;base64," + base64.b64encode(image_bytes).decode()}


def test_image_to_sourced_treatment_json(client, payload):
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["predicted_class"] == "Tomato___Late_blight"
    assert data["advisory"]["status"] == "disease_detected"
    assert data["treatment"] == data["advisory"]["treatment"]
    assert data["treatment"]["sources"]
    assert data["gradcam_base64"].startswith("data:image/jpeg;base64,")
    assert "severity" not in data  # probability is not a severity measurement


def test_multipart_upload(client, image_bytes):
    response = client.post("/api/predict", data={"image": (io.BytesIO(image_bytes), "leaf.png")})
    assert response.status_code == 200


@pytest.mark.parametrize("label", ["Tomato___Late_blight", "Tomato___healthy"])
@pytest.mark.parametrize("confidence,uncertain", [(59.99999, True), (60, False)])
def test_server_threshold_is_authoritative(client, payload, monkeypatch, label, confidence, uncertain):
    monkeypatch.setattr(web, "ENGINE", StubEngine(label, confidence))
    data = client.post("/api/predict", json=payload).get_json()
    advisory = data["advisory"]
    assert advisory["low_confidence"] is uncertain
    assert bool(advisory["warning_ur"]) is uncertain
    if uncertain:
        assert "withheld" in data["treatment"]["chemical_treatment"].lower()
        assert "withheld" in data["treatment"]["organic_treatment"].lower()
        assert advisory["status"] == "uncertain"


@pytest.mark.parametrize("body", [{}, [], {"image": "not base64"}, {"image": 42},
                                   {"image": base64.b64encode(b"not an image").decode()}])
def test_invalid_input_is_400(client, body):
    response = client.post("/api/predict", json=body)
    assert response.status_code == 400
    assert "error" in response.get_json()


@pytest.mark.parametrize("alpha", [-0.1, 1.1, "nan", "infinity", None, True])
def test_invalid_alpha_is_400(client, payload, alpha):
    assert client.post("/api/predict", json={**payload, "alpha": alpha}).status_code == 400


@pytest.mark.parametrize("crop", ["unknown", 42, []])
def test_invalid_crop_is_400(client, payload, crop):
    assert client.post("/api/predict", json={**payload, "crop": crop}).status_code == 400


def test_request_size_limit(client):
    response = client.post("/api/predict", data=b"x" * (10 * 1024 * 1024 + 1),
                           content_type="application/json")
    assert response.status_code == 413


def test_small_image_rejected(client):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="PNG")
    response = client.post("/api/predict", json={"image": base64.b64encode(buffer.getvalue()).decode()})
    assert response.status_code == 400


def test_model_failure_has_no_fake_diagnosis(client, payload, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("internal tensor failure")
    monkeypatch.setattr(web.ENGINE, "predict", fail)
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 503
    assert "treatment" not in response.get_json()
    assert "internal tensor" not in response.get_json()["error"]


def test_missing_model_and_health(client, payload, monkeypatch, tmp_path):
    engine = ModelEngine(tmp_path / "absent.pt", web.CLASS_NAMES)
    assert not engine.ready
    monkeypatch.setattr(web, "ENGINE", engine)
    assert client.get("/api/health").status_code == 503
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 503
    assert "predicted_class" not in response.get_json()


def test_incompatible_checkpoint_never_becomes_ready(tmp_path):
    import torch
    checkpoint = tmp_path / "wrong.pt"
    torch.save({"class_names": list(reversed(web.CLASS_NAMES)), "model_state_dict": {}}, checkpoint)
    engine = ModelEngine(checkpoint, web.CLASS_NAMES)
    assert not engine.ready
    assert "class order" in engine.error


def test_corrupted_checkpoint_never_becomes_ready(tmp_path):
    checkpoint = tmp_path / "corrupt.pt"
    checkpoint.write_bytes(b"not a checkpoint")
    engine = ModelEngine(checkpoint, web.CLASS_NAMES)
    assert not engine.ready
    with pytest.raises(ModelUnavailable):
        engine.predict(Image.new("RGB", (100, 100)))


def test_missing_kb_and_unknown_mapping_fail_closed(client, payload, monkeypatch):
    monkeypatch.setattr(web, "ENGINE", StubEngine("unknown"))
    assert client.post("/api/predict", json=payload).status_code == 503
    monkeypatch.setattr(web, "TREATMENT_KB", None)
    assert client.post("/api/predict", json=payload).status_code == 503
    assert not client.get("/api/health").get_json()["kb_ready"]


def test_gradcam_failure_is_explicit_not_synthetic(client, payload, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("no gradients")
    monkeypatch.setattr(web.ENGINE, "generate_gradcam", fail)
    data = client.post("/api/predict", json=payload).get_json()
    assert data["gradcam_base64"] is None
    assert data["gradcam_error"]
    assert data["advisory"]["treatment"]["sources"]
    assert client.post("/api/gradcam", json=payload).status_code == 503


def test_unknown_target_and_missing_real_sample(client, payload, monkeypatch, tmp_path):
    assert client.post("/api/gradcam", json={**payload, "target_class": "unknown"}).status_code == 400
    monkeypatch.setattr(web, "SAMPLE_DIRS", [tmp_path])
    assert client.get("/api/viva-sample/Tomato___healthy").status_code == 404
    assert client.get("/api/viva-sample/unknown").status_code == 400


def test_frontend_serves_advisory_script(client):
    html = client.get("/").get_data(as_text=True)
    assert "advisoryWarningText" in html and "treatmentSources" in html
    assert "gradcam-heatmap-layer" not in html
    response = client.get("/static/advisory.js")
    assert response.status_code == 200
    assert "advisory.low_confidence" in response.get_data(as_text=True)


def test_filtered_ranking_keeps_original_confidence(monkeypatch):
    engine = ModelEngine.__new__(ModelEngine)
    from threading import RLock
    from types import SimpleNamespace
    engine.ready, engine._lock = True, RLock()
    engine.class_names = web.CLASS_NAMES
    engine.inference = SimpleNamespace(predict=lambda *a, **k: {
        "top_k": [
            {"class_name": "Apple___healthy", "crop": "Apple", "condition": "Healthy", "probability": .7},
            {"class_name": "Tomato___Late_blight", "crop": "Tomato", "condition": "Late Blight", "probability": .2}],
        "latency_ms": 1,
    })
    diagnosis = engine.predict(Image.new("RGB", (100, 100)), filter_crop="Tomato")
    assert diagnosis["confidence"] == 20


def test_real_checkpoint_image_to_advisory():
    """One real image through actual checkpoint, Grad-CAM and treatment retrieval."""
    if not web.CHECKPOINT_PATH.is_file():
        pytest.skip("Download baseline weights to run the real-model smoke test")
    assert web.ENGINE.ready, web.ENGINE.error
    import pandas as pd
    manifest = pd.read_csv(web.ROOT_DIR / "splits/val.csv")
    relative = manifest.iloc[0]["path"]
    path = web.ROOT_DIR / "data/raw/plantvillage/color" / relative
    if not path.is_file():
        pytest.skip("Download the curated dataset to run the real-image smoke test")
    with web.app.test_client() as client:
        response = client.post("/api/predict", json={"image": base64.b64encode(path.read_bytes()).decode()})
    assert response.status_code == 200, response.get_json()
    data = response.get_json()
    assert data["predicted_class"] in web.CLASS_NAMES
    assert 0 <= data["confidence"] <= 100
    assert data["gradcam_base64"] and data["gradcam_error"] is None
    encoded = data["gradcam_base64"].split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as overlay:
        assert overlay.size == (224, 224)  # exact center-cropped model view
    assert data["treatment"]["sources"]
