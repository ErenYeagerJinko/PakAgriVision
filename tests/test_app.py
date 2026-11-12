"""Unit and integration tests for the PakAgriVision Flask application."""

import base64
import io
import pytest
from PIL import Image
from app.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_leaf_b64():
    """Generate a 224x224 RGB image encoded as base64."""
    img = Image.new("RGB", (224, 224), color=(46, 125, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def test_home_page_renders(client):
    """Test that the main frontend loads successfully with rendered CSV metrics."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "PakAgriVision" in html
    assert "ResNet-50" in html
    assert "Benchmark Results" in html


def test_api_health(client):
    """Test the health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["num_classes"] == 38
    assert data["epochs_recorded"] == 15


def test_api_metrics(client):
    """Test the test summary metrics endpoint derived from metrics.json."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.get_json()
    assert "accuracy" in data
    assert data["accuracy"] > 0.99
    assert data["n_samples"] == 4768


def test_api_history(client):
    """Test training history endpoint derived from results/baseline/history.csv."""
    response = client.get("/api/history")
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 15
    assert data[0]["epoch"] == 1
    assert data[-1]["epoch"] == 15
    assert "val_acc" in data[0]


def test_api_per_class(client):
    """Test per-class performance endpoint derived from per_class.csv."""
    response = client.get("/api/per-class")
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 38
    first = data[0]
    assert "crop" in first
    assert "disease" in first
    assert "f1" in first


def test_api_predict(client, sample_leaf_b64):
    """Test the /api/predict endpoint with image upload."""
    payload = {
        "image": sample_leaf_b64,
        "alpha": 0.55,
        "crop": None,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert "predicted_class" in data
    assert "crop" in data
    assert "crop_ur" in data
    assert "disease" in data
    assert "disease_ur" in data
    assert "confidence" in data
    assert "gradcam_base64" in data
    assert data["gradcam_base64"].startswith("data:image/jpeg;base64,")


def test_api_predict_with_crop_filter(client, sample_leaf_b64):
    """Test /api/predict with active crop filter."""
    payload = {
        "image": sample_leaf_b64,
        "alpha": 0.55,
        "crop": "Tomato",
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["crop"] == "Tomato"

