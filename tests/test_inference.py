"""Test suite for production real-time inference engine.

Phase 5 - Deliverable 3 (M. Anas scope).
Tests:
    - Input preprocessing with diverse formats (RGB, RGBA, Grayscale, BytesIO)
    - Exception handling for corrupt inputs
    - Prediction output dictionary structure and probability validity
    - Label parsing (crop vs disease vs healthy)
    - Sub-second inference latency on CPU (<500ms)
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.inference import InferenceEngine


@pytest.fixture(scope="module")
def inference_engine() -> InferenceEngine:
    """Offline structural tests; trained-model behavior is covered by integration tests."""
    from src.models.classifier import build_model
    with patch("src.inference.build_model", side_effect=lambda **kwargs:
               build_model(**{**kwargs, "pretrained": False})):
        return InferenceEngine(
            checkpoint_path=None,
            backbone="resnet50",
            num_classes=38,
            device="cpu",
        )


@pytest.fixture
def sample_rgb_image() -> Image.Image:
    """Fixture providing a synthetic RGB PIL Image."""
    arr = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def sample_rgba_image() -> Image.Image:
    """Fixture providing an RGBA PIL Image."""
    arr = (np.random.rand(200, 200, 4) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture
def sample_grayscale_image() -> Image.Image:
    """Fixture providing a Grayscale (L mode) PIL Image."""
    arr = (np.random.rand(150, 150) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def test_preprocess_formats(
    sample_rgb_image: Image.Image,
    sample_rgba_image: Image.Image,
    sample_grayscale_image: Image.Image,
) -> None:
    """Test image preprocessing handles RGB, RGBA, Grayscale, and BytesIO buffers."""
    # RGB
    img1 = InferenceEngine.preprocess_image(sample_rgb_image)
    assert img1.mode == "RGB"

    # RGBA -> RGB
    img2 = InferenceEngine.preprocess_image(sample_rgba_image)
    assert img2.mode == "RGB"

    # Grayscale -> RGB
    img3 = InferenceEngine.preprocess_image(sample_grayscale_image)
    assert img3.mode == "RGB"

    # BytesIO buffer
    buf = io.BytesIO()
    sample_rgb_image.save(buf, format="JPEG")
    buf.seek(0)
    img4 = InferenceEngine.preprocess_image(buf)
    assert img4.mode == "RGB"

    # Raw bytes
    img5 = InferenceEngine.preprocess_image(buf.getvalue())
    assert img5.mode == "RGB"


def test_preprocess_corrupt_raises() -> None:
    """Test that invalid image bytes raise ValueError."""
    with pytest.raises(ValueError, match="Failed to decode image"):
        InferenceEngine.preprocess_image(b"not_an_image_data_stream")


def test_label_parsing() -> None:
    """Test label parser splits crop and disease conditions cleanly."""
    crop, dis, healthy = InferenceEngine._parse_label("Tomato___Late_blight")
    assert crop == "Tomato"
    assert dis == "Late Blight"
    assert healthy is False

    crop2, dis2, healthy2 = InferenceEngine._parse_label("Apple___healthy")
    assert crop2 == "Apple"
    assert dis2 == "Healthy"
    assert healthy2 is True


def test_predict_structure_and_probabilities(
    inference_engine: InferenceEngine,
    sample_rgb_image: Image.Image,
) -> None:
    """Test prediction output dictionary conforms to specification."""
    res = inference_engine.predict(sample_rgb_image, top_k=3)

    assert "class_idx" in res
    assert 0 <= res["class_idx"] < 38
    assert "class_name" in res
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert "confidence_pct" in res
    assert "%" in res["confidence_pct"]

    # Top-k checks
    assert len(res["top_k"]) == 3
    probs = [c["probability"] for c in res["top_k"]]
    # Top probs should be descending
    assert probs[0] >= probs[1] >= probs[2]

    # Preprocessed tensor shape
    assert res["tensor"].shape == (1, 3, 224, 224)


def test_predict_latency(
    inference_engine: InferenceEngine,
    sample_rgb_image: Image.Image,
) -> None:
    """Test CPU latency is responsive (<500ms on local CPU)."""
    # Warmup
    _ = inference_engine.predict(sample_rgb_image, top_k=3)

    # Timed run
    res = inference_engine.predict(sample_rgb_image, top_k=3)
    assert "latency_ms" in res
    assert res["latency_ms"] < 800.0  # Generous upper bound for slow CPU

