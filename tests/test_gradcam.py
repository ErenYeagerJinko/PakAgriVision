"""Unit tests for Grad-CAM explainability pipeline.

Phase 5 - Deliverable 3 (M. Anas scope).
Tests:
    - Target layer hook attachment and graceful removal
    - Heatmap spatial resolution and [0.0, 1.0] value bounds
    - Target class specificity (different class indices generate different heatmaps)
    - Overlay blending dimensions and RGB format
    - Context manager safety
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.explainability.gradcam import GradCAM, generate_gradcam_visualization
from src.models.classifier import build_model


@pytest.fixture(scope="module")
def classifier_model() -> torch.nn.Module:
    """Fixture providing initialized ResNet-50 classifier on CPU."""
    model = build_model(
        backbone="resnet50",
        num_classes=38,
        dropout=0.3,
        pretrained=False,
    )
    model.eval()
    return model


@pytest.fixture
def dummy_input_tensor() -> torch.Tensor:
    """Fixture providing normalized 4D tensor (1, 3, 224, 224)."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def dummy_pil_image() -> Image.Image:
    """Fixture providing synthetic RGB PIL image (224x224)."""
    arr = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_gradcam_hook_lifecycle(classifier_model: torch.nn.Module) -> None:
    """Test that hooks register correctly and are cleared upon remove_hooks()."""
    cam = GradCAM(model=classifier_model)
    assert len(cam._handles) == 2

    cam.remove_hooks()
    assert len(cam._handles) == 0


def test_gradcam_context_manager(classifier_model: torch.nn.Module) -> None:
    """Test context manager cleanup mechanics."""
    with GradCAM(model=classifier_model) as cam:
        assert len(cam._handles) == 2
    assert len(cam._handles) == 0


def test_generate_cam_properties(
    classifier_model: torch.nn.Module,
    dummy_input_tensor: torch.Tensor,
) -> None:
    """Test heatmap shape, dtype, and strict [0.0, 1.0] normalization."""
    with GradCAM(model=classifier_model) as cam:
        heatmap = cam.generate_cam(dummy_input_tensor, target_class_idx=0)

        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape == (224, 224)
        assert heatmap.dtype == np.float32
        assert float(np.min(heatmap)) >= 0.0
        assert float(np.max(heatmap)) <= 1.0


def test_target_class_differentiation(
    classifier_model: torch.nn.Module,
    dummy_input_tensor: torch.Tensor,
) -> None:
    """Test that targeting two distinct classes yields distinct feature attributions."""
    with GradCAM(model=classifier_model) as cam:
        cam_class0 = cam.generate_cam(dummy_input_tensor, target_class_idx=0)
        cam_class1 = cam.generate_cam(dummy_input_tensor, target_class_idx=1)

        # The heatmaps for two different classes should not be identical
        diff = np.abs(cam_class0 - cam_class1)
        assert np.mean(diff) > 1e-5


def test_overlay_heatmap_blending(
    dummy_pil_image: Image.Image,
) -> None:
    """Test blending returns proper PIL image with matching dimensions."""
    dummy_heatmap = np.random.rand(224, 224).astype(np.float32)

    overlay = GradCAM.overlay_heatmap(
        pil_image=dummy_pil_image,
        heatmap=dummy_heatmap,
        alpha=0.4,
        colormap_name="jet",
    )

    assert isinstance(overlay, Image.Image)
    assert overlay.size == dummy_pil_image.size
    assert overlay.mode == "RGB"


def test_generate_gradcam_visualization_convenience(
    classifier_model: torch.nn.Module,
    dummy_input_tensor: torch.Tensor,
    dummy_pil_image: Image.Image,
) -> None:
    """Test end-to-end convenience wrapper function."""
    heatmap, overlay = generate_gradcam_visualization(
        model=classifier_model,
        input_tensor=dummy_input_tensor,
        pil_image=dummy_pil_image,
        target_class_idx=None,
        alpha=0.5,
    )

    assert heatmap.shape == (224, 224)
    assert overlay.size == (224, 224)
    assert overlay.mode == "RGB"

