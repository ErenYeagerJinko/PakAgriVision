"""Grad-CAM (Gradient-weighted Class Activation Mapping) explainability engine.

Phase 5 - Deliverable 3 (Track A: Application Development).
Provides:
    - PyTorch forward and backward hook mechanics attached to model target layer
      (default: ResNet-50 layer4[-1] or EfficientNet-B0 top conv block)
    - Gradient-weighted feature map pooling and ReLU rectification
    - Heatmap spatial normalization to range [0.0, 1.0]
    - RGB overlay blending with customizable colormaps (jet, turbo, viridis) and alpha opacity
    - Memory-safe hook removal via context managers / cleanup methods
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) generator.

    Attaches forward and backward hooks to a target convolutional layer to extract
    activation maps and feature gradients with respect to a target class score.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
    ) -> None:
        """Initialize Grad-CAM hooks on target layer.

        Args:
            model: PyTorch classifier model (e.g. CropDiseaseClassifier).
            target_layer: Target convolutional module to hook into. If None,
                          attempts to read model.target_layer or defaults to
                          the final bottleneck of layer4 in ResNet-50.
        """
        self.model = model
        self.model.eval()

        # Resolve target layer
        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "target_layer"):
            self.target_layer = model.target_layer
        elif hasattr(model, "model") and hasattr(model.model, "layer4"):
            self.target_layer = model.model.layer4[-1]
        elif hasattr(model, "layer4"):
            self.target_layer = model.layer4[-1]
        else:
            raise ValueError(
                "Unable to automatically resolve target layer for Grad-CAM. "
                "Please pass target_layer explicitly."
            )

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles: List[Any] = []

        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target convolutional layer."""
        def forward_hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(module: nn.Module, grad_input: Any, grad_output: Tuple[torch.Tensor, ...]) -> None:
            # Gradients with respect to target layer activations
            self.gradients = grad_output[0].detach()

        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        h_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        self._handles.extend([h_fwd, h_bwd])

    def remove_hooks(self) -> None:
        """Safely remove registered hooks to prevent memory leaks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.remove_hooks()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Generate 2D spatial heatmap for a given input tensor and class index.

        Args:
            input_tensor: 4D Tensor of shape (1, C, H, W).
            target_class_idx: Integer index of target class. If None, uses top-1 predicted class.

        Returns:
            Normalized 2D float32 numpy array with values in [0.0, 1.0] matching input (H, W).
        """
        if input_tensor.dim() != 4 or input_tensor.size(0) != 1:
            raise ValueError(f"input_tensor must be of shape (1, C, H, W), got {input_tensor.shape}")

        device = next(self.model.parameters()).device
        tensor = input_tensor.to(device)

        # Ensure gradients are enabled for the backward pass
        self.model.zero_grad()
        with torch.enable_grad():
            tensor.requires_grad_(True)
            logits = self.model(tensor)

            if target_class_idx is None:
                target_class_idx = int(torch.argmax(logits, dim=-1).item())

            score = logits[0, target_class_idx]
            score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations or gradients from target layer.")

        # Global Average Pooling (GAP) of gradients across spatial dimensions (H_feat, W_feat)
        # activations: (1, C_feat, H_feat, W_feat)
        # gradients:   (1, C_feat, H_feat, W_feat)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # Shape: (1, C_feat, 1, 1)

        # Weighted linear combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # Shape: (1, 1, H_feat, W_feat)

        # Apply ReLU to retain only positive contributions toward target class
        cam = F.relu(cam)

        # Upsample heatmap to match original input tensor spatial dimensions (H, W)
        cam = F.interpolate(
            cam,
            size=(tensor.shape[2], tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        cam_np = cam.squeeze().cpu().numpy()

        # Spatial normalization to strictly [0.0, 1.0]
        cam_min, cam_max = float(np.min(cam_np)), float(np.max(cam_np))
        if cam_max - cam_min > 1e-8:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        return cam_norm.astype(np.float32)

    @staticmethod
    def overlay_heatmap(
        pil_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap_name: str = "jet",
    ) -> Image.Image:
        """Blend a 2D normalized Grad-CAM heatmap with original PIL RGB image.

        Args:
            pil_image: Source RGB PIL Image.
            heatmap: 2D numpy array with values in [0.0, 1.0].
            alpha: Heatmap blend opacity weight (0.0 = original image, 1.0 = raw heatmap).
            colormap_name: Matplotlib colormap name ('jet', 'turbo', 'viridis', 'plasma').

        Returns:
            Blended PIL.Image in RGB mode.
        """
        alpha = float(np.clip(alpha, 0.0, 1.0))
        img_w, img_h = pil_image.size

        # Resize heatmap if dimensions differ from original image
        if heatmap.shape != (img_h, img_w):
            heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
            heatmap_resized = heatmap_img.resize((img_w, img_h), resample=Image.Resampling.BILINEAR)
            heatmap = np.array(heatmap_resized, dtype=np.float32) / 255.0

        # Apply colormap using Matplotlib
        cmap = plt.get_cmap(colormap_name)
        colored_heatmap = cmap(heatmap)[:, :, :3]  # Strip alpha -> (H, W, 3) in [0.0, 1.0]

        # Convert original PIL image to float numpy [0.0, 1.0]
        orig_np = np.array(pil_image.convert("RGB"), dtype=np.float32) / 255.0

        # Alpha blend: (1 - alpha) * image + alpha * heatmap
        blended = (1.0 - alpha) * orig_np + alpha * colored_heatmap
        blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

        return Image.fromarray(blended, mode="RGB")


def generate_gradcam_visualization(
    model: nn.Module,
    input_tensor: torch.Tensor,
    pil_image: Image.Image,
    target_class_idx: Optional[int] = None,
    alpha: float = 0.5,
    colormap_name: str = "jet",
) -> Tuple[np.ndarray, Image.Image]:
    """Convenience helper to compute heatmap and blended overlay in a single call.

    Args:
        model: Trained classifier.
        input_tensor: 4D Tensor (1, C, H, W).
        pil_image: Original input PIL Image.
        target_class_idx: Target class index (or None for top-1).
        alpha: Blending opacity (0.0 to 1.0).
        colormap_name: Colormap name.

    Returns:
        (heatmap_numpy_array, blended_pil_image)
    """
    with GradCAM(model=model) as cam_generator:
        heatmap = cam_generator.generate_cam(
            input_tensor=input_tensor,
            target_class_idx=target_class_idx,
        )
        overlay = cam_generator.overlay_heatmap(
            pil_image=pil_image,
            heatmap=heatmap,
            alpha=alpha,
            colormap_name=colormap_name,
        )
    return heatmap, overlay

