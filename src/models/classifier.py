"""Crop disease classification model architecture.

Phase 3 & 4 - Model Architecture and Deliverable 2.
Supports:
    - ResNet-50 (default) with custom head: Dropout(p) -> Linear(2048, num_classes)
    - EfficientNet-B0 (alternative) with custom head: Dropout(p) -> Linear(1280, num_classes)
    - Backbone freeze/unfreeze mechanism for 1-epoch head-only warm-up
    - Exposes `model.target_layer` for Grad-CAM interpretability
"""

from typing import Any, Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    from torchvision import models
except ImportError:
    torch = None
    nn = None
    models = None


class CropDiseaseClassifier(nn.Module if nn is not None else object):
    """Wrapper classifier module for crop disease detection."""

    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_classes: int = 38,
        dropout: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.pretrained = pretrained

        if self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            in_features = base.fc.in_features  # 2048
            base.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
            self.model = base
            self.target_layer = self.model.layer4[-1]
            self.head_module = self.model.fc

        elif self.backbone_name in ("efficientnet_b0", "efficientnet-b0"):
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            in_features = base.classifier[1].in_features  # 1280
            base.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
            self.model = base
            self.target_layer = self.model.features[-1]
            self.head_module = self.model.classifier

        else:
            raise ValueError(
                f"Unsupported backbone: '{backbone_name}'. Supported options: 'resnet50', 'efficientnet_b0'."
            )

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def set_backbone_trainable(self, trainable: bool = True) -> None:
        """Freeze or unfreeze backbone parameters while keeping classification head trainable."""
        head_params = set(self.head_module.parameters())
        for param in self.model.parameters():
            if param in head_params:
                param.requires_grad = True
            else:
                param.requires_grad = trainable

    def count_parameters(self) -> Dict[str, int]:
        """Return total, trainable, and frozen parameter counts."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
        }


def build_model(
    backbone: str = "resnet50",
    num_classes: int = 38,
    dropout: float = 0.3,
    pretrained: bool = True,
) -> Any:
    """Build and return crop disease classifier model.

    Args:
        backbone: Backbone feature extractor ('resnet50' or 'efficientnet_b0').
        num_classes: Number of disease/health target classes (default 38).
        dropout: Dropout rate prior to final linear classification layer.
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        CropDiseaseClassifier instance with target_layer exposed for Grad-CAM.
    """
    if torch is None:
        raise ImportError("PyTorch is required to build the model.")
    return CropDiseaseClassifier(
        backbone_name=backbone,
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained,
    )


def set_backbone_trainable(model: Any, trainable: bool = True) -> None:
    """Freeze or unfreeze backbone layers of the classifier model.

    Args:
        model: Classifier model built with `build_model`.
        trainable: True to unfreeze backbone parameters; False to freeze them.
    """
    if hasattr(model, "set_backbone_trainable"):
        model.set_backbone_trainable(trainable)
    elif hasattr(model, "model") and hasattr(model, "head_module"):
        head_params = set(model.head_module.parameters())
        for param in model.model.parameters():
            if param in head_params:
                param.requires_grad = True
            else:
                param.requires_grad = trainable
    else:
        raise AttributeError("Model does not expose backbone freezing mechanism.")


def get_target_layer(model: Any) -> Any:
    """Retrieve the target convolutional layer for Grad-CAM visualization."""
    if hasattr(model, "target_layer"):
        return model.target_layer
    raise AttributeError("Model does not expose target_layer for Grad-CAM.")
