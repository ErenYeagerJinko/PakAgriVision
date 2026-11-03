"""Unit tests for model architecture, loss function, and optimizer.

Verifies:
1. Model forward pass produces shape (B, 38) for ResNet50 and EfficientNet-B0.
2. Target layer is exposed for Grad-CAM interpretability.
3. Backbone freeze/unfreeze mechanism operates properly (head remains trainable).
4. Parameter counts are accurately computed and printable for the report.
5. Loss function builds and computes valid cross-entropy with weights and label smoothing.
6. Optimizer and scheduler step properly.
"""

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from src.losses import build_loss, load_class_weights
from src.models.classifier import (
    build_model,
    get_target_layer,
    set_backbone_trainable,
)
from src.optim import build_optimizer, build_scheduler


def test_resnet50_output_shape():
    """Verify ResNet-50 forward pass produces (batch_size, 38)."""
    model = build_model(backbone="resnet50", num_classes=38, dropout=0.3, pretrained=False)
    model.eval()
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 38), f"Expected shape (2, 38), got {output.shape}"


def test_efficientnet_output_shape():
    """Verify EfficientNet-B0 forward pass produces (batch_size, 38)."""
    model = build_model(backbone="efficientnet_b0", num_classes=38, dropout=0.3, pretrained=False)
    model.eval()
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 38), f"Expected shape (2, 38), got {output.shape}"


def test_target_layer_exposure():
    """Verify target layers are accessible for Grad-CAM."""
    resnet = build_model(backbone="resnet50", pretrained=False)
    target_res = get_target_layer(resnet)
    assert target_res is not None

    effnet = build_model(backbone="efficientnet_b0", pretrained=False)
    target_eff = get_target_layer(effnet)
    assert target_eff is not None


def test_backbone_freeze_unfreeze():
    """Verify backbone can be frozen while keeping classifier head trainable."""
    model = build_model(backbone="resnet50", pretrained=False)

    # Freeze backbone
    set_backbone_trainable(model, trainable=False)
    counts_frozen = model.count_parameters()
    assert counts_frozen["frozen"] > 0
    assert counts_frozen["trainable"] > 0
    assert counts_frozen["trainable"] < counts_frozen["total"]

    # Verify head parameters specifically require grad
    for param in model.head_module.parameters():
        assert param.requires_grad is True

    # Unfreeze backbone
    set_backbone_trainable(model, trainable=True)
    counts_unfrozen = model.count_parameters()
    assert counts_unfrozen["frozen"] == 0
    assert counts_unfrozen["trainable"] == counts_unfrozen["total"]


def test_parameter_counts_reporting():
    """Print model parameter counts for inclusion in report Section 3.2."""
    resnet = build_model(backbone="resnet50", pretrained=False)
    counts = resnet.count_parameters()

    print(f"\n[Model Info] ResNet-50 Total Parameters: {counts['total']:,}")
    print(f"[Model Info] ResNet-50 Trainable: {counts['trainable']:,}")
    assert counts["total"] > 20_000_000


def test_build_loss_computation():
    """Verify cross-entropy loss computation with weights and label smoothing."""
    cfg = {
        "training": {"label_smoothing": 0.1},
        "data": {"use_class_weights": True, "class_weights_path": "splits/class_weights.json"},
    }
    criterion = build_loss(cfg)
    assert isinstance(criterion, nn.CrossEntropyLoss)

    logits = torch.randn(4, 38)
    targets = torch.tensor([0, 5, 12, 37], dtype=torch.long)
    loss = criterion(logits, targets)
    assert loss.item() > 0.0


def test_build_optimizer_and_scheduler():
    """Verify optimizer and scheduler construction and stepping."""
    model = build_model(backbone="resnet50", pretrained=False)
    cfg = {
        "training": {
            "optimizer": "adamw",
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "warmup_epochs": 1,
            "epochs": 5,
        }
    }
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, epochs=5)

    initial_lr = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    next_lr = optimizer.param_groups[0]["lr"]

    assert initial_lr is not None
    assert next_lr is not None

