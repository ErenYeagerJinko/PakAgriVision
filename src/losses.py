"""Loss function definitions and builder for PakAgriVision.

Phase 3 & 4 - Deliverable 2.
Supports:
    - Cross-Entropy Loss
    - Optional class weights (e.g., from splits/class_weights.json) for class imbalance
    - Optional label smoothing
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


def load_class_weights(
    weights_source: Optional[Union[str, Path, List[float], Any]] = None,
    device: Optional[Any] = None,
) -> Optional[Any]:
    """Load or format class weights into a PyTorch float tensor.

    Args:
        weights_source: Path to class_weights.json, list of floats, or torch Tensor.
        device: Device to place the tensor on.

    Returns:
        torch.FloatTensor of shape (38,) or None.
    """
    if weights_source is None:
        return None

    if torch is None:
        raise ImportError("PyTorch is required for load_class_weights.")

    if isinstance(weights_source, torch.Tensor):
        weights = weights_source.float()
    elif isinstance(weights_source, (list, tuple)):
        weights = torch.tensor(weights_source, dtype=torch.float32)
    elif isinstance(weights_source, (str, Path)):
        path = Path(weights_source)
        if not path.is_file():
            raise FileNotFoundError(f"Class weights file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "ordered_weights" in data:
            weights = torch.tensor(data["ordered_weights"], dtype=torch.float32)
        elif isinstance(data, list):
            weights = torch.tensor(data, dtype=torch.float32)
        else:
            raise ValueError(f"Unexpected format in class weights file: {path}")
    else:
        raise TypeError(f"Unsupported class_weights type: {type(weights_source)}")

    if device is not None:
        weights = weights.to(device)

    return weights


def build_loss(
    cfg: Optional[Dict[str, Any]] = None,
    class_weights: Optional[Any] = None,
    device: Optional[Any] = None,
) -> Any:
    """Construct loss function according to experiment configuration.

    Supports CrossEntropyLoss with optional train-only class weights and
    optional label smoothing.

    Args:
        cfg: Configuration dictionary (from YAML). Reads:
             - training.label_smoothing (float, default 0.0)
             - data.use_class_weights (bool, default False)
             - data.class_weights_path (str, optional, defaults to splits/class_weights.json)
        class_weights: Optional explicit tensor, list, or filepath of 38 class weights.
        device: Device to place the loss weight tensor on.

    Returns:
        nn.CrossEntropyLoss criterion.
    """
    if torch is None or nn is None:
        raise ImportError("PyTorch is required to build loss function.")

    label_smoothing = 0.0
    use_weights = False
    weights_path = Path("splits") / "class_weights.json"

    if cfg is not None:
        training_cfg = cfg.get("training", {})
        data_cfg = cfg.get("data", {})

        label_smoothing = float(training_cfg.get("label_smoothing", 0.0))
        use_weights = bool(data_cfg.get("use_class_weights", False))
        if "class_weights_path" in data_cfg:
            weights_path = Path(data_cfg["class_weights_path"])

    weight_tensor = None
    if class_weights is not None:
        weight_tensor = load_class_weights(class_weights, device=device)
    elif use_weights and weights_path.is_file():
        weight_tensor = load_class_weights(weights_path, device=device)

    return nn.CrossEntropyLoss(
        weight=weight_tensor,
        label_smoothing=label_smoothing,
    )
