"""Loss function definitions and builder for PakAgriVision.

Agreed Day 1 signature:
    - build_loss(cfg, class_weights, device)
"""

from typing import Any, Dict, Optional


def build_loss(
    cfg: Optional[Dict[str, Any]] = None,
    class_weights: Optional[Any] = None,
    device: Optional[Any] = None,
) -> Any:
    """Construct loss function according to experiment configuration.

    Supports cross-entropy loss with optional class weights (for class imbalance)
    and optional label smoothing.

    Args:
        cfg: Configuration dictionary containing training loss parameters.
        class_weights: Optional 1D tensor or list of 38 class weights.
        device: Device to place the loss weight tensor on.

    Returns:
        Callable PyTorch loss criterion.
    """
    raise NotImplementedError("Stub: Implementation landing in Day 1 part 2.")

