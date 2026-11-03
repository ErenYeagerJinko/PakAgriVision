"""Optimizer and learning rate scheduler builders for PakAgriVision.

Agreed Day 1 signatures:
    - build_optimizer(model, cfg)
    - build_scheduler(optimizer, cfg, epochs, steps_per_epoch)
"""

from typing import Any, Dict, Optional


def build_optimizer(model: Any, cfg: Optional[Dict[str, Any]] = None) -> Any:
    """Construct optimizer according to configuration.

    Supports AdamW (default) and SGD with momentum.

    Args:
        model: PyTorch model containing trainable parameters.
        cfg: Configuration dictionary containing optimizer parameters (lr, weight_decay).

    Returns:
        Configured PyTorch optimizer instance.
    """
    raise NotImplementedError("Stub: Implementation landing in Day 1 part 2.")


def build_scheduler(
    optimizer: Any,
    cfg: Optional[Dict[str, Any]] = None,
    epochs: int = 15,
    steps_per_epoch: int = 1,
) -> Any:
    """Construct learning rate scheduler with warm-up followed by cosine annealing.

    Args:
        optimizer: PyTorch optimizer instance.
        cfg: Configuration dictionary.
        epochs: Total number of training epochs.
        steps_per_epoch: Number of optimization steps per epoch (if step-wise).

    Returns:
        PyTorch learning rate scheduler instance.
    """
    raise NotImplementedError("Stub: Implementation landing in Day 1 part 2.")

