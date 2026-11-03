"""Optimizer and learning rate scheduler builders for PakAgriVision.

Phase 3 & 4 - Deliverable 2.
Supports:
    - AdamW (default) and SGD with momentum
    - Cosine annealing scheduler with 1-epoch warm-up
"""

from typing import Any, Dict, Optional

try:
    import torch
    import torch.optim as optim
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR,
        LinearLR,
        SequentialLR,
    )
except ImportError:
    torch = None
    optim = None


def build_optimizer(model: Any, cfg: Optional[Dict[str, Any]] = None) -> Any:
    """Construct optimizer according to configuration.

    Supports AdamW (default) and SGD with momentum.

    Args:
        model: PyTorch model instance or iterable of parameters.
        cfg: Optional configuration dictionary containing:
             - training.optimizer: 'adamw' or 'sgd'
             - training.lr: learning rate (default 3e-4)
             - training.weight_decay: weight decay (default 1e-4)
             - training.momentum: momentum factor for SGD (default 0.9)

    Returns:
        torch.optim.Optimizer instance.
    """
    if torch is None or optim is None:
        raise ImportError("PyTorch is required to build optimizer.")

    params = model.parameters() if hasattr(model, "parameters") else model

    opt_name = "adamw"
    lr = 3e-4
    weight_decay = 1e-4
    momentum = 0.9

    if cfg is not None:
        train_cfg = cfg.get("training", {})
        opt_name = train_cfg.get("optimizer", "adamw").lower()
        lr = float(train_cfg.get("lr", 3e-4))
        weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        momentum = float(train_cfg.get("momentum", 0.9))

    if opt_name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(
            f"Unsupported optimizer: '{opt_name}'. Supported: 'adamw', 'sgd'."
        )


def build_scheduler(
    optimizer: Any,
    cfg: Optional[Dict[str, Any]] = None,
    epochs: int = 15,
    steps_per_epoch: int = 1,
) -> Any:
    """Construct learning rate scheduler with warm-up followed by cosine annealing.

    Args:
        optimizer: Configured PyTorch optimizer.
        cfg: Optional configuration dictionary containing:
             - training.warmup_epochs: warmup duration in epochs (default 1)
             - training.epochs: total epochs (used if epochs arg is default)
             - training.min_lr: minimum learning rate for cosine (default 1e-6)
        epochs: Total number of training epochs.
        steps_per_epoch: Optimization steps per epoch.

    Returns:
        PyTorch lr_scheduler instance.
    """
    if torch is None:
        raise ImportError("PyTorch is required to build learning rate scheduler.")

    warmup_epochs = 1
    min_lr = 1e-6

    if cfg is not None:
        train_cfg = cfg.get("training", {})
        warmup_epochs = int(train_cfg.get("warmup_epochs", 1))
        epochs = int(train_cfg.get("epochs", epochs))
        min_lr = float(train_cfg.get("min_lr", 1e-6))

    if warmup_epochs <= 0 or epochs <= warmup_epochs:
        return CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=min_lr)

    warmup_sched = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warmup_epochs),
        eta_min=min_lr,
    )

    return SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_epochs],
    )
