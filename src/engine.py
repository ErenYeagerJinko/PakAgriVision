"""Execution engine for model training, validation, and early stopping.

Phase 3 & 4 - Deliverable 2 (Engine & Execution).
Provides:
    - train_one_epoch: Single epoch training with mixed-precision and gradient clipping.
    - validate: Full-dataset validation evaluating loss, accuracy, and macro-F1.
    - EarlyStopping: Metric tracker managing training early termination.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


class EarlyStopping:
    """Early stopping monitor based on validation performance."""

    def __init__(
        self,
        patience: int = 4,
        min_delta: float = 1e-4,
        mode: str = "max",
    ) -> None:
        """
        Args:
            patience: Number of epochs to wait for improvement before stopping.
            min_delta: Minimum change in monitored metric to qualify as an improvement.
            mode: 'max' (higher is better, e.g. macro-F1) or 'min' (lower is better, e.g. loss).
        """
        if mode not in ("max", "min"):
            raise ValueError(f"Unsupported mode: '{mode}'. Must be 'max' or 'min'.")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.is_improved = False

    def step(self, current_score: float) -> bool:
        """Evaluate current score against best recorded score.

        Args:
            current_score: Metric value achieved in current epoch.

        Returns:
            True if early stopping criterion has been met, False otherwise.
        """
        if self.best_score is None:
            self.best_score = current_score
            self.is_improved = True
            self.counter = 0
            return False

        if self.mode == "max":
            improved = current_score > (self.best_score + self.min_delta)
        else:
            improved = current_score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = current_score
            self.is_improved = True
            self.counter = 0
        else:
            self.is_improved = False
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.is_improved = False


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_clip_norm: Optional[float] = 1.0,
    progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
) -> Tuple[float, float]:
    """Execute one training epoch over the given dataloader.

    Supports mixed-precision training (torch.cuda.amp) and gradient clipping.

    Args:
        model: PyTorch neural network module.
        dataloader: PyTorch DataLoader for training data.
        criterion: Loss function (e.g. CrossEntropyLoss).
        optimizer: Configured PyTorch optimizer (e.g. AdamW).
        device: Device to execute computation on (CPU or CUDA).
        scaler: Optional GradScaler for mixed precision (FP16).
        grad_clip_norm: Maximum norm for gradient clipping (None or <= 0 to disable).
        progress_callback: Optional callback receiving (batch_idx, total_batches, loss, acc).

    Returns:
        (average_train_loss, train_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct_samples = 0
    total_samples = 0

    use_amp = scaler is not None and device.type == "cuda"
    total_batches = len(dataloader)

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()

            if grad_clip_norm is not None and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()

            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            optimizer.step()

        # Track statistics
        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        preds = torch.argmax(outputs, dim=1)
        correct_samples += (preds == targets).sum().item()
        total_samples += batch_size

        if progress_callback is not None:
            cur_loss = running_loss / max(1, total_samples)
            cur_acc = correct_samples / max(1, total_samples)
            progress_callback(batch_idx + 1, total_batches, cur_loss, cur_acc)

    avg_loss = running_loss / max(1, total_samples)
    train_acc = correct_samples / max(1, total_samples)

    return avg_loss, train_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluate model on the validation dataset.

    Calculates:
    - Average validation loss across all samples
    - Overall classification accuracy
    - Macro-averaged F1 score computed across the complete dataset (NOT batch-averaged).

    Args:
        model: PyTorch neural network module.
        dataloader: PyTorch DataLoader for validation split.
        criterion: Loss function.
        device: Device to execute computation on.

    Returns:
        (val_loss, val_acc, val_macro_f1)
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    if total_samples == 0:
        return 0.0, 0.0, 0.0

    avg_loss = running_loss / total_samples
    preds_arr = np.array(all_preds, dtype=np.int64)
    targets_arr = np.array(all_targets, dtype=np.int64)

    val_acc = float((preds_arr == targets_arr).sum() / total_samples)
    val_macro_f1 = float(f1_score(targets_arr, preds_arr, average="macro", zero_division=0))

    return avg_loss, val_acc, val_macro_f1

