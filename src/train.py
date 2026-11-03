"""Main training pipeline for PakAgriVision.

Phase 3 & 4 - Deliverable 2.
Supports:
    - Configuration driven execution via YAML configs
    - Backbone warm-up schedule (1 epoch frozen -> unfreeze all layers)
    - Mixed-precision training (FP16 via torch.amp) and gradient clipping
    - Early stopping monitored on validation macro-F1
    - Checkpointing (best.pt and last.pt) with session resume support (--resume)
    - Seeded, stratified train subsets (--subset_frac) for rapid tuning
    - Run logging: history.csv, config.yaml, and summary.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from src.config import DATA_DIR, ROOT, SEED, SPLITS_DIR
from src.data.dataset import get_dataloaders
from src.data.splits import load_class_names
from src.engine import EarlyStopping, train_one_epoch, validate
from src.losses import build_loss
from src.models.classifier import build_model
from src.optim import build_optimizer, build_scheduler


def setup_seed(seed: int = SEED) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    p = Path(config_path)
    if not p.is_file():
        raise FileNotFoundError(f"Configuration file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else {}


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: Path | str,
    last_filename: str = "last.pt",
    best_filename: str = "best.pt",
) -> None:
    """Save training checkpoint to disk.

    Args:
        state: Dictionary holding model weights, optimizer/scheduler state, epoch, and metadata.
        is_best: Whether this checkpoint achieved the highest validation score so far.
        checkpoint_dir: Directory where checkpoints are saved.
        last_filename: Filename for the most recent epoch checkpoint.
        best_filename: Filename for the best performing checkpoint.
    """
    save_dir = Path(checkpoint_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    last_path = save_dir / last_filename
    torch.save(state, last_path)

    if is_best:
        best_path = save_dir / best_filename
        torch.save(state, best_path)


def load_checkpoint(
    checkpoint_path: Path | str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, Any]:
    """Restore training state from a saved checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint .pt file.
        model: Target model to load weights into.
        optimizer: Optional optimizer to restore state into.
        scheduler: Optional learning rate scheduler to restore state into.
        scaler: Optional GradScaler to restore state into.

    Returns:
        Checkpoint dictionary containing metadata and saved metric scores.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Unwrap model if necessary
    target_model = model.model if hasattr(model, "model") else model
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Try loading directly into model wrapper or wrapped model
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        target_model.load_state_dict(state_dict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint


def get_stratified_train_subset(
    train_df: pd.DataFrame,
    subset_frac: float,
    seed: int = SEED,
) -> pd.DataFrame:
    """Sample a stratified fraction of training samples preserving class distribution.

    Args:
        train_df: Full training split DataFrame containing 'label_idx' or 'label'.
        subset_frac: Fraction of samples to retain (0.0 < subset_frac <= 1.0).
        seed: Random seed for deterministic sampling.

    Returns:
        Sampled training DataFrame with preserved columns.
    """
    if subset_frac >= 1.0 or subset_frac <= 0.0:
        return train_df.copy()

    group_col = "label_idx" if "label_idx" in train_df.columns else "label"
    sampled_indices: List[int] = []

    for _, group in train_df.groupby(group_col):
        n_samples = max(1, int(round(len(group) * subset_frac)))
        sampled = group.sample(n=min(len(group), n_samples), random_state=seed)
        sampled_indices.extend(sampled.index.tolist())

    # Return shuffled stratified subset
    subset = train_df.loc[sampled_indices].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return subset


def train(
    cfg: Dict[str, Any],
    run_dir_override: Optional[str | Path] = None,
    resume_checkpoint: Optional[str | Path] = None,
    subset_frac_override: Optional[float] = None,
    device_override: Optional[str] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Execute complete model training and validation workflow.

    Args:
        cfg: Configuration dictionary.
        run_dir_override: Optional override for the run directory.
        resume_checkpoint: Optional path to checkpoint to resume from.
        subset_frac_override: Optional override for training subset fraction.
        device_override: Optional device string ('cuda' or 'cpu').

    Returns:
        (summary_dict, history_dataframe)
    """
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    log_cfg = cfg.get("logging", {})

    seed = int(train_cfg.get("seed", SEED))
    setup_seed(seed)

    # Determine execution device
    if device_override:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine directories
    if run_dir_override:
        run_dir = Path(run_dir_override)
    else:
        run_dir = Path(log_cfg.get("run_dir", "runs/baseline"))
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save copy of active config
    config_save_path = run_dir / "config.yaml"
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)

    # Resolve split manifests
    train_manifest_path = Path(data_cfg.get("train_split", "splits/train.csv"))
    val_manifest_path = Path(data_cfg.get("val_split", "splits/val.csv"))
    test_manifest_path = Path(data_cfg.get("test_split", "splits/test.csv"))

    if not train_manifest_path.is_file():
        raise FileNotFoundError(f"Train split not found: {train_manifest_path}")
    if not val_manifest_path.is_file():
        raise FileNotFoundError(f"Val split not found: {val_manifest_path}")
    if not test_manifest_path.is_file():
        raise FileNotFoundError(f"Test split not found: {test_manifest_path}")

    train_df = pd.read_csv(train_manifest_path)
    val_df = pd.read_csv(val_manifest_path)
    test_df = pd.read_csv(test_manifest_path)

    # Handle training subset fraction
    subset_frac = subset_frac_override
    if subset_frac is None:
        subset_frac = float(data_cfg.get("subset_frac", 1.0))

    if subset_frac < 1.0:
        train_df = get_stratified_train_subset(train_df, subset_frac, seed=seed)

    # Windows-safe num_workers: default to 0 on Windows to avoid freeze/multiprocessing lock
    default_workers = 0 if sys.platform.startswith("win") else 4
    num_workers = int(data_cfg.get("num_workers", default_workers))
    batch_size = int(data_cfg.get("batch_size", 64))

    train_loader, val_loader, _ = get_dataloaders(
        train_manifest=train_df,
        val_manifest=val_df,
        test_manifest=test_df,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    # Build Model
    backbone = str(model_cfg.get("backbone", "resnet50"))
    num_classes = int(model_cfg.get("num_classes", 38))
    dropout = float(model_cfg.get("dropout", 0.3))
    pretrained = bool(model_cfg.get("pretrained", True))

    model = build_model(
        backbone=backbone,
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained,
    )
    model.to(device)

    # Load class names for checkpoint storage
    class_names_file = Path(SPLITS_DIR) / "class_names.json"
    class_names = load_class_names(class_names_file) if class_names_file.is_file() else []

    # Build Loss, Optimizer, and Scheduler
    criterion = build_loss(cfg=cfg, device=device)
    optimizer = build_optimizer(model=model, cfg=cfg)

    total_epochs = int(train_cfg.get("epochs", 15))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 1))
    scheduler = build_scheduler(optimizer=optimizer, cfg=cfg, epochs=total_epochs)

    # Mixed precision & gradient clipping
    use_mp = bool(train_cfg.get("mixed_precision", True)) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_mp else None
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))

    # Early stopping
    patience = int(train_cfg.get("early_stopping_patience", 4))
    early_stopping = EarlyStopping(patience=patience, mode="max")

    start_epoch = 1
    best_val_macro_f1 = -1.0
    history_records: List[Dict[str, Any]] = []

    # Resume from checkpoint if requested
    if resume_checkpoint:
        ckpt = load_checkpoint(
            checkpoint_path=resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_macro_f1 = float(ckpt.get("best_val_macro_f1", -1.0))
        early_stopping.best_score = best_val_macro_f1 if best_val_macro_f1 > 0 else None

    # Load existing history.csv if resuming
    history_csv_path = run_dir / "history.csv"
    if resume_checkpoint and history_csv_path.is_file():
        existing_history = pd.read_csv(history_csv_path)
        history_records = existing_history.to_dict(orient="records")

    best_epoch = start_epoch
    best_val_acc = 0.0
    best_val_loss = float("inf")
    start_total_time = time.time()
    early_stopped = False

    print(f"\n=======================================================")
    print(f"PakAgriVision Training: {backbone.upper()} on {device.type.upper()}")
    print(f"Total Epochs: {total_epochs} | Warmup Epochs: {warmup_epochs} | Batch: {batch_size}")
    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")
    print(f"Run directory: {run_dir.resolve()}")
    print(f"=======================================================\n")

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start_time = time.time()

        # Backbone warmup freeze/unfreeze schedule
        if epoch <= warmup_epochs:
            # Warm-up phase: freeze backbone, train custom classifier head only
            if hasattr(model, "set_backbone_trainable"):
                model.set_backbone_trainable(False)
        else:
            # End-to-end fine-tuning: unfreeze backbone
            if hasattr(model, "set_backbone_trainable"):
                model.set_backbone_trainable(True)

        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            grad_clip_norm=grad_clip_norm,
        )

        # Validate on full validation split
        val_loss, val_acc, val_macro_f1 = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start_time

        # Check for improvement
        is_best = val_macro_f1 > best_val_macro_f1
        if is_best:
            best_val_macro_f1 = val_macro_f1
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch

        # Record metrics
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5),
            "val_macro_f1": round(val_macro_f1, 5),
            "lr": round(current_lr, 8),
            "epoch_time_sec": round(epoch_time, 2),
        }
        history_records.append(record)

        # Update history.csv
        pd.DataFrame(history_records).to_csv(history_csv_path, index=False)

        # Save checkpoints
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "best_val_macro_f1": best_val_macro_f1,
            "config": cfg,
            "class_names": class_names,
        }
        save_checkpoint(
            state=checkpoint_state,
            is_best=is_best,
            checkpoint_dir=run_dir,
            last_filename="last.pt",
            best_filename="best.pt",
        )

        best_marker = " [*BEST*]" if is_best else ""
        print(
            f"Epoch {epoch:02d}/{total_epochs:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% F1: {val_macro_f1:.4f} | "
            f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s{best_marker}"
        )

        # Evaluate early stopping
        if early_stopping.step(val_macro_f1):
            print(
                f"\n[Early Stopping Triggered] Validation macro-F1 did not improve for "
                f"{patience} consecutive epochs. Terminating training at epoch {epoch}."
            )
            early_stopped = True
            break

    total_run_time = time.time() - start_total_time
    history_df = pd.DataFrame(history_records)

    summary = {
        "backbone": backbone,
        "device": device.type,
        "total_epochs_trained": len(history_records),
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_macro_f1, 5),
        "best_val_acc": round(best_val_acc, 5),
        "best_val_loss": round(best_val_loss, 5),
        "total_time_seconds": round(total_run_time, 2),
        "early_stopped": early_stopped,
        "checkpoint_best": str((run_dir / "best.pt").resolve()),
        "checkpoint_last": str((run_dir / "last.pt").resolve()),
    }

    summary_json_path = run_dir / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n================== Training Summary ==================")
    print(f"Best Epoch: {best_epoch} | Best Val Macro-F1: {best_val_macro_f1:.4f}")
    print(f"Best Checkpoint: {summary['checkpoint_best']}")
    print(f"Summary JSON: {summary_json_path}")
    print("======================================================\n")

    return summary, history_df


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="PakAgriVision Model Training Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to training configuration YAML file (default: configs/baseline.yaml)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pt file to resume training from",
    )
    parser.add_argument(
        "--subset_frac",
        type=float,
        default=None,
        help="Fraction of stratified training data to use (0.0 < frac <= 1.0)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override total number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Override output run directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device ('cuda' or 'cpu')",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line interface entry point."""
    args = parse_args()
    cfg = load_yaml_config(args.config)

    # Apply command-line overrides
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg.setdefault("data", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg.setdefault("training", {})["lr"] = args.lr

    train(
        cfg=cfg,
        run_dir_override=args.run_dir,
        resume_checkpoint=args.resume,
        subset_frac_override=args.subset_frac,
        device_override=args.device,
    )


if __name__ == "__main__":
    main()
