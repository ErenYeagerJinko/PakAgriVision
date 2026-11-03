"""Hyperparameter search and tuning harness for PakAgriVision.

Phase 3 & 4 - Deliverable 2.
Implements the 13 structured tuning trials:
    - Stage A (Trials 1-6): Backbone {ResNet50, EfficientNet-B0} x Learning Rate {1e-3, 3e-4, 1e-4}
    - Stage B (Trials 7-13): Best of A varying Dropout {0.0, 0.5}, Batch Size {32, 128},
      Weight Decay {1e-2}, Label Smoothing {0.1}, and Class Weights toggle.

Protocol:
    - 20% stratified train subset (subset_frac=0.20)
    - 4 epochs per trial
    - Evaluated on full validation set by Macro-F1
    - Appends results into results/hparam_search.csv
    - Supports multi-account parallelization via --stage A|B --part i/3
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from src.train import load_yaml_config, train


# Default 13 structured trials specified for Deliverable 2
STAGE_A_TRIALS: List[Dict[str, Any]] = [
    {"trial_id": 1, "stage": "A", "backbone": "resnet50", "lr": 1e-3, "batch_size": 64, "dropout": 0.3, "weight_decay": 1e-4, "label_smoothing": 0.0, "use_class_weights": True},
    {"trial_id": 2, "stage": "A", "backbone": "resnet50", "lr": 3e-4, "batch_size": 64, "dropout": 0.3, "weight_decay": 1e-4, "label_smoothing": 0.0, "use_class_weights": True},
    {"trial_id": 3, "stage": "A", "backbone": "resnet50", "lr": 1e-4, "batch_size": 64, "dropout": 0.3, "weight_decay": 1e-4, "label_smoothing": 0.0, "use_class_weights": True},
    {"trial_id": 4, "stage": "A", "backbone": "efficientnet_b0", "lr": 1e-3, "batch_size": 64, "dropout": 0.3, "weight_decay": 1e-4, "label_smoothing": 0.0, "use_class_weights": True},
    {"trial_id": 5, "stage": "A", "backbone": "efficientnet_b0", "lr": 3e-4, "batch_size": 64, "dropout": 0.3, "weight_decay": 1e-4, "label_smoothing": 0.0, "use_class_weights": True},
    {"trial_id": 6, "stage": "A", "backbone": "efficientnet_b0", "lr": 1e-4, "batch_size": 64, "dropout": 0.3, "weight_decay": 1e-4, "label_smoothing": 0.0, "use_class_weights": True},
]

# Stage B trials benchmark against the best of Stage A (defaulting to ResNet50, lr=3e-4)
STAGE_B_TRIALS: List[Dict[str, Any]] = [
    {"trial_id": 7, "stage": "B", "backbone": "resnet50", "lr": 3e-4, "batch_size": 64, "dropout": 0.0, "weight_decay": 1e-4, "label_smoothing": 0.0, "use_class_weights": True},
    {"trial_id": 8, "stage": "B", "backbone": "resnet50", "lr": 3e-4, "batch_size": 64, "dropout": 0.5, "weight_decay": 1e-4, "label_smoothing": 0.0, "use_class_weights": True},
    {"trial_id": 9, "stage": "B", "backbone": "resnet50", "lr": 3e-4, "batch_size": 32, "dropout": 0.3, "weight_decay": 1e-4, "label_smoothing": 0.0, "use_class_weights": True},
    {"trial_id": 10, "stage": "B", "backbone": "resnet50", "lr": 3e-4, "batch_size": 128, "dropout": 0.3, "weight_decay": 1e-4, "label_smoothing": 0.0, "use_class_weights": True},
    {"trial_id": 11, "stage": "B", "backbone": "resnet50", "lr": 3e-4, "batch_size": 64, "dropout": 0.3, "weight_decay": 1e-2, "label_smoothing": 0.0, "use_class_weights": True},
    {"trial_id": 12, "stage": "B", "backbone": "resnet50", "lr": 3e-4, "batch_size": 64, "dropout": 0.3, "weight_decay": 1e-4, "label_smoothing": 0.1, "use_class_weights": True},
    {"trial_id": 13, "stage": "B", "backbone": "resnet50", "lr": 3e-4, "batch_size": 64, "dropout": 0.3, "weight_decay": 1e-4, "label_smoothing": 0.0, "use_class_weights": False},
]

ALL_TRIALS = STAGE_A_TRIALS + STAGE_B_TRIALS


def partition_trials(trials: List[Dict[str, Any]], part_str: str) -> List[Dict[str, Any]]:
    """Partition a list of trials into k equal chunks and return the requested index.

    Args:
        trials: List of trial definition dictionaries.
        part_str: Partition descriptor in format 'index/total' (e.g. '1/3', '2/3', '3/3').

    Returns:
        List of trial definitions assigned to this partition.
    """
    try:
        parts = part_str.strip().split("/")
        idx = int(parts[0])
        total = int(parts[1])
        if idx < 1 or idx > total:
            raise ValueError
    except Exception:
        raise ValueError(f"Invalid partition format '{part_str}'. Expected format like '1/3', '2/3', '3/3'.")

    # Split list into approximately equal slices
    n = len(trials)
    chunk_size = (n + total - 1) // total
    start = (idx - 1) * chunk_size
    end = min(n, start + chunk_size)
    return trials[start:end]


def build_trial_config(
    base_cfg: Dict[str, Any],
    trial: Dict[str, Any],
    epochs: int = 4,
    subset_frac: float = 0.2,
) -> Dict[str, Any]:
    """Merge trial hyperparameter overrides into base configuration.

    Args:
        base_cfg: Base YAML configuration dictionary.
        trial: Trial parameter dictionary.
        epochs: Number of epochs to train for each trial (default: 4).
        subset_frac: Training subset fraction (default: 0.2).

    Returns:
        Updated configuration dictionary.
    """
    cfg = copy.deepcopy(base_cfg)

    # Model overrides
    cfg.setdefault("model", {})
    cfg["model"]["backbone"] = trial["backbone"]
    cfg["model"]["dropout"] = trial["dropout"]

    # Training overrides
    cfg.setdefault("training", {})
    cfg["training"]["lr"] = trial["lr"]
    cfg["training"]["weight_decay"] = trial["weight_decay"]
    cfg["training"]["label_smoothing"] = trial["label_smoothing"]
    cfg["training"]["epochs"] = epochs
    cfg["training"]["warmup_epochs"] = 1

    # Data overrides
    cfg.setdefault("data", {})
    cfg["data"]["batch_size"] = trial["batch_size"]
    cfg["data"]["use_class_weights"] = trial["use_class_weights"]
    cfg["data"]["subset_frac"] = subset_frac

    return cfg


def save_trial_result(
    result_record: Dict[str, Any],
    output_csv: Path | str,
) -> None:
    """Safely append trial results into the hyperparameter search CSV file."""
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    row_df = pd.DataFrame([result_record])

    if csv_path.is_file():
        existing_df = pd.read_csv(csv_path)
        # Avoid duplicate trial entries if re-run
        if "trial_id" in existing_df.columns:
            existing_df = existing_df[existing_df["trial_id"] != result_record["trial_id"]]
        updated_df = pd.concat([existing_df, row_df], ignore_index=True)
        updated_df.sort_values(by="trial_id", inplace=True)
        updated_df.to_csv(csv_path, index=False)
    else:
        row_df.to_csv(csv_path, index=False)


def run_trials(
    trials: List[Dict[str, Any]],
    base_config_path: str | Path = "configs/base.yaml",
    output_csv_path: str | Path = "results/hparam_search.csv",
    epochs_per_trial: int = 4,
    subset_frac: float = 0.2,
    device_override: Optional[str] = None,
) -> pd.DataFrame:
    """Execute sequence of hyperparameter trials.

    Args:
        trials: List of trial dictionaries to execute.
        base_config_path: Path to baseline YAML configuration.
        output_csv_path: CSV destination path.
        epochs_per_trial: Number of epochs per trial (default: 4).
        subset_frac: Training data fraction (default: 0.2).
        device_override: Optional device string.

    Returns:
        DataFrame summarizing all executed trials.
    """
    base_cfg = load_yaml_config(base_config_path)
    results: List[Dict[str, Any]] = []

    print(f"\n=======================================================")
    print(f"PakAgriVision Hyperparameter Tuning Harness")
    print(f"Total Trials to Run: {len(trials)}")
    print(f"Epochs per Trial: {epochs_per_trial} | Train Subset: {subset_frac*100:.0f}%")
    print(f"Output CSV: {Path(output_csv_path).resolve()}")
    print(f"=======================================================\n")

    for i, trial in enumerate(trials, start=1):
        trial_id = trial["trial_id"]
        stage = trial["stage"]
        run_name = f"trial_{trial_id:02d}_stage_{stage}"
        trial_run_dir = Path("runs") / "tune" / run_name

        print(f"\n>>> Running Trial {i}/{len(trials)} [ID: {trial_id} | Stage {stage}]")
        print(f"    Params: Backbone={trial['backbone']} | LR={trial['lr']} | Batch={trial['batch_size']} | "
              f"Dropout={trial['dropout']} | WD={trial['weight_decay']} | Smooth={trial['label_smoothing']} | "
              f"ClassWeights={trial['use_class_weights']}")

        trial_cfg = build_trial_config(
            base_cfg=base_cfg,
            trial=trial,
            epochs=epochs_per_trial,
            subset_frac=subset_frac,
        )

        trial_start = time.time()
        summary, _ = train(
            cfg=trial_cfg,
            run_dir_override=trial_run_dir,
            subset_frac_override=subset_frac,
            device_override=device_override,
        )
        trial_duration = time.time() - trial_start

        record = {
            "trial_id": trial_id,
            "stage": stage,
            "backbone": trial["backbone"],
            "lr": trial["lr"],
            "batch_size": trial["batch_size"],
            "dropout": trial["dropout"],
            "weight_decay": trial["weight_decay"],
            "label_smoothing": trial["label_smoothing"],
            "use_class_weights": trial["use_class_weights"],
            "best_epoch": summary["best_epoch"],
            "val_loss": summary["best_val_loss"],
            "val_acc": summary["best_val_acc"],
            "val_macro_f1": summary["best_val_macro_f1"],
            "time_seconds": round(trial_duration, 1),
        }
        results.append(record)
        save_trial_result(record, output_csv_path)

        print(f"    -> Result: Val Loss: {record['val_loss']:.4f} | Val Acc: {record['val_acc']*100:.2f}% | "
              f"Val Macro-F1: {record['val_macro_f1']:.4f} in {trial_duration:.1f}s")

    return pd.DataFrame(results)


def parse_args() -> argparse.Namespace:
    """Parse command line options."""
    parser = argparse.ArgumentParser(description="PakAgriVision Hyperparameter Tuning")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["A", "B", "all"],
        default="all",
        help="Trial stage to execute: 'A' (backbone x LR), 'B' (regularization), or 'all' (default)",
    )
    parser.add_argument(
        "--part",
        type=str,
        default=None,
        help="Partition fraction for parallel training across GPU accounts (e.g. '1/3', '2/3', '3/3')",
    )
    parser.add_argument(
        "--trial",
        type=int,
        default=None,
        help="Run a single specific trial ID (1 to 13)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Base configuration YAML file (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/hparam_search.csv",
        help="Output CSV file path (default: results/hparam_search.csv)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of epochs per trial (default: 4)",
    )
    parser.add_argument(
        "--subset_frac",
        type=float,
        default=0.20,
        help="Fraction of stratified training split to use (default: 0.20)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device override ('cuda' or 'cpu')",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for tuning."""
    args = parse_args()

    # Filter trials by requested stage or trial ID
    if args.trial is not None:
        selected_trials = [t for t in ALL_TRIALS if t["trial_id"] == args.trial]
        if not selected_trials:
            print(f"Error: Trial ID {args.trial} not found (valid IDs: 1..13).")
            sys.exit(1)
    elif args.stage == "A":
        selected_trials = STAGE_A_TRIALS
    elif args.stage == "B":
        selected_trials = STAGE_B_TRIALS
    else:
        selected_trials = ALL_TRIALS

    # Apply partitioning if requested
    if args.part is not None:
        selected_trials = partition_trials(selected_trials, args.part)
        print(f"Partition {args.part}: Assigned {len(selected_trials)} trials.")

    if not selected_trials:
        print("No trials assigned to this execution partition.")
        return

    run_trials(
        trials=selected_trials,
        base_config_path=args.config,
        output_csv_path=args.output,
        epochs_per_trial=args.epochs,
        subset_frac=args.subset_frac,
        device_override=args.device,
    )


if __name__ == "__main__":
    main()

