"""Visualization and plotting utilities for model training and tuning.

Phase 3 & 4 - Deliverable 2.
Generates publication-quality figures:
    - loss_curves.png: Train vs. validation loss across epochs.
    - accuracy_curves.png: Train vs. validation accuracy across epochs.
    - val_f1_curve.png: Validation Macro-F1 with unfreeze point and best epoch annotated.
    - hparam_search.png: Comparative evaluation of the 13 hyperparameter trials.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for headless/servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Set consistent styling
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_loss_curves(
    df: pd.DataFrame,
    output_path: Path | str,
    unfreeze_epoch: int = 1,
    best_epoch: Optional[int] = None,
) -> None:
    """Plot training vs. validation loss curves across epochs."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    epochs = df["epoch"].values
    train_loss = df["train_loss"].values
    val_loss = df["val_loss"].values

    ax.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=2, marker="o", markersize=4)
    ax.plot(epochs, val_loss, label="Validation Loss", color="#ff7f0e", linewidth=2, marker="s", markersize=4)

    # Annotate unfreeze point
    if unfreeze_epoch in epochs and len(epochs) > 1:
        ax.axvline(
            x=unfreeze_epoch,
            color="#2ca02c",
            linestyle="--",
            alpha=0.7,
            label=f"Backbone Unfreeze (Epoch {unfreeze_epoch})",
        )

    # Annotate best epoch
    if best_epoch is not None and best_epoch in epochs:
        ax.axvline(
            x=best_epoch,
            color="#d62728",
            linestyle=":",
            alpha=0.7,
            label=f"Best Checkpoint (Epoch {best_epoch})",
        )

    ax.set_title("Training and Validation Loss", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax.set_xticks(epochs)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#ddd")

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_accuracy_curves(
    df: pd.DataFrame,
    output_path: Path | str,
    unfreeze_epoch: int = 1,
    best_epoch: Optional[int] = None,
) -> None:
    """Plot training vs. validation accuracy curves across epochs."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    epochs = df["epoch"].values
    train_acc = df["train_acc"].values * 100.0
    val_acc = df["val_acc"].values * 100.0

    ax.plot(epochs, train_acc, label="Train Accuracy", color="#1f77b4", linewidth=2, marker="o", markersize=4)
    ax.plot(epochs, val_acc, label="Validation Accuracy", color="#ff7f0e", linewidth=2, marker="s", markersize=4)

    if unfreeze_epoch in epochs and len(epochs) > 1:
        ax.axvline(
            x=unfreeze_epoch,
            color="#2ca02c",
            linestyle="--",
            alpha=0.7,
            label=f"Backbone Unfreeze (Epoch {unfreeze_epoch})",
        )

    if best_epoch is not None and best_epoch in epochs:
        ax.axvline(
            x=best_epoch,
            color="#d62728",
            linestyle=":",
            alpha=0.7,
            label=f"Best Checkpoint (Epoch {best_epoch})",
        )

    ax.set_title("Training and Validation Accuracy", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xticks(epochs)
    ax.set_ylim([0, 105])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#ddd")

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_val_f1_curve(
    df: pd.DataFrame,
    output_path: Path | str,
    unfreeze_epoch: int = 1,
    best_epoch: Optional[int] = None,
) -> None:
    """Plot validation Macro-F1 score with unfreeze and best epoch annotations."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    epochs = df["epoch"].values
    val_f1 = df["val_macro_f1"].values

    ax.plot(epochs, val_f1, label="Validation Macro-F1", color="#6a1b9a", linewidth=2.2, marker="^", markersize=5)

    if unfreeze_epoch in epochs and len(epochs) > 1:
        ax.axvline(
            x=unfreeze_epoch,
            color="#2ca02c",
            linestyle="--",
            alpha=0.75,
            label=f"Backbone Unfreeze (Epoch {unfreeze_epoch})",
        )

    # Determine best epoch if not supplied
    if best_epoch is None and len(val_f1) > 0:
        best_idx = int(np.argmax(val_f1))
        best_epoch = int(epochs[best_idx])

    if best_epoch is not None and best_epoch in epochs:
        best_val = float(df.loc[df["epoch"] == best_epoch, "val_macro_f1"].values[0])
        ax.axvline(
            x=best_epoch,
            color="#d62728",
            linestyle=":",
            alpha=0.8,
            label=f"Best Macro-F1: {best_val:.4f} (Epoch {best_epoch})",
        )
        ax.scatter([best_epoch], [best_val], color="#d62728", s=80, zorder=5)

    ax.set_title("Validation Macro-F1 Progression", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Macro-F1 Score", fontsize=11)
    ax.set_xticks(epochs)
    ax.set_ylim([0, 1.05])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#ddd", loc="lower right")

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_all_training_curves(
    history_source: Union[Path, str, pd.DataFrame],
    output_dir: Union[Path, str] = "reports/figures",
    unfreeze_epoch: int = 1,
    best_epoch: Optional[int] = None,
) -> None:
    """Generate all standard training figures from a history DataFrame or CSV."""
    if isinstance(history_source, (Path, str)):
        p = Path(history_source)
        if not p.is_file():
            raise FileNotFoundError(f"History file not found: {p}")
        df = pd.read_csv(p)
    else:
        df = history_source.copy()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_loss_curves(df, out_dir / "loss_curves.png", unfreeze_epoch, best_epoch)
    plot_accuracy_curves(df, out_dir / "accuracy_curves.png", unfreeze_epoch, best_epoch)
    plot_val_f1_curve(df, out_dir / "val_f1_curve.png", unfreeze_epoch, best_epoch)
    print(f"Generated training plots in: {out_dir.resolve()}")


def plot_hparam_search(
    hparam_source: Union[Path, str, pd.DataFrame],
    output_path: Union[Path, str] = "reports/figures/hparam_search.png",
) -> None:
    """Generate comparative visualization of hyperparameter search trials."""
    if isinstance(hparam_source, (Path, str)):
        p = Path(hparam_source)
        if not p.is_file():
            raise FileNotFoundError(f"Hyperparameter search CSV not found: {p}")
        df = pd.read_csv(p)
    else:
        df = hparam_source.copy()

    if df.empty:
        print("Warning: Hyperparameter search DataFrame is empty. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    # Sort trials by trial_id
    df = df.sort_values(by="trial_id").reset_index(drop=True)
    trial_labels = [f"T{row.trial_id} ({row.stage})" for _, row in df.iterrows()]
    f1_scores = df["val_macro_f1"].values

    colors = ["#1565c0" if row.stage == "A" else "#ef6c00" for _, row in df.iterrows()]

    bars = ax.bar(trial_labels, f1_scores, color=colors, width=0.6, edgecolor="#333", linewidth=0.7)

    # Add score annotations above bars
    for bar, score in zip(bars, f1_scores):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.015,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            rotation=0,
        )

    # Legend patches
    import matplotlib.patches as mpatches
    patch_a = mpatches.Patch(color="#1565c0", label="Stage A (Backbone × LR)")
    patch_b = mpatches.Patch(color="#ef6c00", label="Stage B (Regularization / Head)")
    ax.legend(handles=[patch_a, patch_b], loc="lower right", frameon=True, facecolor="white")

    ax.set_title("Hyperparameter Search: Validation Macro-F1 across Trials", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Trial ID (Stage)", fontsize=11)
    ax.set_ylabel("Validation Macro-F1", fontsize=11)
    ax.set_ylim([0, max(1.0, float(np.max(f1_scores)) + 0.1)])
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Generated hyperparameter search plot: {out.resolve()}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="PakAgriVision Plotting Utility")
    parser.add_argument(
        "--history",
        type=str,
        default=None,
        help="Path to history.csv file",
    )
    parser.add_argument(
        "--hparams",
        type=str,
        default=None,
        help="Path to hparam_search.csv file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/figures",
        help="Output directory for training curves (default: reports/figures)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="reports/figures/hparam_search.png",
        help="Output path for hyperparameter search plot",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for plotting."""
    args = parse_args()

    if args.history is not None:
        plot_all_training_curves(args.history, output_dir=args.output_dir)

    if args.hparams is not None:
        plot_hparam_search(args.hparams, output_path=args.output_path)


if __name__ == "__main__":
    main()

