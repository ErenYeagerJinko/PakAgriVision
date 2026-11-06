"""Evaluate a trained checkpoint on the val, test or field split.

Phase 4 - Deliverable 2 (Evaluation).
Develop and debug on --split val. Run --split test exactly once, on the final model.
The field split (splits/field_test.csv) is evaluation-only and never used for training.

Outputs written to --out_dir (default: results/<run_name>/<split>/):
    - metrics.json: summary metrics plus checkpoint/split metadata
    - per_class.csv: support, predicted count, precision, recall, F1, AUC per class
    - predictions.csv: path, true/predicted label, confidence, top-3 labels per image
    - confusion_matrix.csv: raw counts (rows = true, columns = predicted)
    - probs.npy: softmax probabilities (local only; *.npy is git-ignored)
Figures written to --figures_dir with the split as prefix:
    - <split>_confusion_matrix.png, <split>_roc_curves.png, <split>_per_class_f1.png

Usage:
    python -m src.evaluate --checkpoint runs/baseline/best.pt --split val
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from src.config import RAW_PV, ROOT, SPLITS_DIR
from src.data.dataset import FieldDataset, PlantDataset
from src.data.splits import load_class_names
from src.data.transforms import get_eval_transforms
from src.metrics import (
    compute_all_metrics,
    compute_confusion_matrix,
    per_class_table,
    top_confused_pairs,
)
from src.models.classifier import build_model

SPLIT_FILES = {
    "val": SPLITS_DIR / "val.csv",
    "test": SPLITS_DIR / "test.csv",
    "field": SPLITS_DIR / "field_test.csv",
}


def load_model_from_checkpoint(
    checkpoint_path: Path | str,
    class_names: Sequence[str],
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Rebuild the model described in a training checkpoint and load its weights.

    Refuses to continue if the checkpoint's class order differs from
    splits/class_names.json, since every label index would then be wrong.
    """
    ckpt = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model_cfg = ckpt.get("config", {}).get("model", {}) if isinstance(ckpt, dict) else {}

    ckpt_classes = ckpt.get("class_names") if isinstance(ckpt, dict) else None
    if ckpt_classes and list(ckpt_classes) != list(class_names):
        raise ValueError("Checkpoint class order does not match splits/class_names.json.")

    model = build_model(
        backbone=str(model_cfg.get("backbone", "resnet50")),
        num_classes=len(class_names),
        dropout=float(model_cfg.get("dropout", 0.3)),
        pretrained=False,
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()

    info = {
        "checkpoint": str(checkpoint_path),
        "backbone": str(model_cfg.get("backbone", "resnet50")),
        "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "best_val_macro_f1_at_training": ckpt.get("best_val_macro_f1") if isinstance(ckpt, dict) else None,
    }
    return model, info


def build_eval_dataset(split: str, limit: Optional[int] = None) -> PlantDataset:
    """Dataset for one split, always with the deterministic eval transform."""
    df = pd.read_csv(SPLIT_FILES[split])
    if limit:
        df = df.groupby("label_idx", group_keys=False).head(max(1, limit // df["label_idx"].nunique()))
        df = df.reset_index(drop=True)
    if split == "field":
        return FieldDataset(df, root_dir=ROOT, transform=get_eval_transforms())
    return PlantDataset(df, root_dir=RAW_PV / "color", transform=get_eval_transforms())


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader; returns (true labels, softmax probabilities)."""
    model.eval()
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    total = len(loader)
    for i, (images, targets) in enumerate(loader, 1):
        logits = model(images.to(device))
        all_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        all_targets.append(targets.numpy())
        if i % 10 == 0 or i == total:
            print(f"  batch {i}/{total}", flush=True)
    return np.concatenate(all_targets), np.concatenate(all_probs)


def predictions_frame(
    paths: Sequence[str], y_true: np.ndarray, probs: np.ndarray, class_names: Sequence[str]
) -> pd.DataFrame:
    """One row per image: true and predicted class, confidence and top-3 predictions."""
    y_pred = probs.argmax(axis=1)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    return pd.DataFrame(
        {
            "path": list(paths),
            "true_idx": y_true,
            "true_label": [class_names[i] for i in y_true],
            "pred_idx": y_pred,
            "pred_label": [class_names[i] for i in y_pred],
            "confidence": probs.max(axis=1).round(6),
            "true_class_prob": probs[np.arange(len(y_true)), y_true].round(6),
            "correct": y_pred == y_true,
            "top3_labels": ["|".join(class_names[j] for j in row) for row in top3],
        }
    )


def short_name(class_name: str) -> str:
    """Compact label for plots, e.g. 'Tomato___Late_blight' -> 'Tomato: Late blight'."""
    crop, _, disease = class_name.partition("___")
    return f"{crop}: {disease.replace('_', ' ')}"


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], title: str, output_path: Path) -> None:
    """Row-normalised confusion matrix heatmap (each row sums to 1 where the class is present)."""
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
    labels = [short_name(c) for c in class_names]
    fig, ax = plt.subplots(figsize=(16, 14), dpi=200)
    sns.heatmap(norm, cmap="Blues", vmin=0, vmax=1, square=True, xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "Fraction of true class", "shrink": 0.7}, ax=ax)
    ax.set_xlabel("Predicted class", fontsize=12)
    ax.set_ylabel("True class", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(axis="both", labelsize=7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_roc_curves(y_true: np.ndarray, probs: np.ndarray, class_names: Sequence[str],
                    per_class: pd.DataFrame, title: str, output_path: Path) -> None:
    """One-vs-rest ROC curve per class (thin lines), highlighting the 3 lowest-AUC classes."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=200)
    defined = per_class.dropna(subset=["auc"])
    worst = set(defined.nsmallest(3, "auc")["label_idx"])
    for c in defined["label_idx"]:
        fpr, tpr, _ = roc_curve(y_true == c, probs[:, c])
        if c in worst:
            ax.plot(fpr, tpr, linewidth=1.8,
                    label=f"{short_name(class_names[c])} (AUC {per_class.loc[c, 'auc']:.4f})")
        else:
            ax.plot(fpr, tpr, color="#9ecae1", linewidth=0.7, alpha=0.8)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
    ax.set_xlabel("False positive rate", fontsize=11)
    ax.set_ylabel("True positive rate", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=8, title="Lowest-AUC classes", title_fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_per_class_f1(per_class: pd.DataFrame, title: str, output_path: Path) -> None:
    """Horizontal bar chart of per-class F1, sorted, for classes present in the split."""
    data = per_class.dropna(subset=["f1"]).sort_values("f1")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.22 * len(data))), dpi=200)
    ax.barh([short_name(c) for c in data["class_name"]], data["f1"], color="#3182bd")
    ax.set_xlim(max(0.0, data["f1"].min() - 0.05), 1.0)
    ax.set_xlabel("F1 score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def evaluate(
    checkpoint: Path | str,
    split: str,
    out_dir: Path | str,
    figures_dir: Path | str,
    batch_size: int = 64,
    num_workers: int = 0,
    device_name: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate ``checkpoint`` on ``split`` and write all outputs; returns the metrics dict."""
    out_dir, figures_dir = Path(out_dir), Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    class_names = load_class_names(SPLITS_DIR / "class_names.json")

    model, ckpt_info = load_model_from_checkpoint(checkpoint, class_names, device)
    dataset = build_eval_dataset(split, limit=limit)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"Evaluating {ckpt_info['backbone']} on '{split}' ({len(dataset)} images, {device.type})")

    start = time.time()
    y_true, probs = predict(model, loader, device)
    y_pred = probs.argmax(axis=1)

    metrics = compute_all_metrics(y_true, y_pred, probs, class_names)
    per_class = per_class_table(y_true, y_pred, probs, class_names)
    cm = compute_confusion_matrix(y_true, y_pred, len(class_names))
    metrics.update({
        "split": split,
        "split_file": str(SPLIT_FILES[split].relative_to(ROOT).as_posix()),
        "limit": limit,
        "eval_seconds": round(time.time() - start, 1),
        "top_confused_pairs": top_confused_pairs(cm, class_names, k=10),
        **ckpt_info,
    })

    paths = dataset.df[dataset.path_col].astype(str).tolist()
    predictions_frame(paths, y_true, probs, class_names).to_csv(out_dir / "predictions.csv", index=False)
    per_class.round(6).to_csv(out_dir / "per_class.csv", index=False)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out_dir / "confusion_matrix.csv")
    np.save(out_dir / "probs.npy", probs.astype(np.float32))
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    label = {"val": "Validation", "test": "Test", "field": "Field"}[split]
    plot_confusion_matrix(cm, class_names, f"{label} Confusion Matrix (row-normalised)",
                          figures_dir / f"{split}_confusion_matrix.png")
    plot_per_class_f1(per_class, f"{label} Per-Class F1", figures_dir / f"{split}_per_class_f1.png")
    if metrics["n_auc_classes"]:
        plot_roc_curves(y_true, probs, class_names, per_class, f"{label} One-vs-Rest ROC Curves",
                        figures_dir / f"{split}_roc_curves.png")

    auc = metrics["macro_auc_ovr"]
    print(
        f"\n{label}: acc {metrics['accuracy']:.4f} | top-3 {metrics['top3_accuracy']:.4f} | "
        f"macro P/R/F1 {metrics['macro_precision']:.4f}/{metrics['macro_recall']:.4f}/{metrics['macro_f1']:.4f} | "
        f"macro AUC {auc if auc is None else round(auc, 4)} ({metrics['n_auc_classes']} classes)"
    )
    print(f"Outputs: {out_dir}  Figures: {figures_dir}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PakAgriVision checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="runs/baseline/best.pt",
                        help="Path to checkpoint .pt (default: runs/baseline/best.pt)")
    parser.add_argument("--split", choices=sorted(SPLIT_FILES), default="val",
                        help="Split to evaluate. Use 'test' only once, on the final model.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: results/<run_name>/<split>)")
    parser.add_argument("--figures_dir", type=str, default="reports/figures",
                        help="Directory for figures (default: reports/figures)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu' (default: auto)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Debug only: evaluate a small class-balanced subset of about N images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split == "test":
        print("NOTE: the test split must be evaluated only once, on the final model.", file=sys.stderr)
    run_name = Path(args.checkpoint).resolve().parent.name
    out_dir = args.out_dir or f"results/{run_name}/{args.split}"
    evaluate(
        checkpoint=args.checkpoint,
        split=args.split,
        out_dir=out_dir,
        figures_dir=args.figures_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
