"""Ensemble evaluation script combining predictions from multiple checkpoints.

Combines whole-image model (field_full) and cropped-leaf model (field_cropped)
with optional Test-Time Augmentation (TTA) to maximize field accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Sequence

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import ROOT, SPLITS_DIR
from src.data.splits import load_class_names
from src.evaluate import build_eval_dataset, load_model_from_checkpoint
from src.metrics import compute_all_metrics, compute_confusion_matrix, top_confused_pairs


@torch.no_grad()
def predict_with_tta(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_tta: bool = True,
) -> np.ndarray:
    """Predict softmax probabilities with optional Test-Time Augmentation (horizontal flip)."""
    model.eval()
    all_probs = []

    for images, _ in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits.float(), dim=1)

        if use_tta:
            # Horizontal flip TTA
            flipped_images = torch.flip(images, dims=[-1])
            flipped_logits = model(flipped_images)
            flipped_probs = torch.softmax(flipped_logits.float(), dim=1)
            probs = (probs + flipped_probs) / 2.0

        all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs)


def evaluate_ensemble(
    checkpoints: Sequence[str | Path],
    split: str = "field",
    use_tta: bool = True,
    device_name: str | None = None,
    out_dir: str | Path | None = None,
) -> dict:
    """Evaluate ensemble of checkpoints on the specified split."""
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    class_names = load_class_names(Path(SPLITS_DIR) / "class_names.json")

    dataset = build_eval_dataset(split)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    y_true = dataset.df["label_idx"].to_numpy()

    print(f"\n================ Running Multi-Model Ensemble ================")
    print(f"Target split: '{split}' ({len(dataset)} images) on {device.type.upper()}")
    print(f"Test-Time Augmentation (TTA): {'ENABLED' if use_tta else 'DISABLED'}")
    print(f"Models in ensemble ({len(checkpoints)}):")
    for cp in checkpoints:
        print(f"  - {cp}")
    print("==============================================================")

    model_probs: List[np.ndarray] = []

    for i, cp in enumerate(checkpoints, 1):
        cp_path = Path(cp)
        if not cp_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {cp_path}")

        print(f"Evaluating Model {i}/{len(checkpoints)}: {cp_path.parent.name} ...")
        model, _ = load_model_from_checkpoint(cp_path, class_names, device)
        probs = predict_with_tta(model, loader, device, use_tta=use_tta)
        model_probs.append(probs)

    # Average probabilities across all models
    ensemble_probs = np.mean(model_probs, axis=0)
    y_pred = ensemble_probs.argmax(axis=1)

    metrics = compute_all_metrics(y_true, y_pred, ensemble_probs, class_names)
    cm = compute_confusion_matrix(y_true, y_pred, len(class_names))

    print(f"\n================ Ensemble Evaluation Results ================")
    print(f"Field Top-1 Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Field Top-3 Accuracy: {metrics['top3_accuracy']*100:.2f}%")
    print(f"Macro Precision:      {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:         {metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score:       {metrics['macro_f1']:.4f}")
    print(f"Macro AUC (ROC):      {metrics['macro_auc_ovr']:.4f}")
    print("=============================================================\n")

    if out_dir:
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        with open(out_p / "ensemble_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved ensemble metrics to: {out_p / 'ensemble_metrics.json'}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multi-model ensemble with TTA.")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=["runs/field_full/best.pt", "runs/field_cropped/best.pt"],
        help="List of checkpoint paths to ensemble",
    )
    parser.add_argument("--split", choices=["val", "test", "field"], default="field")
    parser.add_argument("--no-tta", action="store_true", help="Disable Test-Time Augmentation")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="results/ensemble/field")
    args = parser.parse_args()

    evaluate_ensemble(
        checkpoints=args.checkpoints,
        split=args.split,
        use_tta=not args.no_tta,
        device_name=args.device,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
