"""Error analysis on the outputs of src/evaluate.py.

Phase 4 - Deliverable 2 (Error Analysis, report section 4.5).
Reads predictions.csv, per_class.csv and confusion_matrix.csv from an evaluation
output directory (no model is re-run) and writes:
    - top_confused_pairs.csv: the 10 largest off-diagonal confusion counts
    - worst_classes.csv: the 10 lowest-F1 classes with their main confusion partner
    - misclassified_samples.csv: the images shown in the grid
    - <split>_misclassified_grid.png (in --figures_dir): >= 12 misclassified images,
      highest-confidence errors first, with true/predicted class and confidence

The grid is for looking at the images; causes in the report must come from what the
images show, not from guesses.

Usage:
    python -m src.analysis.error_analysis --eval_dir results/baseline/val
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.config import DATA_DIR, RAW_PV, ROOT
from src.metrics import top_confused_pairs


def load_eval_outputs(eval_dir: Path | str) -> Dict[str, object]:
    """Load the files written by src/evaluate.py for one split."""
    eval_dir = Path(eval_dir)
    with open(eval_dir / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    cm_df = pd.read_csv(eval_dir / "confusion_matrix.csv", index_col=0)
    return {
        "metrics": metrics,
        "predictions": pd.read_csv(eval_dir / "predictions.csv"),
        "per_class": pd.read_csv(eval_dir / "per_class.csv"),
        "cm": cm_df.to_numpy(),
        "class_names": list(cm_df.index),
    }


def worst_classes(per_class: pd.DataFrame, cm: np.ndarray, class_names: Sequence[str], k: int = 10) -> pd.DataFrame:
    """The ``k`` lowest-F1 classes, each with the class it is most often predicted as.

    Ties on F1 are broken by lower recall, then by name, so the order is deterministic.
    """
    present = per_class.dropna(subset=["f1"]).copy()
    present = present.sort_values(["f1", "recall", "class_name"]).head(k)
    off_diag = np.array(cm, copy=True)
    np.fill_diagonal(off_diag, 0)

    partners, partner_counts, incoming_partners = [], [], []
    for idx in present["label_idx"]:
        row = off_diag[idx]
        col = off_diag[:, idx]
        partners.append(class_names[int(row.argmax())] if row.max() > 0 else "")
        partner_counts.append(int(row.max()))
        incoming_partners.append(class_names[int(col.argmax())] if col.max() > 0 else "")

    present["main_confused_as"] = partners
    present["n_confused_as"] = partner_counts
    present["main_false_positive_from"] = incoming_partners
    return present.reset_index(drop=True)


def select_misclassified(predictions: pd.DataFrame, n: int = 16) -> pd.DataFrame:
    """Pick misclassified images for the grid: highest-confidence errors first.

    At most two images per true class are taken first so one bad class does not fill
    the grid; remaining slots are filled with the next highest-confidence errors.
    """
    errors = predictions[~predictions["correct"].astype(bool)].sort_values("confidence", ascending=False)
    if errors.empty:
        return errors
    diverse = errors.groupby("true_label", group_keys=False).head(2).head(n)
    rest = errors.drop(diverse.index).head(max(0, n - len(diverse)))
    return pd.concat([diverse, rest]).sort_values("confidence", ascending=False).reset_index(drop=True)


def resolve_image(path_str: str) -> Optional[Path]:
    """Find an image listed in predictions.csv (val/test paths are relative to the PlantVillage color dir)."""
    p = Path(path_str)
    for candidate in (p, ROOT / p, RAW_PV / "color" / p, DATA_DIR / p):
        if candidate.is_file():
            return candidate
    return None


def _short(class_name: str) -> str:
    crop, _, disease = class_name.partition("___")
    return f"{crop}: {disease.replace('_', ' ')}"


def plot_misclassified_grid(samples: pd.DataFrame, title: str, output_path: Path, cols: int = 4) -> int:
    """Save a grid of misclassified images; returns how many images were drawn."""
    found = [(row, resolve_image(row["path"])) for _, row in samples.iterrows()]
    found = [(row, p) for row, p in found if p is not None]
    if not found:
        return 0
    rows = int(np.ceil(len(found) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.6 * rows), dpi=200)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (row, img_path) in zip(axes, found):
        with Image.open(img_path) as img:
            ax.imshow(img.convert("RGB"))
        caption = (
            f"True: {_short(row['true_label'])}\n"
            f"Pred: {_short(row['pred_label'])} ({row['confidence']:.2f})"
        )
        ax.set_title("\n".join(textwrap.fill(line, 34) for line in caption.split("\n")), fontsize=7)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path)
    plt.close(fig)
    return len(found)


def run_error_analysis(eval_dir: Path | str, figures_dir: Path | str, n_images: int = 16) -> Dict[str, object]:
    """Write the error-analysis tables and grid for one evaluated split; returns a summary."""
    eval_dir, figures_dir = Path(eval_dir), Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    data = load_eval_outputs(eval_dir)
    split = data["metrics"].get("split", eval_dir.name)
    class_names, cm, predictions = data["class_names"], data["cm"], data["predictions"]

    pairs = pd.DataFrame(top_confused_pairs(cm, class_names, k=10))
    pairs.to_csv(eval_dir / "top_confused_pairs.csv", index=False)

    worst = worst_classes(data["per_class"], cm, class_names, k=10)
    worst.to_csv(eval_dir / "worst_classes.csv", index=False)

    samples = select_misclassified(predictions, n=n_images)
    samples.to_csv(eval_dir / "misclassified_samples.csv", index=False)
    label = {"val": "Validation", "test": "Test", "field": "Field"}.get(split, split)
    n_drawn = plot_misclassified_grid(
        samples, f"{label}: misclassified images (highest confidence first)",
        figures_dir / f"{split}_misclassified_grid.png",
    )

    n_errors = int((~predictions["correct"].astype(bool)).sum())
    summary = {
        "split": split,
        "n_images": int(len(predictions)),
        "n_errors": n_errors,
        "error_rate": n_errors / len(predictions) if len(predictions) else 0.0,
        "n_high_confidence_errors": int(((~predictions["correct"].astype(bool)) & (predictions["confidence"] >= 0.9)).sum()),
        "n_grid_images": n_drawn,
    }
    with open(eval_dir / "error_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"{label}: {n_errors} errors of {len(predictions)} images "
          f"({summary['n_high_confidence_errors']} with confidence >= 0.9); grid shows {n_drawn}.")
    if not pairs.empty:
        print("Top confused pairs:")
        for _, r in pairs.iterrows():
            print(f"  {r['true_class']} -> {r['predicted_class']}: {r['count']}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Error analysis on src/evaluate.py outputs.")
    parser.add_argument("--eval_dir", type=str, required=True,
                        help="Evaluation output directory, e.g. results/baseline/val")
    parser.add_argument("--figures_dir", type=str, default="reports/figures")
    parser.add_argument("--n_images", type=int, default=16, help="Misclassified images in the grid (>= 12)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_error_analysis(args.eval_dir, args.figures_dir, n_images=max(12, args.n_images))


if __name__ == "__main__":
    main()
