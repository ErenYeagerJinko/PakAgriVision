"""Bias check on the outputs of src/evaluate.py.

Phase 4 - Deliverable 2 (Bias Check, report section 4.6).
There are no human subjects in this project, so subgroups are crops, classes and
image sources. Reads a lab evaluation directory (val or test) and, optionally, a
field evaluation directory, and writes to --out_dir:
    - f1_vs_train_size.csv + Spearman correlation: is performance tied to class size?
    - per_crop.csv: accuracy and F1 grouped by crop (class-name prefix before '___')
    - healthy_vs_diseased.csv: healthy/diseased accuracy and the false-"healthy" rate
    - lab_vs_field.csv: lab vs field gap, with lab scores also restricted to the classes
      present in the field set so both sides cover the same classes
    - bias_check.json: all of the above in one file
Figures (in --figures_dir): bias_f1_vs_train_size.png, bias_per_crop.png, bias_lab_vs_field.png

Usage:
    python -m src.analysis.bias_check --lab_dir results/baseline/val --field_dir results/baseline/field
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import SPLITS_DIR
from src.metrics import compute_all_metrics


def crop_of(class_name: str) -> str:
    """Crop name from a PlantVillage class string, e.g. 'Tomato___Late_blight' -> 'Tomato'."""
    return class_name.split("___", 1)[0]


def is_healthy(class_name: str) -> bool:
    return class_name.endswith("___healthy")


def _load(eval_dir: Path) -> Dict[str, object]:
    with open(eval_dir / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    return {
        "metrics": metrics,
        "predictions": pd.read_csv(eval_dir / "predictions.csv"),
        "per_class": pd.read_csv(eval_dir / "per_class.csv"),
    }


def f1_vs_train_size(per_class: pd.DataFrame, train_df: pd.DataFrame) -> Dict[str, object]:
    """Join per-class F1 with training-set class counts and compute Spearman's rho."""
    counts = train_df["label_idx"].value_counts().rename("train_count")
    table = per_class.dropna(subset=["f1"]).set_index("label_idx").join(counts).reset_index()
    table = table[["label_idx", "class_name", "train_count", "support", "f1", "recall", "precision"]]
    rho, p_value = spearmanr(table["train_count"], table["f1"])
    return {
        "table": table,
        "spearman_rho": None if np.isnan(rho) else float(rho),
        "spearman_p": None if np.isnan(p_value) else float(p_value),
        "train_count_min": int(table["train_count"].min()),
        "train_count_max": int(table["train_count"].max()),
    }


def per_crop_table(predictions: pd.DataFrame, per_class: pd.DataFrame) -> pd.DataFrame:
    """Accuracy over each crop's images and mean/min F1 over its classes."""
    preds = predictions.assign(crop=predictions["true_label"].map(crop_of))
    classes = per_class.dropna(subset=["f1"]).assign(crop=lambda d: d["class_name"].map(crop_of))
    by_images = preds.groupby("crop").agg(n_images=("correct", "size"), accuracy=("correct", "mean"))
    by_classes = classes.groupby("crop").agg(
        n_classes=("f1", "size"), mean_f1=("f1", "mean"), min_f1=("f1", "min")
    )
    worst = classes.loc[classes.groupby("crop")["f1"].idxmin(), ["crop", "class_name"]].set_index("crop")
    table = by_images.join(by_classes).join(worst.rename(columns={"class_name": "weakest_class"}))
    return table.reset_index().sort_values("mean_f1")


def healthy_vs_diseased(predictions: pd.DataFrame) -> pd.DataFrame:
    """Accuracy for healthy vs diseased images and how often each is flipped to the other group.

    false_healthy_rate (diseased rows) is the share of diseased leaves predicted as any
    healthy class: the costly error, because the farmer would not treat the crop.
    false_disease_rate (healthy rows) is the share of healthy leaves predicted as a disease.
    """
    true_h = predictions["true_label"].map(is_healthy)
    pred_h = predictions["pred_label"].map(is_healthy)
    correct = predictions["correct"].astype(bool)
    rows = []
    for group, mask in (("healthy", true_h), ("diseased", ~true_h)):
        n = int(mask.sum())
        flipped = int((mask & (pred_h != true_h)).sum())
        rows.append({
            "group": group,
            "n_images": n,
            "accuracy": float(correct[mask].mean()) if n else np.nan,
            "n_flipped_to_other_group": flipped,
            "false_healthy_rate": flipped / n if (n and group == "diseased") else np.nan,
            "false_disease_rate": flipped / n if (n and group == "healthy") else np.nan,
        })
    return pd.DataFrame(rows)


def _metrics_from_predictions(predictions: pd.DataFrame, class_names: Sequence[str]) -> Dict[str, float]:
    """Accuracy, top-3 and macro-F1 recomputed from predictions.csv (no probabilities needed)."""
    m = compute_all_metrics(predictions["true_idx"], predictions["pred_idx"], None, class_names)
    top3 = [t in str(row).split("|") for t, row in zip(predictions["true_label"], predictions["top3_labels"])]
    return {
        "n_images": int(len(predictions)),
        "n_classes": int(m["n_classes_present"]),
        "accuracy": float(m["accuracy"]),
        "top3_accuracy": float(np.mean(top3)) if top3 else 0.0,
        "macro_f1": float(m["macro_f1"]),
    }


def lab_vs_field(lab: Dict[str, object], field: Dict[str, object], class_names: Sequence[str]) -> pd.DataFrame:
    """Lab (all classes), lab restricted to field classes, and field scores side by side."""
    lab_preds = lab["predictions"]
    field_preds = field["predictions"]
    field_classes = set(field_preds["true_idx"])
    lab_same = lab_preds[lab_preds["true_idx"].isin(field_classes)]
    rows = [
        {"set": f"lab ({lab['metrics'].get('split', 'lab')}, all classes)", **_metrics_from_predictions(lab_preds, class_names)},
        {"set": f"lab ({lab['metrics'].get('split', 'lab')}, field classes only)", **_metrics_from_predictions(lab_same, class_names)},
        {"set": "field", **_metrics_from_predictions(field_preds, class_names)},
    ]
    table = pd.DataFrame(rows)
    table["macro_auc_ovr"] = [lab["metrics"].get("macro_auc_ovr"), None, field["metrics"].get("macro_auc_ovr")]
    return table


def plot_f1_vs_size(table: pd.DataFrame, rho: Optional[float], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    colors = ["#2ca02c" if is_healthy(c) else "#1f77b4" for c in table["class_name"]]
    ax.scatter(table["train_count"], table["f1"], c=colors, s=30, edgecolor="white", linewidth=0.5)
    for _, r in table.nsmallest(2, "f1").iterrows():
        ax.annotate(r["class_name"].replace("___", ": ").replace("_", " "), (r["train_count"], r["f1"]),
                    fontsize=6, xytext=(4, -2), textcoords="offset points")
    rho_text = "n/a" if rho is None else f"{rho:.3f}"
    ax.set_title(f"Per-class F1 vs training images (Spearman rho = {rho_text})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Training images in class", fontsize=10)
    ax.set_ylabel("F1 score", fontsize=10)
    ax.scatter([], [], c="#1f77b4", label="Diseased class")
    ax.scatter([], [], c="#2ca02c", label="Healthy class")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_per_crop(table: pd.DataFrame, output_path: Path) -> None:
    data = table.sort_values("mean_f1")
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    ax.barh(data["crop"], data["mean_f1"], color="#3182bd", label="Mean class F1")
    ax.scatter(data["min_f1"], data["crop"], color="#d62728", zorder=3, s=18, label="Weakest class F1")
    ax.set_xlim(max(0.0, data["min_f1"].min() - 0.05), 1.0)
    ax.set_xlabel("F1 score", fontsize=10)
    ax.set_title("Performance by crop", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_lab_vs_field(table: pd.DataFrame, output_path: Path) -> None:
    metrics = ["accuracy", "top3_accuracy", "macro_f1"]
    x = np.arange(len(metrics))
    width = 0.8 / len(table)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    for i, (_, row) in enumerate(table.iterrows()):
        bars = ax.bar(x + i * width, [row[m] for m in metrics], width, label=row["set"])
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(x + width * (len(table) - 1) / 2)
    ax.set_xticklabels(["Accuracy", "Top-3 accuracy", "Macro-F1"])
    ax.set_ylim(0, 1.3)
    ax.set_title("Lab vs field performance", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="upper center", ncol=len(table))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_bias_check(
    lab_dir: Path | str,
    out_dir: Path | str,
    figures_dir: Path | str,
    field_dir: Optional[Path | str] = None,
    train_split: Path | str = SPLITS_DIR / "train.csv",
) -> Dict[str, object]:
    """Run every bias check, write tables/figures, and return the combined summary."""
    out_dir, figures_dir = Path(out_dir), Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    lab = _load(Path(lab_dir))
    class_names: List[str] = list(lab["per_class"]["class_name"])

    size = f1_vs_train_size(lab["per_class"], pd.read_csv(train_split))
    size["table"].round(6).to_csv(out_dir / "f1_vs_train_size.csv", index=False)
    plot_f1_vs_size(size["table"], size["spearman_rho"], figures_dir / "bias_f1_vs_train_size.png")

    crops = per_crop_table(lab["predictions"], lab["per_class"])
    crops.round(6).to_csv(out_dir / "per_crop.csv", index=False)
    plot_per_crop(crops, figures_dir / "bias_per_crop.png")

    health = healthy_vs_diseased(lab["predictions"])
    health.round(6).to_csv(out_dir / "healthy_vs_diseased.csv", index=False)

    summary: Dict[str, object] = {
        "lab_split": lab["metrics"].get("split"),
        "class_size": {k: v for k, v in size.items() if k != "table"},
        "per_crop": crops.round(6).to_dict(orient="records"),
        "healthy_vs_diseased": health.round(6).to_dict(orient="records"),
    }

    if field_dir is not None:
        field = _load(Path(field_dir))
        gap = lab_vs_field(lab, field, class_names)
        gap.round(6).to_csv(out_dir / "lab_vs_field.csv", index=False)
        plot_lab_vs_field(gap, figures_dir / "bias_lab_vs_field.png")
        summary["lab_vs_field"] = gap.round(6).to_dict(orient="records")
        summary["field_healthy_vs_diseased"] = healthy_vs_diseased(field["predictions"]).round(6).to_dict(orient="records")

    with open(out_dir / "bias_check.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=lambda o: None if o is None or (isinstance(o, float) and np.isnan(o)) else o)

    rho = size["spearman_rho"]
    print(f"F1 vs train size: Spearman rho = {rho if rho is None else round(rho, 3)} "
          f"(p = {size['spearman_p']}), train counts {size['train_count_min']}-{size['train_count_max']}")
    print(crops[["crop", "n_classes", "accuracy", "mean_f1", "min_f1", "weakest_class"]].to_string(index=False))
    print(health.to_string(index=False))
    if field_dir is not None:
        print(gap.to_string(index=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bias check on src/evaluate.py outputs.")
    parser.add_argument("--lab_dir", type=str, required=True, help="Lab eval dir, e.g. results/baseline/val")
    parser.add_argument("--field_dir", type=str, default=None, help="Field eval dir, e.g. results/baseline/field")
    parser.add_argument("--out_dir", type=str, default=None, help="Output dir (default: <lab_dir>/../bias_check)")
    parser.add_argument("--figures_dir", type=str, default="reports/figures")
    parser.add_argument("--train_split", type=str, default=str(SPLITS_DIR / "train.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or str(Path(args.lab_dir).parent / "bias_check")
    run_bias_check(args.lab_dir, out_dir, args.figures_dir, field_dir=args.field_dir, train_split=args.train_split)


if __name__ == "__main__":
    main()
