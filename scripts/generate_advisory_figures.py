"""Reproduce real prediction/Grad-CAM examples and the advisory workflow figure.

Uses the first manifest entry for each declared class/domain, without selecting
images by model performance. Does not rerun or overwrite held-out benchmark scores.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
from PIL import Image

from app.model_engine import ModelEngine
from app.treatment_advisor import build_advisory
from knowledge_base.kb_service import TreatmentKB


def main():
    figures = ROOT / "reports/figures"
    output = ROOT / "results/advisory_demo"
    figures.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    kb = TreatmentKB()
    checkpoint = ROOT / "runs/baseline/best.pt"
    engine = ModelEngine(checkpoint, list(kb.class_names))
    if not engine.ready:
        raise RuntimeError(engine.error)
    specs = [("val", "Tomato___Late_blight"), ("val", "Tomato___healthy"),
             ("field_test", "Tomato___Late_blight")]
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), gridspec_kw={"width_ratios": [1, 1, 1.65]})
    records = []
    for row, (split, label) in enumerate(specs):
        manifest = pd.read_csv(ROOT / f"splits/{split}.csv")
        sample = manifest.loc[manifest.label == label].iloc[0]
        relative = str(sample["filepath"] if "filepath" in sample else sample["path"])
        path = ROOT / relative if split == "field_test" else ROOT / "data/raw/plantvillage/color" / relative
        with Image.open(path) as source:
            image = source.convert("RGB")
        diagnosis = engine.predict(image)
        advisory = build_advisory(kb, diagnosis["predicted_class"], diagnosis["confidence"])
        overlay = engine.generate_gradcam(image, diagnosis["predicted_class"])
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"{'Field' if split == 'field_test' else 'Lab validation'}: reference label\n" + label.replace("___", ": ").replace("_", " "), fontsize=10)
        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title("Real Grad-CAM (predicted class)", fontsize=10)
        treatment = advisory["treatment"]
        summary = (
            f"Prediction: {diagnosis['predicted_class'].replace('___', ': ').replace('_', ' ')}\n"
            f"Confidence: {diagnosis['confidence']:.2f}% | Status: {advisory['status']}\n\n"
            + textwrap.fill("Immediate action: " + treatment["immediate_action"], 60) + "\n\n"
            + textwrap.fill("Chemical control: " + treatment["chemical_treatment"], 60) + "\n\n"
            + (textwrap.fill(advisory["warning"], 60) if advisory["low_confidence"] else
               "Follow local product labels; consult an agronomist.")
        )
        axes[row, 2].text(0, .98, summary, transform=axes[row, 2].transAxes,
                          va="top", fontsize=9, linespacing=1.35)
        for ax in axes[row]:
            ax.axis("off")
        records.append({"split": split, "path": path.relative_to(ROOT).as_posix(), "reference_label": label,
                        "prediction": diagnosis, "advisory": advisory})
    fig.suptitle("Prediction-to-advisory examples from the trained baseline", fontsize=16)
    fig.text(.5, .01, "Illustrative cases, not a performance estimate. Attention is not proof of disease; confidence is not severity.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=[0, .035, 1, .96])
    fig.savefig(figures / "advisory_prediction_examples.png", dpi=300)
    plt.close(fig)
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    (output / "predictions.json").write_text(json.dumps({
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(), "checkpoint_sha256": digest,
        "selection": "First manifest row for each fixed class/domain; no correctness filtering",
        "device": engine.device, "examples": records,
    }, indent=2), encoding="utf-8")
    workflow_figure(figures)
    for record in records:
        print(record["reference_label"], "->", record["prediction"]["predicted_class"],
              f"{record['prediction']['confidence']:.2f}%", record["advisory"]["status"])


def workflow_figure(figures):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set(xlim=(0, 12), ylim=(0, 6))
    ax.axis("off")
    boxes = [
        (0.2, 3.9, "Leaf photo\nJPEG / PNG / WebP"),
        (3.2, 3.9, "Validate + normalize\nRGB / size / request limits"),
        (6.2, 3.9, "Trained ResNet-50\nOriginal class probabilities"),
        (9.2, 3.9, "Exact class lookup\n38 schema-validated records"),
        (3.2, 1.1, "Confidence < 60%\nWithhold product advice\nRetake photo + verify in person"),
        (6.2, 1.1, "Confidence >= 60%\nHealthy: no treatment products\nDisease: sourced guidance"),
        (9.2, 1.1, "Advisory panels\nSources + label precautions\nAgronomist disclaimer"),
    ]
    for x, y, label in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 2.5, 1.25, boxstyle="round,pad=0.12",
                                    facecolor="#e8f5e9", edgecolor="#1b5e20", linewidth=1.5))
        ax.text(x + 1.25, y + .625, label, ha="center", va="center", fontsize=9)
    for start, end in [((2.85, 4.5), (3.05, 4.5)), ((5.85, 4.5), (6.05, 4.5)),
                       ((8.85, 4.5), (9.05, 4.5)), ((10.45, 3.75), (4.45, 2.5)),
                       ((10.45, 3.75), (7.45, 2.5)), ((5.7, 1.0), (9.05, 1.0)),
                       ((8.85, 1.75), (9.05, 1.75))]:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#455a64"})
    ax.text(6, .35, "Unavailable model / invalid KB: no diagnosis or treatment. Grad-CAM failure: explicit notice, no synthetic overlay.",
            ha="center", fontsize=10)
    ax.set_title("Local Flask application: confidence-aware advisory workflow", fontsize=15, pad=18)
    fig.tight_layout()
    fig.savefig(figures / "advisory_workflow.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
