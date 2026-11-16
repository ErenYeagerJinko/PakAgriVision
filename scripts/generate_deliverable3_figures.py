"""Generate publication-quality figures for Deliverable 3 report and defense presentation.

Produces:
    1. reports/figures/domain_adaptation_progression.png
    2. reports/figures/app_pipeline.png
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set high-quality styling
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})


def plot_domain_adaptation_progression():
    """Plot Top-1 / Top-3 accuracy leap and macro metrics across all 6 model milestones."""
    stages = [
        "1. Day 2 Baseline\n(Lab Only)",
        "2. Few-Shot\n(57 Field)",
        "3. Full Field\n(~2k Images)",
        "4. Leaf-Cropped\n(8.4k Crops)",
        "5. Two-Stage\n(YOLO+ResNet)",
        "6. Cascade+\nCrop Prior",
    ]

    top1 = [24.73, 33.33, 54.84, 53.76, 67.74, 80.65]
    top3 = [46.24, 55.91, 82.80, 78.49, 90.32, 96.77]
    precision = [0.2204, 0.3424, 0.5829, 0.6105, 0.5980, 0.8240]
    f1 = [0.1970, 0.2938, 0.5355, 0.5025, 0.5420, 0.7910]
    auc = [0.8001, 0.8558, 0.9524, 0.9459, 0.9610, 0.9850]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # ── Panel 1: Top-1 vs Top-3 Accuracy Leap ──
    x = np.arange(len(stages))
    width = 0.36

    rects1 = ax1.bar(x - width/2, top1, width, label="Top-1 Accuracy (%)", color="#2e7d32", alpha=0.9, edgecolor="black", linewidth=0.8)
    rects2 = ax1.bar(x + width/2, top3, width, label="Top-3 Accuracy (%)", color="#81c784", alpha=0.9, edgecolor="black", linewidth=0.8)

    ax1.set_ylabel("Accuracy on Held-Out Field Test Set (%)")
    ax1.set_title("(a) Field Robustness Progression (93 Test Images)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, rotation=0)
    ax1.set_ylim(0, 110)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

    # Add value labels
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5)

    # Annotate leap
    ax1.annotate(
        "Day 2 Baseline Failure\n(24.73% Top-1)",
        xy=(0, 24.73), xytext=(0.2, 45),
        arrowprops=dict(facecolor="#d32f2f", shrink=0.08, width=1.5, headwidth=6),
        ha="center", fontsize=8.5, color="#b71c1c", fontweight="bold"
    )
    ax1.annotate(
        "Final Deployment\n80.65% Top-1 / 96.77% Top-3",
        xy=(5, 96.77), xytext=(4.3, 102),
        arrowprops=dict(facecolor="#1b5e20", shrink=0.08, width=1.5, headwidth=6),
        ha="center", fontsize=8.5, color="#1b5e20", fontweight="bold"
    )

    # ── Panel 2: Metric Quality (Precision, F1, Macro AUC) ──
    ax2.plot(x, [p * 100 for p in precision], marker="s", linewidth=2.2, color="#1565c0", label="Macro Precision (%)")
    ax2.plot(x, [f * 100 for f in f1], marker="o", linewidth=2.2, color="#e65100", label="Macro F1-Score (%)")
    ax2.plot(x, [a * 100 for a in auc], marker="^", linewidth=2.2, color="#6a1b9a", linestyle="--", label="Macro AUC-ROC (%)")

    ax2.set_ylabel("Metric Value (%)")
    ax2.set_title("(b) Multi-Metric Generalization Progression")
    ax2.set_xticks(x)
    ax2.set_xticklabels(stages, rotation=0)
    ax2.set_ylim(0, 110)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)

    for i in range(len(stages)):
        ax2.annotate(f"{auc[i]*100:.1f}", (x[i], auc[i]*100 + 2.5), ha="center", fontsize=8, color="#6a1b9a")
        ax2.annotate(f"{precision[i]*100:.1f}", (x[i], precision[i]*100 - 5.5), ha="center", fontsize=8, color="#1565c0")

    plt.tight_layout()
    out_path = FIGURES_DIR / "domain_adaptation_progression.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Generated {out_path} ({out_path.stat().st_size:,} bytes)")


from matplotlib.patches import FancyBboxPatch

def plot_app_pipeline():
    """Generate pipeline workflow schematic diagram."""
    fig, ax = plt.subplots(figsize=(13, 3.8), dpi=300)
    ax.axis("off")

    boxes = [
        {"x": 0.02, "w": 0.14, "title": "1. Input Photo\n(Farm Capture)", "desc": "Unconstrained phone photo\nwith soil, hands, & sky", "color": "#e0f2fe", "edge": "#0284c7"},
        {"x": 0.19, "w": 0.16, "title": "2. YOLOv8n Stage\n(Leaf Detection)", "desc": "Locates primary leaf\n100% test detection rate\nCrops with 5% margin", "color": "#fef08a", "edge": "#ca8a04"},
        {"x": 0.38, "w": 0.16, "title": "3. ResNet-50 Stage\n(Pathology CNN)", "desc": "Trained on 8.4k leaf crops\nFilters background noise\nOutputs 38-class logits", "color": "#bbf7d0", "edge": "#16a34a"},
        {"x": 0.57, "w": 0.18, "title": "4. Contextual Prior\n(Crop Filter)", "desc": "User selects host crop\nFilters false cross-alarms\nBoosts to 80.65% Top-1", "color": "#fed7aa", "edge": "#ea580c"},
        {"x": 0.78, "w": 0.20, "title": "5. UI & Advisory\n(Flask Dashboard)", "desc": "Layer 4 Grad-CAM heatmap\nTop-3 differential diagnosis\nUrdu/English treatment KB", "color": "#e9d5ff", "edge": "#9333ea"},
    ]

    for b in boxes:
        rect = FancyBboxPatch((b["x"], 0.12), b["w"], 0.76, boxstyle="round,pad=0.02",
                              facecolor=b["color"], edgecolor=b["edge"], linewidth=2, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(b["x"] + b["w"]/2, 0.72, b["title"], ha="center", va="center", transform=ax.transAxes,
                fontsize=11, fontweight="bold", color="#0f172a")
        ax.text(b["x"] + b["w"]/2, 0.38, b["desc"], ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#334155", multialignment="center")

    # Add connecting arrows
    for i in range(len(boxes) - 1):
        x_start = boxes[i]["x"] + boxes[i]["w"]
        x_end = boxes[i+1]["x"]
        ax.annotate(
            "", xy=(x_end, 0.5), xytext=(x_start, 0.5), xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", lw=2.5, color="#475569")
        )

    plt.tight_layout()
    out_path = FIGURES_DIR / "app_pipeline.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Generated {out_path} ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    plot_domain_adaptation_progression()
    plot_app_pipeline()
