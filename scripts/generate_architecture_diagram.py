"""Generate publication-quality model architecture diagram for Deliverable 2 report."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def create_architecture_diagram(output_pdf: Path, output_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.axis("off")

    # Background color
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    # Helper function to draw rounded boxes
    def draw_box(x, y, w, h, title, subtitle, color, edgecolor, shape_text="", text_color="#1A202C"):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=color,
            edgecolor=edgecolor,
            linewidth=1.8,
            zorder=3
        )
        ax.add_patch(box)
        # Title
        ax.text(
            x + w / 2, y + h * 0.65, title,
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            color=text_color, zorder=4
        )
        # Subtitle
        if subtitle:
            ax.text(
                x + w / 2, y + h * 0.35, subtitle,
                ha="center", va="center", fontsize=8.5,
                color="#4A5568", zorder=4
            )
        # Shape badge
        if shape_text:
            shape_box = patches.FancyBboxPatch(
                (x + w / 2 - 0.55, y - 0.22), 1.1, 0.16,
                boxstyle="round,pad=0.01,rounding_size=0.04",
                facecolor="#EDF2F7",
                edgecolor="#CBD5E0",
                linewidth=1.0,
                zorder=5
            )
            ax.add_patch(shape_box)
            ax.text(
                x + w / 2, y - 0.14, shape_text,
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="#2D3748", zorder=6
            )

    # Helper for drawing arrows
    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#4A5568",
                lw=2.0,
                mutation_scale=16,
                shrinkA=0, shrinkB=0
            ),
            zorder=2
        )
        if label:
            ax.text(
                (x1 + x2) / 2, ((y1 + y2) / 2) + 0.12, label,
                ha="center", va="bottom", fontsize=8.5, color="#718096", fontweight="semibold", zorder=4
            )

    # 1. Input Leaf
    draw_box(0.5, 4.0, 1.8, 1.4, "Input Image", "RGB Normalized\n(Mean / Std)", "#EBF8FF", "#3182CE", "3 × 224 × 224")

    draw_arrow(2.3, 4.7, 3.0, 4.7)

    # 2. ResNet50 Backbone (Large Container) - widened and clear padding
    backbone_box = patches.FancyBboxPatch(
        (3.0, 2.5), 5.2, 4.4,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor="#F7FAFC",
        edgecolor="#2B6CB0",
        linewidth=2.2,
        linestyle="--",
        zorder=2
    )
    ax.add_patch(backbone_box)
    # Title placed well inside container
    ax.text(5.6, 6.62, "ResNet-50 Feature Extractor", ha="center", va="center", fontsize=11.5, fontweight="bold", color="#2B6CB0")
    ax.text(5.6, 6.32, "(ImageNet Pretrained · Total Params: ~23.5M)", ha="center", va="center", fontsize=8.5, fontweight="semibold", color="#4A5568")
    ax.text(5.6, 6.05, "Exposes target_layer for Grad-CAM explainability", ha="center", va="center", fontsize=8, color="#718096")

    # Internal Backbone Stages
    draw_box(3.3, 4.7, 2.1, 0.95, "Conv1 & MaxPool", "7×7 conv, stride 2", "#FFFFFF", "#CBD5E0", "64 × 56 × 56")
    draw_arrow(4.35, 4.7, 4.35, 4.15)
    draw_box(3.3, 3.2, 2.1, 0.95, "Layer 1 & Layer 2", "7 Bottlenecks (16 convs)", "#FFFFFF", "#CBD5E0", "512 × 28 × 28")

    draw_arrow(5.4, 3.67, 5.8, 3.67)

    draw_box(5.8, 3.2, 2.1, 0.95, "Layer 3", "6 Bottlenecks (18 convs)", "#FFFFFF", "#CBD5E0", "1024 × 14 × 14")
    draw_arrow(6.85, 4.15, 6.85, 4.7)

    # Layer 4 (Grad-CAM layer highlighted)
    draw_box(5.8, 4.7, 2.1, 0.95, "Layer 4 (Bottleneck 3)", "Target Layer for Grad-CAM", "#FEFCBF", "#D69E2E", "2048 × 7 × 7")

    draw_arrow(8.2, 4.7, 8.9, 4.7)

    # 3. Global Average Pooling
    draw_box(8.9, 4.0, 1.9, 1.4, "Adaptive AvgPool", "Spatial Reduction\nKernel (7×7)", "#EDF2F7", "#718096", "2048 × 1 × 1")

    draw_arrow(10.8, 4.7, 11.5, 4.7)

    # 4. Custom Classification Head Container
    head_box = patches.FancyBboxPatch(
        (11.5, 2.5), 2.5, 4.4,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor="#F0FFF4",
        edgecolor="#38A169",
        linewidth=2.2,
        zorder=2
    )
    ax.add_patch(head_box)
    ax.text(12.75, 6.62, "Custom Head", ha="center", va="center", fontsize=11.5, fontweight="bold", color="#276749")
    ax.text(12.75, 6.32, "77,862 params", ha="center", va="center", fontsize=8.5, color="#48BB78")

    draw_box(11.75, 4.8, 2.0, 0.95, "Dropout Layer", "p = 0.3 regularization", "#FFFFFF", "#9AE6B4")
    draw_arrow(12.75, 4.8, 12.75, 4.25)
    draw_box(11.75, 3.3, 2.0, 0.95, "Linear Classifier", "In: 2048 -> Out: 38", "#FFFFFF", "#38A169", "38 classes")

    # Output Arrow and Softmax
    draw_arrow(12.75, 3.3, 12.75, 1.9)
    draw_box(11.45, 0.9, 2.6, 0.95, "Softmax Probabilities", "38 Crop Disease Classes", "#EBF8FF", "#3182CE")

    # 5. Training Strategy Callout (Bottom left)
    strat_box = patches.FancyBboxPatch(
        (0.5, 0.6), 10.3, 1.6,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor="#FFFFFF",
        edgecolor="#E2E8F0",
        linewidth=1.5,
        zorder=2
    )
    ax.add_patch(strat_box)
    ax.text(0.7, 1.9, "Two-Stage Progressive Fine-Tuning Strategy", fontsize=10.5, fontweight="bold", color="#2D3748")
    strat_desc = (
        "• Stage 1 (Warmup - Epoch 1): Backbone frozen (requires_grad=False). Only Custom Head is trained to stabilize random weights.\n"
        "• Stage 2 (Fine-Tuning - Epochs 2-15): Full network unfrozen (requires_grad=True). End-to-end backpropagation with AdamW\n"
        "  (LR=3e-4, weight_decay=1e-4) and Cosine Annealing learning rate schedule down to 1e-6."
    )
    ax.text(0.7, 1.05, strat_desc, fontsize=8.5, color="#4A5568", linespacing=1.4)

    # Title Banner
    ax.text(7.5, 7.8, "PakAgriVision Classifier Architecture (ResNet-50)", ha="center", va="center", fontsize=16, fontweight="bold", color="#1A202C")
    ax.text(7.5, 7.4, "Deliverable 2: Model Architecture, Custom Head, and Two-Stage Progressive Training", ha="center", va="center", fontsize=10, color="#718096")

    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.5)
    plt.tight_layout()

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.savefig(output_png, format="png", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Generated architecture diagram:\n  - {output_pdf}\n  - {output_png}")


if __name__ == "__main__":
    pdf_path = Path("reports/figures/architecture.pdf")
    png_path = Path("reports/figures/architecture.png")
    create_architecture_diagram(pdf_path, png_path)

