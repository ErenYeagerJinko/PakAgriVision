"""Generate visual sanity check grid for data augmentation and benchmark throughput.

Creates reports/figures/aug_examples.png displaying:
- 3 representative crop disease samples (e.g., Tomato Early Blight, Potato Late Blight, Healthy Pepper)
- 1 Original image + 7 stochastic augmentations per row
- Denormalised back to RGB for realistic visual inspection.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import IMAGENET_MEAN, IMAGENET_STD
from src.data.transforms import denormalize_tensor, get_train_transforms


def create_synthetic_leaf(
    crop_type: str,
    base_color: Tuple[int, int, int] = (34, 139, 34),
    lesion_type: str = "none",
) -> Image.Image:
    """Generate a realistic synthetic diseased leaf image for demonstration."""
    size = (256, 256)
    img = Image.new("RGB", size, (240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Draw leaf shape
    leaf_bbox = (30, 20, 226, 236)
    draw.ellipse(leaf_bbox, fill=base_color, outline=(20, 100, 20), width=2)

    # Central leaf vein
    draw.line([(128, 20), (128, 236)], fill=(144, 238, 144), width=3)
    for y in range(40, 220, 25):
        draw.line([(128, y), (70, y - 15)], fill=(120, 210, 120), width=1)
        draw.line([(128, y), (186, y - 15)], fill=(120, 210, 120), width=1)

    # Add disease lesions based on type
    if lesion_type == "target_spots":  # Early Blight (concentric rings)
        for center, rad in [((95, 90), 22), ((160, 150), 28), ((110, 180), 18)]:
            for r in range(rad, 2, -5):
                col = (60 + r * 4, 35 + r * 2, 10)
                draw.ellipse(
                    (center[0] - r, center[1] - r, center[0] + r, center[1] + r),
                    fill=col,
                    outline=(200, 190, 40),
                )
    elif lesion_type == "water_soaked":  # Late Blight (large necrotic dark patches)
        for bbox in [((60, 60), (120, 130)), ((130, 110), (200, 190))]:
            draw.ellipse(
                (bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]),
                fill=(45, 45, 30),
                outline=(160, 170, 70),
                width=3,
            )

    return img


def generate_augmentation_grid(output_path: Path) -> None:
    """Generate and save the 3x8 original + 7 augmentations figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = [
        ("Tomato (Early Blight)", create_synthetic_leaf("Tomato", (46, 125, 50), "target_spots")),
        ("Potato (Late Blight)", create_synthetic_leaf("Potato", (56, 142, 60), "water_soaked")),
        ("Pepper (Healthy)", create_synthetic_leaf("Pepper", (43, 160, 43), "none")),
    ]

    train_tf = get_train_transforms(img_size=224)

    fig, axes = plt.subplots(nrows=3, ncols=8, figsize=(18, 7.5), dpi=200)

    for row_idx, (class_name, orig_pil) in enumerate(samples):
        # Column 0: Original resized
        orig_resized = orig_pil.resize((224, 224))
        axes[row_idx, 0].imshow(orig_resized)
        axes[row_idx, 0].set_title("Original", fontsize=10, fontweight="bold")
        axes[row_idx, 0].set_ylabel(class_name, fontsize=10, fontweight="bold")
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

        # Columns 1-7: Augmented versions
        for col_idx in range(1, 8):
            augmented_tensor = train_tf(orig_pil)
            denorm = denormalize_tensor(augmented_tensor).permute(1, 2, 0).numpy()
            axes[row_idx, col_idx].imshow(denorm)
            axes[row_idx, col_idx].set_title(f"Aug {col_idx}", fontsize=9)
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

    plt.suptitle(
        "Data Augmentation Pipeline Sanity Grid\n"
        "(RandomResizedCrop [0.6-1.0], Flips, 30° Rotation, ColorJitter)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Augmentation sanity grid successfully saved to: {output_path}")


def benchmark_transforms(num_iterations: int = 500) -> float:
    """Benchmark data transform pipeline throughput in images/second."""
    sample = create_synthetic_leaf("Tomato", (46, 125, 50), "target_spots")
    train_tf = get_train_transforms(img_size=224)

    # Warmup
    for _ in range(20):
        _ = train_tf(sample)

    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = train_tf(sample)
    duration = time.perf_counter() - start

    fps = num_iterations / duration
    print(f"Benchmark: Processed {num_iterations} transforms in {duration:.2f}s ({fps:.1f} images/sec)")
    return fps


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate augmentation grid and benchmark transforms.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "reports" / "figures" / "aug_examples.png",
        help="Path where the augmentation figure is saved.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run transform throughput benchmark.",
    )
    args = parser.parse_args()

    generate_augmentation_grid(args.output_path)
    if args.benchmark:
        benchmark_transforms()
    return 0


if __name__ == "__main__":
    sys.exit(main())
