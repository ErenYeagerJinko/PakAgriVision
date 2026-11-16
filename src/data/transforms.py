"""Image transformation and data augmentation pipelines.

Provides standardized PyTorch transform pipelines:
1. Training transform: stochastic spatial augmentations and color jittering to simulate
   field camera variations and prevent background artifact overfitting.
2. Evaluation transform: deterministic resize, center crop, and ImageNet normalisation
   for validation, test, and field evaluation sets.
3. Denormalisation utility for visualization, Grad-CAM, and reporting.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
from torchvision import transforms

from src.config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE


def get_train_transforms(
    img_size: int = IMG_SIZE,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    use_randaug: bool = False,
    use_field_aug: bool = False,
) -> transforms.Compose:
    """Build the stochastic data augmentation pipeline for model training.

    Augmentations applied:
    - RandomResizedCrop: simulates varying camera distances and leaf framing.
    - RandomHorizontalFlip & RandomVerticalFlip: exploits rotational invariance.
    - RandomRotation (30 deg): simulates handheld mobile capture angles.
    - ColorJitter: simulates varied sunlight, shadows, and camera white-balance.
    - Optional RandAugment or FieldAug: automated diverse field augmentation policies.
    - Normalization: standard ImageNet mean and std.
    """
    if use_field_aug:
        transform_list: List[transforms.Transform] = [
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=35),
            transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.08),
            transforms.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), value="random"),
        ]
        return transforms.Compose(transform_list)

    transform_list: List[transforms.Transform] = [
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    ]

    if use_randaug:
        transform_list.append(transforms.RandAugment(num_ops=2, magnitude=9))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return transforms.Compose(transform_list)


def get_eval_transforms(
    img_size: int = IMG_SIZE,
    resize_size: int = 256,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> transforms.Compose:
    """Build the deterministic evaluation transform pipeline.

    Used strictly for validation, test, and field robustness evaluation.
    Zero random augmentation leaks into evaluation sets.
    """
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:
    """Invert ImageNet normalisation on a tensor for plotting or inspection.

    Args:
        tensor: Tensor of shape (C, H, W) or (B, C, H, W) normalized with mean/std.

    Returns:
        Tensor in range [0.0, 1.0].
    """
    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)

    if tensor.dim() == 4:
        mean_t = mean_t.unsqueeze(0)
        std_t = std_t.unsqueeze(0)

    denorm = tensor * std_t + mean_t
    return torch.clamp(denorm, 0.0, 1.0)
