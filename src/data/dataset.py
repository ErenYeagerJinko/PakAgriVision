"""PyTorch Dataset implementations and DataLoader builders.

Includes:
- PlantDataset: Standard dataset loader reading split manifests (train/val/test).
- FieldDataset: Dataset loader tailored for real field image robustness sets.
- create_weighted_sampler: Balanced sampler supporting cost-sensitive batching.
- get_dataloaders: Reproducible DataLoader factory with Windows-safe defaults.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.config import DATA_DIR, RAW_PV, SEED
from src.data.transforms import get_eval_transforms, get_train_transforms


def seed_worker(worker_id: int) -> None:
    """Worker initialisation function to guarantee reproducible DataLoader multi-processing."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class PlantDataset(Dataset):
    """PyTorch Dataset loading plant leaf images from manifest CSVs or DataFrames."""

    def __init__(
        self,
        csv_source: Union[Path, str, pd.DataFrame],
        root_dir: Optional[Union[Path, str]] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            csv_source: Path to manifest CSV file, or an existing pandas DataFrame.
            root_dir: Root directory prepended to relative paths. If None, paths are
                      resolved relative to RAW_PV / "color" or repository ROOT.
            transform: PyTorch torchvision transform pipeline.
        """
        if isinstance(csv_source, (Path, str)):
            p = Path(csv_source)
            if not p.exists():
                raise FileNotFoundError(f"Manifest CSV not found: {p}")
            self.df = pd.read_csv(p)
        elif isinstance(csv_source, pd.DataFrame):
            self.df = csv_source.copy().reset_index(drop=True)
        else:
            raise TypeError(f"Unsupported csv_source type: {type(csv_source)}")

        # Detect image path column ('path' or 'filepath')
        if "path" in self.df.columns:
            self.path_col = "path"
        elif "filepath" in self.df.columns:
            self.path_col = "filepath"
        else:
            raise ValueError(f"Manifest must contain a 'path' or 'filepath' column. Found: {list(self.df.columns)}")

        if "label_idx" not in self.df.columns:
            raise ValueError("Manifest must contain a 'label_idx' column.")

        self.root_dir = Path(root_dir) if root_dir else None
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, rel_path_str: str) -> Path:
        """Resolve full image path."""
        p = Path(rel_path_str)
        if p.is_absolute() and p.exists():
            return p

        # Check with explicit root_dir if provided
        if self.root_dir:
            candidate = self.root_dir / p
            if candidate.exists():
                return candidate

        # Check standard default raw locations
        for base in [RAW_PV / "color", RAW_PV, DATA_DIR, Path.cwd()]:
            candidate = base / p
            if candidate.exists():
                return candidate

        # Return root_dir / p or p if not found on disk yet (for graceful error handling)
        return (self.root_dir / p) if self.root_dir else p

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(str(row[self.path_col]))
        label_idx = int(row["label_idx"])

        try:
            with Image.open(img_path) as img:
                rgb_img = img.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Error loading image at index {idx} ({img_path}): {exc}") from exc

        if self.transform is not None:
            image_tensor = self.transform(rgb_img)
        else:
            # Default to tensor conversion if no transform provided
            image_tensor = torch.from_numpy(np.array(rgb_img)).permute(2, 0, 1).float() / 255.0

        target_tensor = torch.tensor(label_idx, dtype=torch.long)
        return image_tensor, target_tensor


class FieldDataset(PlantDataset):
    """Dataset specialized for real field evaluation sets (strictly uses eval transforms)."""

    def __init__(
        self,
        csv_source: Union[Path, str, pd.DataFrame],
        root_dir: Optional[Union[Path, str]] = None,
        transform: Optional[Callable] = None,
    ):
        eval_transform = transform if transform is not None else get_eval_transforms()
        super().__init__(csv_source=csv_source, root_dir=root_dir, transform=eval_transform)


def create_weighted_sampler(
    dataset: PlantDataset,
    class_weights: Union[Sequence[float], Dict[int, float]],
) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler based on class weights for cost-sensitive training.

    Args:
        dataset: The PlantDataset instance (train split).
        class_weights: Sequence or mapping of class index to positive weight.

    Returns:
        WeightedRandomSampler configured with per-sample weights.
    """
    labels = dataset.df["label_idx"].values
    if isinstance(class_weights, dict):
        sample_weights = [class_weights.get(int(y), 1.0) for y in labels]
    else:
        weights_arr = np.array(class_weights)
        sample_weights = [weights_arr[int(y)] for y in labels]

    weights_tensor = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=len(weights_tensor),
        replacement=True,
    )


def get_dataloaders(
    train_manifest: Union[Path, str, pd.DataFrame],
    val_manifest: Union[Path, str, pd.DataFrame],
    test_manifest: Union[Path, str, pd.DataFrame],
    root_dir: Optional[Union[Path, str]] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = SEED,
    use_weighted_sampler: bool = False,
    class_weights: Optional[Sequence[float]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Factory creating reproducible train, val, and test DataLoaders.

    Args:
        train_manifest: Path or DataFrame for training split.
        val_manifest: Path or DataFrame for validation split.
        test_manifest: Path or DataFrame for test split.
        root_dir: Base directory containing raw image folders.
        batch_size: Number of images per batch (default: 32).
        num_workers: Worker processes for loading. Default 0 (Windows-safe).
        seed: Random seed for generator reproducibility.
        use_weighted_sampler: Whether to sample training batches inversely to class frequency.
        class_weights: Weights required if use_weighted_sampler is True.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = PlantDataset(
        csv_source=train_manifest,
        root_dir=root_dir,
        transform=get_train_transforms(),
    )
    val_dataset = PlantDataset(
        csv_source=val_manifest,
        root_dir=root_dir,
        transform=get_eval_transforms(),
    )
    test_dataset = PlantDataset(
        csv_source=test_manifest,
        root_dir=root_dir,
        transform=get_eval_transforms(),
    )

    # Seed generator for train loader
    generator = torch.Generator()
    generator.manual_seed(seed)

    if use_weighted_sampler:
        if class_weights is None:
            raise ValueError("class_weights must be provided when use_weighted_sampler=True")
        train_sampler = create_weighted_sampler(train_dataset, class_weights)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            pin_memory=torch.cuda.is_available(),
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
