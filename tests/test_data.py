"""Deliverable 1 data pipeline acceptance test suite.

Verifies all requirements defined in Phase 2 Plan (A4):
1. 38 class names exist and are non-empty.
2. Dataset splits are strictly disjoint (zero path overlap).
3. Every class is represented in the dataset splits.
4. Batch images have shape (B, 3, 224, 224) and float32 dtype.
5. Batch targets are in range [0, 37] and int64 dtype.
6. Class weights vector has length 38 and all values > 0.
7. Evaluation transform is strictly deterministic.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from PIL import Image

from src.data.dataset import PlantDataset, get_dataloaders
from src.data.splits import (
    compute_class_weights,
    create_stratified_group_splits,
    get_class_mappings,
    load_class_names,
)
from src.data.transforms import get_eval_transforms, get_train_transforms


@pytest.fixture
def test_environment(tmp_path: Path) -> Tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], Dict[str, int]]:
    """Synthesize an entire 38-class mini-dataset with split manifests."""
    class_names, class_to_idx = get_class_mappings()
    root_dir = tmp_path / "dataset"

    records = []
    # Create 2 synthetic images per class across all 38 classes
    for class_name in class_names:
        class_folder = root_dir / class_name
        class_folder.mkdir(parents=True, exist_ok=True)

        for i in range(3):
            img_arr = np.random.randint(40, 200, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(img_arr)
            filename = f"leaf_{i:03d}.jpg"
            img.save(class_folder / filename)

            records.append({
                "path": f"{class_name}/{filename}",
                "label": class_name,
                "group_id": f"grp_{class_name}_{i}",
                "hash_type": "unique",
            })

    full_df = pd.DataFrame(records)
    train_df, val_df, test_df = create_stratified_group_splits(
        df=full_df,
        class_to_idx=class_to_idx,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    return root_dir, train_df, val_df, test_df, class_names, class_to_idx


def test_38_class_names():
    """Assert exactly 38 unique, non-empty class names exist."""
    class_names = load_class_names()
    assert isinstance(class_names, list)
    assert len(class_names) == 38, f"Expected 38 classes, found {len(class_names)}"
    assert len(set(class_names)) == 38, "Class names must be strictly unique"
    for name in class_names:
        assert isinstance(name, str) and name.strip(), "Class names must be non-empty strings"


def test_splits_disjoint(test_environment):
    """Assert splits have zero overlapping paths."""
    _, train_df, val_df, test_df, _, _ = test_environment
    train_paths = set(train_df["path"])
    val_paths = set(val_df["path"])
    test_paths = set(test_df["path"])

    assert train_paths.isdisjoint(val_paths), "Overlap between train and val!"
    assert train_paths.isdisjoint(test_paths), "Overlap between train and test!"
    assert val_paths.isdisjoint(test_paths), "Overlap between val and test!"


def test_every_class_in_dataset(test_environment):
    """Assert all 38 classes are represented across the dataset."""
    _, train_df, _, _, class_names, _ = test_environment
    classes_in_train = set(train_df["label"])
    assert classes_in_train == set(class_names), "Every class must be represented in training split"


def test_batch_tensor_shape_and_dtype(test_environment):
    """Assert one batch has shape (B, 3, 224, 224) and float32 dtype."""
    root_dir, train_df, val_df, test_df, _, _ = test_environment
    train_loader, _, _ = get_dataloaders(
        train_manifest=train_df,
        val_manifest=val_df,
        test_manifest=test_df,
        root_dir=root_dir,
        batch_size=8,
        num_workers=0,
    )

    batch_images, _ = next(iter(train_loader))
    assert batch_images.shape == (8, 3, 224, 224), f"Unexpected batch shape: {batch_images.shape}"
    assert batch_images.dtype == torch.float32, f"Expected float32, got {batch_images.dtype}"


def test_labels_range_and_dtype(test_environment):
    """Assert target labels are in [0, 37] and int64 dtype."""
    root_dir, train_df, val_df, test_df, _, _ = test_environment
    train_loader, _, _ = get_dataloaders(
        train_manifest=train_df,
        val_manifest=val_df,
        test_manifest=test_df,
        root_dir=root_dir,
        batch_size=8,
        num_workers=0,
    )

    _, batch_labels = next(iter(train_loader))
    assert batch_labels.dtype == torch.long or batch_labels.dtype == torch.int64
    assert (batch_labels >= 0).all() and (batch_labels <= 37).all(), "Labels must be within [0, 37]"


def test_class_weights_validity(test_environment):
    """Assert class weights vector has length 38 and all elements > 0."""
    _, train_df, _, _, _, class_to_idx = test_environment
    ordered_weights, weights_dict = compute_class_weights(train_df, class_to_idx)

    assert len(ordered_weights) == 38, f"Expected 38 weights, got {len(ordered_weights)}"
    assert all(w > 0 for w in ordered_weights), "All class weights must be strictly positive"
    assert len(weights_dict) == 38


def test_eval_transform_determinism():
    """Assert eval transform applied twice to identical image yields equal tensors."""
    dummy_img = Image.fromarray(np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8))
    eval_tf = get_eval_transforms(img_size=224)

    t1 = eval_tf(dummy_img)
    t2 = eval_tf(dummy_img)
    assert torch.equal(t1, t2), "Evaluation transform must be bitwise deterministic!"
