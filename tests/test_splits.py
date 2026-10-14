"""Unit tests for dataset partitioning, leakage prevention, and class weight calculation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.splits import (
    compute_class_weights,
    create_stratified_group_splits,
    get_class_mappings,
    load_class_names,
)


@pytest.fixture
def mock_dataset_df() -> pd.DataFrame:
    """Generate a synthetic dataset dataframe covering all 38 classes with some duplicate groups."""
    class_names = load_class_names()
    records = []

    # Create varied samples per class (imbalanced: 10 to 40 samples per class)
    for idx, class_name in enumerate(class_names):
        count = 10 + (idx % 20)
        for i in range(count):
            path = f"{class_name}/img_{i:04d}.jpg"
            # Simulate every 3rd image sharing a group_id with the previous image
            group_id = f"group_{class_name}_{i // 2}"
            records.append({
                "path": path,
                "label": class_name,
                "group_id": group_id,
                "hash_type": "near_duplicate" if (i % 2 == 1) else "unique",
            })

    return pd.DataFrame(records)


def test_load_class_names():
    class_names = load_class_names()
    assert isinstance(class_names, list)
    assert len(class_names) == 38
    assert len(set(class_names)) == 38


def test_splits_disjoint_and_leak_free(mock_dataset_df: pd.DataFrame):
    _, class_to_idx = get_class_mappings()
    train_df, val_df, test_df = create_stratified_group_splits(
        df=mock_dataset_df,
        class_to_idx=class_to_idx,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    # 1. Row counts add up
    total_split = len(train_df) + len(val_df) + len(test_df)
    assert total_split == len(mock_dataset_df)

    # 2. Path disjointness
    train_paths = set(train_df["path"])
    val_paths = set(val_df["path"])
    test_paths = set(test_df["path"])
    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)

    # 3. Zero group leakage
    train_groups = set(train_df["path"].map(mock_dataset_df.set_index("path")["group_id"]))
    val_groups = set(val_df["path"].map(mock_dataset_df.set_index("path")["group_id"]))
    test_groups = set(test_df["path"].map(mock_dataset_df.set_index("path")["group_id"]))
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    # 4. Label coverage in train
    assert set(train_df["label"]) == set(class_to_idx.keys())


def test_class_weights(mock_dataset_df: pd.DataFrame):
    _, class_to_idx = get_class_mappings()
    train_df, _, _ = create_stratified_group_splits(
        df=mock_dataset_df,
        class_to_idx=class_to_idx,
        seed=42,
    )

    ordered_weights, weights_dict = compute_class_weights(train_df, class_to_idx)
    assert len(ordered_weights) == 38
    assert len(weights_dict) == 38
    assert all(w > 0 for w in ordered_weights)

    # Verify inverse frequency behavior: minority class must have higher weight than majority class
    counts = train_df["label"].value_counts()
    min_class = counts.idxmin()
    max_class = counts.idxmax()
    assert weights_dict[min_class] > weights_dict[max_class]


def test_split_reproducibility(mock_dataset_df: pd.DataFrame):
    _, class_to_idx = get_class_mappings()
    train_1, val_1, test_1 = create_stratified_group_splits(
        df=mock_dataset_df,
        class_to_idx=class_to_idx,
        seed=42,
    )
    train_2, val_2, test_2 = create_stratified_group_splits(
        df=mock_dataset_df,
        class_to_idx=class_to_idx,
        seed=42,
    )

    pd.testing.assert_frame_equal(train_1, train_2)
    pd.testing.assert_frame_equal(val_1, val_2)
    pd.testing.assert_frame_equal(test_1, test_2)
