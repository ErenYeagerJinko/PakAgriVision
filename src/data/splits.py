"""Dataset partitioning and class weight computation with strict leakage prevention.

This module guarantees:
1. Deterministic 70 / 15 / 15 stratified train / validation / test splits.
2. Group-aware partitioning ensuring all exact and near-duplicate images of a physical leaf
   stay quarantined within the same split (zero test-set leakage).
3. Integer label mapping strictly bound to splits/class_names.json.
4. Balanced class weight calculation derived strictly from the training split only:
   w_c = N_train / (K * n_c)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import SEED, SPLITS_DIR


def load_class_names(class_names_path: Optional[Path | str] = None) -> List[str]:
    """Load ordered list of class names from JSON."""
    p = Path(class_names_path) if class_names_path else (SPLITS_DIR / "class_names.json")
    with open(p, encoding="utf-8") as f:
        class_names = json.load(f)
    if not isinstance(class_names, list) or not class_names:
        raise ValueError(f"Invalid class names file: {p}")
    return class_names


def get_class_mappings(
    class_names_path: Optional[Path | str] = None,
) -> Tuple[List[str], Dict[str, int]]:
    """Return ordered class names and mapping dict {class_name: label_idx}."""
    class_names = load_class_names(class_names_path)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    return class_names, class_to_idx


def compute_class_weights(
    train_df: pd.DataFrame,
    class_to_idx: Dict[str, int],
) -> Tuple[List[float], Dict[str, float]]:
    """Compute balanced class weights strictly from the training split.

    Formula:
        w_c = N_train / (K * n_c)
    where:
        N_train = total number of training samples
        K       = total number of classes (len(class_to_idx))
        n_c     = count of samples for class c in train_df

    Returns:
        (ordered_weights_list, per_class_dict)
    """
    total_samples = len(train_df)
    num_classes = len(class_to_idx)

    if total_samples == 0:
        raise ValueError("Cannot compute class weights from an empty training split.")

    counts = train_df["label"].value_counts().to_dict()
    ordered_weights: List[float] = []
    class_weight_dict: Dict[str, float] = {}

    for class_name, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
        count = counts.get(class_name, 0)
        if count == 0:
            # If a class has 0 samples in train, set weight to 0.0 or 1.0 with warning
            weight = 0.0
        else:
            weight = float(total_samples / (num_classes * count))
        ordered_weights.append(weight)
        class_weight_dict[class_name] = weight

    return ordered_weights, class_weight_dict


def create_stratified_group_splits(
    df: pd.DataFrame,
    class_to_idx: Dict[str, int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition dataset into train/val/test splits with group leakage protection.

    Args:
        df: DataFrame containing at least ['path', 'label'].
            Optional column 'group_id' clusters near-duplicates together.
            Optional column 'hash_type' filters out corrupt images.
        class_to_idx: Dict mapping class names to zero-indexed integer targets.
        train_ratio: Proportion for training set (default: 0.70).
        val_ratio: Proportion for validation set (default: 0.15).
        test_ratio: Proportion for test set (default: 0.15).
        seed: Random seed for deterministic reproducibility.

    Returns:
        (train_df, val_df, test_df) each containing ['path', 'label', 'label_idx'].
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0: got {train_ratio + val_ratio + test_ratio}")

    work_df = df.copy()

    # Filter out corrupt images if hash_type is present
    if "hash_type" in work_df.columns:
        work_df = work_df[work_df["hash_type"] != "corrupt"].copy()

    # If group_id is not provided, treat each path as an individual group
    if "group_id" not in work_df.columns or work_df["group_id"].isnull().all():
        work_df["group_id"] = work_df["path"]

    # Assign label_idx
    work_df["label_idx"] = work_df["label"].map(class_to_idx)
    unmapped = work_df[work_df["label_idx"].isnull()]
    if not unmapped.empty:
        unknown_classes = unmapped["label"].unique().tolist()
        raise ValueError(f"Found unknown classes not present in class_names: {unknown_classes}")
    work_df["label_idx"] = work_df["label_idx"].astype(int)

    rng = np.random.RandomState(seed)

    # Group-level aggregation: each group maps to its primary class and member rows
    # A single group_id must never cross splits
    group_records = []
    for group_id, group_rows in work_df.groupby("group_id"):
        # Most frequent label in the group
        primary_label = group_rows["label"].mode()[0]
        label_idx = class_to_idx[primary_label]
        group_records.append({
            "group_id": group_id,
            "label": primary_label,
            "label_idx": label_idx,
            "num_samples": len(group_rows),
            "rows": group_rows,
        })

    group_df = pd.DataFrame(group_records)

    train_rows: List[pd.DataFrame] = []
    val_rows: List[pd.DataFrame] = []
    test_rows: List[pd.DataFrame] = []

    # Stratify by class across groups
    for class_name, cls_groups in group_df.groupby("label"):
        cls_groups = cls_groups.sample(frac=1.0, random_state=rng)
        total_samples = cls_groups["num_samples"].sum()
        target_train = total_samples * train_ratio
        target_val = total_samples * val_ratio

        curr_train = 0
        curr_val = 0

        for _, g in cls_groups.iterrows():
            g_rows = g["rows"]
            g_size = g["num_samples"]

            if curr_train + g_size <= target_train or (curr_train == 0 and target_train > 0):
                train_rows.append(g_rows)
                curr_train += g_size
            elif curr_val + g_size <= target_val or (curr_val == 0 and target_val > 0):
                val_rows.append(g_rows)
                curr_val += g_size
            else:
                test_rows.append(g_rows)

    train_split = (
        pd.concat(train_rows, ignore_index=True)
        if train_rows
        else pd.DataFrame(columns=work_df.columns)
    )
    val_split = (
        pd.concat(val_rows, ignore_index=True)
        if val_rows
        else pd.DataFrame(columns=work_df.columns)
    )
    test_split = (
        pd.concat(test_rows, ignore_index=True)
        if test_rows
        else pd.DataFrame(columns=work_df.columns)
    )

    # Assertions
    train_paths = set(train_split["path"])
    val_paths = set(val_split["path"])
    test_paths = set(test_split["path"])

    assert train_paths.isdisjoint(val_paths), "Overlap detected between train and val paths!"
    assert train_paths.isdisjoint(test_paths), "Overlap detected between train and test paths!"
    assert val_paths.isdisjoint(test_paths), "Overlap detected between val and test paths!"

    # Assert zero group leakage
    train_groups = set(train_split["group_id"])
    val_groups = set(val_split["group_id"])
    test_groups = set(test_split["group_id"])

    assert train_groups.isdisjoint(val_groups), "Group leakage detected between train and val!"
    assert train_groups.isdisjoint(test_groups), "Group leakage detected between train and test!"
    assert val_groups.isdisjoint(test_groups), "Group leakage detected between val and test!"

    # Standardize output columns
    out_cols = ["path", "label", "label_idx"]
    train_out = train_split[out_cols].sort_values("path").reset_index(drop=True)
    val_out = val_split[out_cols].sort_values("path").reset_index(drop=True)
    test_out = test_split[out_cols].sort_values("path").reset_index(drop=True)

    return train_out, val_out, test_out
