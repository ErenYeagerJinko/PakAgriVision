"""Unit tests for error analysis and bias check helpers.

Phase 4 - Deliverable 2.
Verifies on small hand-made prediction tables:
1. Worst-class ranking and confusion partners.
2. Misclassified-image selection (errors only, highest confidence first, class diversity).
3. Crop grouping and the healthy vs diseased false-"healthy" rate.
4. Spearman correlation between class size and F1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.bias_check import crop_of, f1_vs_train_size, healthy_vs_diseased, is_healthy, per_crop_table
from src.analysis.error_analysis import select_misclassified, worst_classes
from src.metrics import compute_confusion_matrix, per_class_table

NAMES = ["Tomato___Early_blight", "Tomato___healthy", "Potato___Late_blight", "Potato___healthy"]


def make_predictions(y_true, y_pred, confidence):
    return pd.DataFrame({
        "path": [f"img_{i}.jpg" for i in range(len(y_true))],
        "true_idx": y_true,
        "true_label": [NAMES[i] for i in y_true],
        "pred_idx": y_pred,
        "pred_label": [NAMES[i] for i in y_pred],
        "confidence": confidence,
        "correct": np.array(y_true) == np.array(y_pred),
        "top3_labels": ["|".join(NAMES[:3])] * len(y_true),
    })


@pytest.fixture
def example():
    y_true = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 3])
    y_pred = np.array([0, 1, 1, 0, 1, 1, 2, 3, 2, 3])
    conf = np.array([0.9, 0.95, 0.6, 0.8, 0.99, 0.7, 0.9, 0.85, 0.9, 0.9])
    return y_true, y_pred, make_predictions(y_true, y_pred, conf)


def test_crop_and_healthy_helpers():
    assert crop_of("Corn___Common_rust") == "Corn"
    assert is_healthy("Apple___healthy") and not is_healthy("Apple___Apple_scab")


def test_worst_classes_and_partner(example):
    y_true, y_pred, _ = example
    per_class = per_class_table(y_true, y_pred, None, NAMES)
    cm = compute_confusion_matrix(y_true, y_pred, len(NAMES))
    worst = worst_classes(per_class, cm, NAMES, k=2)
    assert worst.loc[0, "class_name"] == "Tomato___Early_blight"
    assert worst.loc[0, "main_confused_as"] == "Tomato___healthy"
    assert worst.loc[0, "n_confused_as"] == 2
    assert worst["f1"].is_monotonic_increasing


def test_select_misclassified_orders_errors_by_confidence(example):
    _, _, preds = example
    chosen = select_misclassified(preds, n=12)
    assert len(chosen) == 3  # only 3 errors exist
    assert not chosen["correct"].any()
    assert chosen["confidence"].is_monotonic_decreasing


def test_healthy_vs_diseased_false_healthy_rate(example):
    _, _, preds = example
    table = healthy_vs_diseased(preds).set_index("group")
    # 7 diseased images; 3 are predicted as a healthy class
    # (two tomato early blight -> tomato healthy, one potato late blight -> potato healthy).
    assert table.loc["diseased", "n_images"] == 7
    assert table.loc["diseased", "false_healthy_rate"] == pytest.approx(3 / 7)
    assert table.loc["healthy", "false_disease_rate"] == pytest.approx(0.0)


def test_per_crop_table(example):
    y_true, y_pred, preds = example
    per_class = per_class_table(y_true, y_pred, None, NAMES)
    table = per_crop_table(preds, per_class).set_index("crop")
    assert table.loc["Tomato", "n_images"] == 6
    assert table.loc["Tomato", "accuracy"] == pytest.approx(4 / 6)
    assert table.loc["Potato", "n_classes"] == 2


def test_f1_vs_train_size_spearman():
    per_class = pd.DataFrame({
        "label_idx": [0, 1, 2, 3], "class_name": NAMES, "support": [5, 5, 5, 5],
        "precision": [0.5, 0.6, 0.7, 0.8], "recall": [0.5, 0.6, 0.7, 0.8], "f1": [0.5, 0.6, 0.7, 0.8],
    })
    train = pd.DataFrame({"label_idx": [0] * 10 + [1] * 20 + [2] * 30 + [3] * 40})
    result = f1_vs_train_size(per_class, train)
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["train_count_min"] == 10 and result["train_count_max"] == 40
