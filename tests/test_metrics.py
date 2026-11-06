"""Unit tests for the evaluation metrics module.

Phase 4 - Deliverable 2.
Verifies:
1. Perfect predictions score 1.0 on every metric.
2. Random predictions over 38 classes score close to chance (1/38).
3. Classes missing from y_true (as on the field set) do not crash and are excluded from macro scores.
4. Top-k accuracy, per-class AUC and confused-pair extraction behave as expected.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score, roc_auc_score

from src.metrics import (
    compute_all_metrics,
    compute_confusion_matrix,
    macro_auc_ovr,
    per_class_auc,
    per_class_table,
    top_confused_pairs,
    top_k_accuracy,
)

NUM_CLASSES = 38
CLASS_NAMES = [f"Crop___class_{i}" for i in range(NUM_CLASSES)]


def one_hot_probs(labels: np.ndarray, num_classes: int = NUM_CLASSES, confidence: float = 0.9) -> np.ndarray:
    """Probabilities that put `confidence` on the given label and spread the rest evenly."""
    probs = np.full((len(labels), num_classes), (1.0 - confidence) / (num_classes - 1))
    probs[np.arange(len(labels)), labels] = confidence
    return probs


def test_perfect_predictions_score_one():
    y = np.repeat(np.arange(NUM_CLASSES), 5)
    m = compute_all_metrics(y, y, one_hot_probs(y), CLASS_NAMES)

    for key in ("accuracy", "top3_accuracy", "macro_precision", "macro_recall", "macro_f1",
                "weighted_precision", "weighted_recall", "weighted_f1", "macro_auc_ovr"):
        assert m[key] == pytest.approx(1.0), key
    assert m["n_samples"] == NUM_CLASSES * 5
    assert m["n_classes_present"] == NUM_CLASSES
    assert m["n_auc_classes"] == NUM_CLASSES


def test_random_predictions_near_chance():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, NUM_CLASSES, size=20000)
    probs = rng.dirichlet(np.ones(NUM_CLASSES), size=len(y_true))
    y_pred = probs.argmax(axis=1)
    m = compute_all_metrics(y_true, y_pred, probs, CLASS_NAMES)

    chance = 1.0 / NUM_CLASSES
    assert m["accuracy"] == pytest.approx(chance, abs=0.01)
    assert m["macro_f1"] == pytest.approx(chance, abs=0.01)
    assert m["top3_accuracy"] == pytest.approx(3 * chance, abs=0.015)
    assert m["macro_auc_ovr"] == pytest.approx(0.5, abs=0.02)


def test_missing_classes_do_not_crash_and_are_excluded():
    # Field-set situation: only 3 of 38 classes appear, one prediction goes to an absent class.
    y_true = np.array([0, 0, 1, 1, 5, 5])
    y_pred = np.array([0, 0, 1, 7, 5, 5])
    probs = one_hot_probs(y_pred)
    m = compute_all_metrics(y_true, y_pred, probs, CLASS_NAMES)

    assert m["n_classes_present"] == 3
    assert m["n_auc_classes"] == 3
    expected_f1 = f1_score(y_true, y_pred, labels=[0, 1, 5], average="macro", zero_division=0)
    assert m["macro_f1"] == pytest.approx(expected_f1)
    assert 0.0 < m["macro_f1"] < 1.0

    table = per_class_table(y_true, y_pred, probs, CLASS_NAMES)
    assert len(table) == NUM_CLASSES
    assert np.isnan(table.loc[7, "f1"]) and np.isnan(table.loc[7, "auc"])
    assert table.loc[7, "n_predicted"] == 1
    assert table.loc[1, "recall"] == pytest.approx(0.5)


def test_single_class_split_has_no_auc():
    y = np.zeros(4, dtype=int)
    m = compute_all_metrics(y, y, one_hot_probs(y), CLASS_NAMES)
    assert m["macro_auc_ovr"] is None
    assert m["n_auc_classes"] == 0
    assert macro_auc_ovr(y, one_hot_probs(y), NUM_CLASSES) is None


def test_metrics_without_probabilities():
    y = np.array([0, 1, 2, 2])
    m = compute_all_metrics(y, y, None, CLASS_NAMES)
    assert m["accuracy"] == 1.0
    assert m["top3_accuracy"] is None and m["macro_auc_ovr"] is None


def test_top_k_accuracy():
    probs = np.array([
        [0.5, 0.3, 0.2, 0.0],
        [0.1, 0.2, 0.3, 0.4],
        [0.4, 0.3, 0.2, 0.1],
    ])
    y_true = np.array([0, 1, 3])
    assert top_k_accuracy(y_true, probs, k=1) == pytest.approx(1 / 3)
    assert top_k_accuracy(y_true, probs, k=3) == pytest.approx(2 / 3)
    assert top_k_accuracy(y_true, probs, k=10) == pytest.approx(1.0)


def test_per_class_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, size=200)
    probs = rng.dirichlet(np.ones(4), size=200)
    aucs = per_class_auc(y_true, probs, 4)
    for c in range(4):
        assert aucs[c] == pytest.approx(roc_auc_score(y_true == c, probs[:, c]))


def test_confusion_matrix_and_top_confused_pairs():
    names = ["A", "B", "C"]
    y_true = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2, 2, 0])
    cm = compute_confusion_matrix(y_true, y_pred, 3)
    assert cm.shape == (3, 3)
    assert cm.sum() == len(y_true)

    pairs = top_confused_pairs(cm, names, k=10)
    assert pairs[0] == {"true_class": "A", "predicted_class": "B", "count": 2,
                        "share_of_true_class": pytest.approx(2 / 3)}
    assert len(pairs) == 3  # zero-count pairs are dropped
    assert all(p["true_class"] != p["predicted_class"] for p in pairs)
