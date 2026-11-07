"""Classification metrics for model evaluation.

Phase 4 - Deliverable 2 (Evaluation).
Provides:
    - top_k_accuracy: fraction of samples whose true class is in the k highest scores.
    - macro_auc_ovr: macro one-vs-rest AUC-ROC from softmax probabilities.
    - compute_all_metrics: accuracy, top-3, macro/weighted P/R/F1 and macro AUC.
    - per_class_table: support, precision, recall, F1 and AUC for every class.

All functions work on plain NumPy arrays so they can be tested without a model.
Macro scores are averaged over the classes present in ``y_true``. On val/test every
class is present; on the small field set some classes are missing, and averaging over
absent classes would add meaningless zeros.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _as_int_array(values: Sequence[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.int64).ravel()


def top_k_accuracy(y_true: Sequence[int], probs: np.ndarray, k: int = 3) -> float:
    """Fraction of samples whose true label is among the ``k`` highest-scoring classes."""
    y_true = _as_int_array(y_true)
    probs = np.asarray(probs, dtype=np.float64)
    if len(y_true) == 0:
        return 0.0
    k = min(k, probs.shape[1])
    top_k = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([y in row for y, row in zip(y_true, top_k)]))


def per_class_auc(y_true: Sequence[int], probs: np.ndarray, num_classes: int) -> np.ndarray:
    """One-vs-rest AUC for each class; NaN where it is undefined.

    AUC needs at least one positive and one negative sample, so a class absent from
    ``y_true`` (common on the field set) or a split containing only one class gets NaN.
    """
    y_true = _as_int_array(y_true)
    probs = np.asarray(probs, dtype=np.float64)
    aucs = np.full(num_classes, np.nan)
    for c in range(num_classes):
        positives = y_true == c
        if positives.any() and not positives.all():
            aucs[c] = roc_auc_score(positives, probs[:, c])
    return aucs


def macro_auc_ovr(y_true: Sequence[int], probs: np.ndarray, num_classes: int) -> Optional[float]:
    """Macro one-vs-rest AUC-ROC over the classes where AUC is defined (None if none)."""
    aucs = per_class_auc(y_true, probs, num_classes)
    if np.all(np.isnan(aucs)):
        return None
    return float(np.nanmean(aucs))


def compute_all_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probs: Optional[np.ndarray],
    class_names: Sequence[str],
) -> Dict[str, object]:
    """Compute the summary metrics reported for a split.

    Args:
        y_true: True class indices.
        y_pred: Predicted class indices (argmax of ``probs``).
        probs: Softmax probabilities, shape (N, num_classes). If None, top-3 and AUC are skipped.
        class_names: Ordered class names from ``splits/class_names.json``.

    Returns:
        Dict with sample/class counts, accuracy, top-3 accuracy, macro and weighted
        precision/recall/F1, and macro one-vs-rest AUC.
    """
    y_true = _as_int_array(y_true)
    y_pred = _as_int_array(y_pred)
    num_classes = len(class_names)
    present = np.unique(y_true)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=present, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=present, average="weighted", zero_division=0
    )

    metrics: Dict[str, object] = {
        "n_samples": int(len(y_true)),
        "n_classes_present": int(len(present)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "top3_accuracy": None,
        "macro_auc_ovr": None,
        "n_auc_classes": 0,
    }

    if probs is not None:
        probs = np.asarray(probs, dtype=np.float64)
        aucs = per_class_auc(y_true, probs, num_classes)
        metrics["top3_accuracy"] = top_k_accuracy(y_true, probs, k=3)
        metrics["n_auc_classes"] = int(np.sum(~np.isnan(aucs)))
        metrics["macro_auc_ovr"] = None if np.all(np.isnan(aucs)) else float(np.nanmean(aucs))

    return metrics


def per_class_table(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probs: Optional[np.ndarray],
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Per-class support, predicted count, precision, recall, F1 and one-vs-rest AUC.

    Every class in ``class_names`` gets a row; classes with zero support have
    NaN recall/F1/AUC so they are not mistaken for real zero scores.
    """
    y_true = _as_int_array(y_true)
    y_pred = _as_int_array(y_pred)
    num_classes = len(class_names)
    labels = np.arange(num_classes)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    predicted = np.bincount(y_pred, minlength=num_classes)[:num_classes]
    aucs = per_class_auc(y_true, probs, num_classes) if probs is not None else np.full(num_classes, np.nan)

    table = pd.DataFrame(
        {
            "label_idx": labels,
            "class_name": list(class_names),
            "support": support.astype(int),
            "n_predicted": predicted.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": aucs,
        }
    )
    absent = table["support"] == 0
    table.loc[absent, ["recall", "f1"]] = np.nan
    return table


def compute_confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    """Confusion matrix with rows = true class, columns = predicted class (all classes kept)."""
    return confusion_matrix(_as_int_array(y_true), _as_int_array(y_pred), labels=np.arange(num_classes))


def top_confused_pairs(cm: np.ndarray, class_names: Sequence[str], k: int = 10) -> List[Dict[str, object]]:
    """The ``k`` largest off-diagonal confusion counts as (true, predicted, count) records."""
    off_diag = np.array(cm, dtype=np.int64, copy=True)
    np.fill_diagonal(off_diag, 0)
    row_totals = np.asarray(cm).sum(axis=1)
    order = np.argsort(-off_diag, axis=None)
    pairs: List[Dict[str, object]] = []
    for flat_idx in order[:k]:
        t, p = np.unravel_index(flat_idx, off_diag.shape)
        count = int(off_diag[t, p])
        if count == 0:
            break
        pairs.append(
            {
                "true_class": class_names[t],
                "predicted_class": class_names[p],
                "count": count,
                "share_of_true_class": count / row_totals[t] if row_totals[t] else 0.0,
            }
        )
    return pairs
