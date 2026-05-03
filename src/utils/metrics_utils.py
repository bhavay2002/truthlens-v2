"""
File: metrics_utils.py
Location: src/utils/

Centralized metric utilities for TruthLens multi-task system.

Supports:
- Binary, multiclass, multilabel tasks
- Tensor + numpy compatibility
- Safe metric computation
- Aggregation for multi-task evaluation
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List

import numpy as np
import torch

logger = logging.getLogger(__name__)


# =========================================================
# HELPERS
# =========================================================

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0 else 0.0


# =========================================================
# BASIC METRICS
# =========================================================

def accuracy(y_true, y_pred) -> float:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    return float((y_true == y_pred).mean())


def precision_recall_f1(y_true, y_pred) -> Dict[str, float]:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# =========================================================
# MULTICLASS METRICS
# =========================================================

def multiclass_f1(y_true, y_pred, num_classes: int) -> float:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    f1_scores = []

    for cls in range(num_classes):
        yt = (y_true == cls).astype(int)
        yp = (y_pred == cls).astype(int)

        metrics = precision_recall_f1(yt, yp)
        f1_scores.append(metrics["f1"])

    return float(np.mean(f1_scores))


# =========================================================
# MULTILABEL METRICS
# =========================================================

def multilabel_f1(y_true, y_pred) -> Dict[str, float]:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    # micro
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    micro_precision = safe_div(tp, tp + fp)
    micro_recall = safe_div(tp, tp + fn)
    micro_f1 = safe_div(2 * micro_precision * micro_recall,
                        micro_precision + micro_recall)

    # macro
    per_label_f1 = []
    for i in range(y_true.shape[1]):
        metrics = precision_recall_f1(y_true[:, i], y_pred[:, i])
        per_label_f1.append(metrics["f1"])

    macro_f1 = float(np.mean(per_label_f1))

    return {
        "micro_f1": float(micro_f1),
        "macro_f1": macro_f1,
    }


# =========================================================
# PREDICTION CONVERSION
# =========================================================

def logits_to_predictions(
    logits: torch.Tensor,
    task_type: str,
    threshold: float = 0.5,
) -> torch.Tensor:

    if task_type == "multiclass":
        return torch.argmax(logits, dim=-1)

    elif task_type == "binary":
        probs = torch.sigmoid(logits)
        return (probs > threshold).long()

    elif task_type == "multilabel":
        probs = torch.sigmoid(logits)
        return (probs > threshold).long()

    else:
        raise ValueError(f"Unknown task_type: {task_type}")


# =========================================================
# TASK METRIC WRAPPER
# =========================================================

def compute_task_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task_type: str,
    num_labels: int,
    threshold: float = 0.5,
) -> Dict[str, float]:

    preds = logits_to_predictions(logits, task_type, threshold)

    if task_type == "binary":
        metrics = precision_recall_f1(labels, preds)
        metrics["accuracy"] = accuracy(labels, preds)
        return metrics

    elif task_type == "multiclass":
        return {
            "accuracy": accuracy(labels, preds),
            "macro_f1": multiclass_f1(labels, preds, num_labels),
        }

    elif task_type == "multilabel":
        return multilabel_f1(labels, preds)

    else:
        raise ValueError(f"Unsupported task_type: {task_type}")


# =========================================================
# MULTI-TASK AGGREGATION
# =========================================================

def aggregate_metrics(
    task_metrics: Dict[str, Dict[str, float]],
    weights: Dict[str, float] | None = None,
) -> Dict[str, float]:

    if not task_metrics:
        return {}

    total_score = 0.0
    total_weight = 0.0

    for task, metrics in task_metrics.items():
        weight = weights.get(task, 1.0) if weights else 1.0

        # choose representative metric
        score = (
            metrics.get("f1")
            or metrics.get("micro_f1")
            or metrics.get("macro_f1")
            or metrics.get("accuracy")
        )

        if score is None:
            logger.warning("No usable metric for task: %s", task)
            continue

        total_score += score * weight
        total_weight += weight

    overall = safe_div(total_score, total_weight)

    return {
        "overall_score": float(overall)
    }


# =========================================================
# METRIC REDUCTION (DDP SAFE)
# =========================================================

def reduce_metrics_across_processes(
    metrics: Dict[str, float],
) -> Dict[str, float]:
    """
    Placeholder for DDP reduction.
    (Integrate with distributed_utils if needed)
    """
    return metrics


# =========================================================
# GENERIC HELPERS (re-exported from src.utils)
# =========================================================

def safe_mean(values, default: float = 0.0) -> float:
    """Mean of a sequence, returning ``default`` for empty / non-finite input."""
    if values is None:
        return float(default)

    arr = _to_numpy(values).astype(float, copy=False).ravel()
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return float(default)

    return float(arr.mean())


def compute_metrics_from_preds(
    y_true,
    y_pred,
    *,
    task_type: str,
    y_proba=None,
    threshold: float = 0.5,
    average: str | None = None,
) -> Dict[str, Any]:
    """Forward to :mod:`src.evaluation.metrics_engine` for hard-prediction inputs.

    This indirection lets callers in ``src.utils`` stay loosely coupled to the
    evaluation package while still exposing the same calculation.
    """
    from src.evaluation.metrics_engine import compute_metrics_from_preds as _impl

    return _impl(
        y_true=y_true,
        y_pred=y_pred,
        task_type=task_type,
        y_proba=y_proba,
        threshold=threshold,
        average=average,
    )


def normalize_score(value, *, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp ``value`` into the closed interval ``[lo, hi]``."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(lo)

    if not np.isfinite(v):
        return float(lo)

    if v < lo:
        return float(lo)
    if v > hi:
        return float(hi)
    return v