"""
File: metrics_engine.py
Location: src/evaluation/

Single source of truth for evaluation metrics.

Public API:
- ``compute_classification_metrics(y_true, y_pred, y_proba=None, ...)``
- ``compute_multilabel_metrics(y_true, y_pred, y_proba=None, ...)``
- ``compute_metrics_from_preds(y_true, y_pred, task_type, *, y_proba=None, ...)``
- ``MetricsEngine`` — multi-task orchestrator
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# COMMON HELPERS
# =========================================================

def _as_1d_int_array(values, *, name: str) -> np.ndarray:
    arr = np.asarray(values)

    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty")

    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D (got shape {arr.shape})")

    return arr


def _as_2d_int_array(values, *, name: str) -> np.ndarray:
    arr = np.asarray(values)

    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty")

    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D (got shape {arr.shape})")

    return arr


def _check_shape_match(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch between y_true {a.shape} and y_pred {b.shape}"
        )


def _binary_proba_for_auc(y_proba: np.ndarray) -> Optional[np.ndarray]:
    """Return per-sample probability of the positive class for binary tasks."""
    if y_proba.ndim == 1:
        return y_proba
    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        return y_proba[:, 1]
    return None


# =========================================================
# CLASSIFICATION METRICS  (binary + multiclass)
# =========================================================

def compute_classification_metrics(
    y_true: Iterable,
    y_pred: Iterable,
    *,
    y_proba: Optional[Iterable] = None,
    average: Optional[str] = None,
    threshold: float = 0.5,
    confidence: Optional[Iterable] = None,
    labels: Optional[Iterable[int]] = None,
    task_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute standard classification metrics.

    Returns at minimum: ``accuracy, precision, recall, f1, f1_macro, f1_micro,
    f1_weighted, mcc, balanced_accuracy, confusion_matrix``. When ``y_proba`` is
    provided, ``roc_auc`` and ``log_loss`` are added (when computable).
    """
    # ``threshold`` and ``confidence`` are accepted to keep the engine API
    # uniform with the multilabel path; classification preds are already hard.
    del threshold, confidence

    y_true_arr = _as_1d_int_array(y_true, name="y_true")

    # Convert probability matrices / logit matrices → class indices before the
    # 1D check so callers that pass raw softmax outputs don't crash here.
    _y_pred = np.asarray(y_pred)
    if _y_pred.ndim == 2:
        _y_pred = np.argmax(_y_pred, axis=1)
    y_pred_arr = _as_1d_int_array(_y_pred, name="y_pred")
    _check_shape_match(y_true_arr, y_pred_arr)

    unique_classes = np.unique(np.concatenate([y_true_arr, y_pred_arr]))

    # CRIT E5: trust the authoritative ``task_type`` instead of inferring binary
    # from the label values. Pruned multiclass slices that happen to contain only
    # {0, 1} previously misrouted to binary averaging and a single-class roc_auc.
    if task_type == "binary":
        is_binary = True
    elif task_type in ("multiclass", "classification"):
        is_binary = False
    else:
        is_binary = (
            unique_classes.size <= 2
            and set(unique_classes.tolist()).issubset({0, 1})
        )

    chosen_average = average or ("binary" if is_binary else "macro")

    cm_labels = list(labels) if labels is not None else sorted(unique_classes.tolist())

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true_arr, y_pred_arr)
        ),
        "precision": float(
            precision_score(
                y_true_arr,
                y_pred_arr,
                average=chosen_average,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true_arr,
                y_pred_arr,
                average=chosen_average,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true_arr,
                y_pred_arr,
                average=chosen_average,
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "f1_micro": float(
            f1_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        ),
        "metric_average": chosen_average,
        "confusion_matrix": confusion_matrix(
            y_true_arr, y_pred_arr, labels=cm_labels
        ).tolist(),
    }

    try:
        metrics["mcc"] = float(matthews_corrcoef(y_true_arr, y_pred_arr))
    except ValueError:
        metrics["mcc"] = 0.0

    # Per-class f1 (helpful for downstream plots)
    metrics["per_class_f1"] = f1_score(
        y_true_arr,
        y_pred_arr,
        average=None,
        labels=cm_labels,
        zero_division=0,
    ).tolist()

    if y_proba is not None:
        proba_arr = np.asarray(y_proba, dtype=float)
        if proba_arr.shape[0] != y_true_arr.shape[0]:
            raise ValueError(
                f"y_proba length {proba_arr.shape[0]} does not match y_true "
                f"length {y_true_arr.shape[0]}"
            )

        try:
            if is_binary:
                positive_proba = _binary_proba_for_auc(proba_arr)
                if positive_proba is not None:
                    # METRIC CORRECTNESS: surface the single-class skip instead of
                    # silently dropping the metric — masks dataset-level bugs.
                    if len(np.unique(y_true_arr)) > 1:
                        metrics["roc_auc"] = float(
                            roc_auc_score(y_true_arr, positive_proba)
                        )
                    else:
                        logger.warning(
                            "roc_auc skipped: y_true has a single class"
                        )
            else:
                if proba_arr.ndim == 2 and len(np.unique(y_true_arr)) > 1:
                    metrics["roc_auc"] = float(
                        roc_auc_score(
                            y_true_arr,
                            proba_arr,
                            multi_class="ovr",
                            average="macro",
                        )
                    )
        except ValueError as exc:
            logger.debug("roc_auc skipped: %s", exc)

        try:
            if proba_arr.ndim == 2:
                metrics["log_loss"] = float(
                    log_loss(y_true_arr, proba_arr, labels=cm_labels)
                )
            elif is_binary:
                metrics["log_loss"] = float(log_loss(y_true_arr, proba_arr))
        except ValueError as exc:
            logger.debug("log_loss skipped: %s", exc)

    return metrics


# =========================================================
# MULTILABEL METRICS
# =========================================================

def compute_multilabel_metrics(
    y_true: Iterable,
    y_pred: Iterable,
    *,
    y_proba: Optional[Iterable] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute multilabel classification metrics.

    Always returns: ``subset_accuracy, element_accuracy, f1_micro, f1_macro,
    f1_samples, f1_weighted, hamming_loss, per_label_f1``.
    Adds ``log_loss`` and ``roc_auc`` when ``y_proba`` is supplied.
    """
    y_true_arr = _as_2d_int_array(y_true, name="y_true")
    y_pred_arr = _as_2d_int_array(y_pred, name="y_pred")
    _check_shape_match(y_true_arr, y_pred_arr)

    metrics: Dict[str, Any] = {
        "subset_accuracy": float(
            np.all(y_true_arr == y_pred_arr, axis=1).mean()
        ),
        "element_accuracy": float((y_true_arr == y_pred_arr).mean()),
        "hamming_loss": float(hamming_loss(y_true_arr, y_pred_arr)),
        "f1_micro": float(
            f1_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "f1_samples": float(
            f1_score(y_true_arr, y_pred_arr, average="samples", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        ),
        "per_label_f1": f1_score(
            y_true_arr, y_pred_arr, average=None, zero_division=0
        ).tolist(),
        "threshold": float(threshold),
    }

    if y_proba is not None:
        proba_arr = np.asarray(y_proba, dtype=float)
        if proba_arr.shape != y_true_arr.shape:
            raise ValueError(
                f"y_proba shape {proba_arr.shape} does not match y_true "
                f"shape {y_true_arr.shape}"
            )

        try:
            # CRIT E6: use sklearn ``log_loss`` per label and average so the
            # multilabel value is on the same scale as the multiclass log_loss
            # reported elsewhere. The hand-rolled element-wise BCE that lived
            # here normalized over N*L instead of per-sample.
            per_label_ll = [
                log_loss(y_true_arr[:, i], proba_arr[:, i], labels=[0, 1])
                for i in range(y_true_arr.shape[1])
                if len(np.unique(y_true_arr[:, i])) > 1
            ]
            if per_label_ll:
                metrics["log_loss"] = float(np.mean(per_label_ll))
        except ValueError as exc:
            logger.debug("multilabel log_loss skipped: %s", exc)

        try:
            # METRIC CORRECTNESS: drop both all-zero and all-one columns; AUC
            # is undefined when a label has no negatives either.
            col_sum = y_true_arr.sum(axis=0)
            n_rows = y_true_arr.shape[0]
            valid_labels = np.where((col_sum > 0) & (col_sum < n_rows))[0]
            if valid_labels.size:
                metrics["roc_auc_macro"] = float(
                    roc_auc_score(
                        y_true_arr[:, valid_labels],
                        proba_arr[:, valid_labels],
                        average="macro",
                    )
                )
        except ValueError as exc:
            logger.debug("multilabel roc_auc skipped: %s", exc)

    return metrics


# =========================================================
# UNIFIED ENTRY (used by Evaluator / EvaluationEngine)
# =========================================================

def compute_metrics_from_preds(
    y_true: Iterable,
    y_pred: Iterable,
    *,
    task_type: str,
    y_proba: Optional[Iterable] = None,
    threshold: float = 0.5,
    average: Optional[str] = None,
) -> Dict[str, Any]:
    """Route to the correct metric computer based on ``task_type``."""
    if task_type in ("binary", "multiclass", "classification"):
        # CRIT E5: forward task_type so compute_classification_metrics doesn't
        # have to re-infer binary vs. multiclass from the label range.
        return compute_classification_metrics(
            y_true,
            y_pred,
            y_proba=y_proba,
            average=average,
            threshold=threshold,
            task_type=task_type,
        )

    if task_type == "multilabel":
        return compute_multilabel_metrics(
            y_true,
            y_pred,
            y_proba=y_proba,
            threshold=threshold,
        )

    raise ValueError(f"Unknown task_type: {task_type!r}")


# =========================================================
# CONFIG + MULTI-TASK ENGINE
# =========================================================

@dataclass
class MetricsEngineConfig:
    default_threshold: float = 0.5
    enable_confidence_weighting: bool = False
    return_per_task: bool = True
    aggregate: bool = True


class MetricsEngine:
    """Stateless multi-task metric orchestrator."""

    # The set of metrics that we average across tasks. Adding extras is safe
    # but we keep the list explicit so downstream consumers know what to expect.
    # CRIT E3: ``log_loss`` is intentionally excluded — it lives on a different
    # scale (``[0, ∞)``) than the bounded metrics here and gets a separate
    # sample-weighted aggregator below.
    _AGG_KEYS = (
        "accuracy",
        "balanced_accuracy",
        "f1",
        "f1_macro",
        "f1_micro",
        "f1_weighted",
        "f1_samples",
        "subset_accuracy",
        "element_accuracy",
        "hamming_loss",
        "mcc",
        "roc_auc",
        "roc_auc_macro",
    )

    def __init__(self, config: Optional[MetricsEngineConfig] = None):
        self.config = config or MetricsEngineConfig()
        logger.info("MetricsEngine initialized")

    # -----------------------------------------------------
    # SINGLE TASK
    # -----------------------------------------------------

    def compute_task(
        self,
        *,
        y_true,
        y_pred,
        y_proba=None,
        task_type: str,
        threshold: Optional[float] = None,
        confidence=None,
    ) -> Dict[str, Any]:
        del confidence  # accepted for API compatibility
        threshold = threshold if threshold is not None else self.config.default_threshold

        return compute_metrics_from_preds(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            task_type=task_type,
            threshold=threshold,
        )

    # -----------------------------------------------------
    # MULTI TASK
    # -----------------------------------------------------

    def compute_multitask(
        self,
        *,
        predictions: Dict[str, Dict[str, Any]],
        task_types: Dict[str, str],
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        # CRIT E3: track per-task sample counts so the aggregator can weight
        # log_loss instead of averaging arithmetically across uneven tasks.
        sample_counts: Dict[str, int] = {}

        for task, data in predictions.items():
            if task not in task_types:
                logger.warning("Missing task type for %s; skipping", task)
                continue

            if "y_true" not in data or "y_pred" not in data:
                logger.warning("Missing y_true/y_pred for %s; skipping", task)
                continue

            threshold = (
                thresholds.get(task)
                if thresholds and task in thresholds
                else None
            )

            try:
                results[task] = self.compute_task(
                    y_true=data["y_true"],
                    y_pred=data["y_pred"],
                    y_proba=data.get("y_proba"),
                    task_type=task_types[task],
                    threshold=threshold,
                )
                try:
                    sample_counts[task] = int(np.asarray(data["y_true"]).shape[0])
                except Exception:
                    sample_counts[task] = 1
            except ValueError as exc:
                logger.warning("Metrics failed for %s: %s", task, exc)

        if self.config.aggregate and results:
            results["__aggregate__"] = self._aggregate(results, sample_counts)

        return results

    # -----------------------------------------------------
    # AGGREGATION (mean across tasks per bounded metric;
    # sample-weighted for log_loss — CRIT E3)
    # -----------------------------------------------------

    def _aggregate(
        self,
        per_task: Dict[str, Dict[str, Any]],
        sample_counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, float]:
        agg: Dict[str, float] = {}
        sample_counts = sample_counts or {}

        # Section 4: weight every bounded metric by per-task sample count.
        # Plain ``np.mean`` lets a 50-sample edge task swing the aggregate
        # the same as a 50,000-sample core task. Falling back to uniform
        # weights when no counts are available preserves the previous
        # behavior for callers that didn't track sizes.
        for key in self._AGG_KEYS:
            pairs = [
                (float(metrics[key]), float(sample_counts.get(task, 1) or 1))
                for task, metrics in per_task.items()
                if isinstance(metrics, dict)
                and isinstance(metrics.get(key), (int, float))
                and np.isfinite(metrics[key])
            ]
            if not pairs:
                continue
            values, weights = zip(*pairs)
            if any(w > 0 for w in weights):
                agg[key] = float(np.average(values, weights=weights))
            else:
                agg[key] = float(np.mean(values))

        # CRIT E3: log_loss aggregated separately, weighted by per-task sample
        # counts. Surfaced under ``log_loss`` so EvaluationEngine._extract_val_loss
        # picks it up as a single early-stopping signal.
        ll_vals: list[float] = []
        ll_weights: list[float] = []
        for task, metrics in per_task.items():
            if not isinstance(metrics, dict):
                continue
            ll = metrics.get("log_loss")
            if isinstance(ll, (int, float)) and np.isfinite(ll):
                ll_vals.append(float(ll))
                ll_weights.append(float(sample_counts.get(task, 1) or 1))
        if ll_vals:
            agg["log_loss"] = float(np.average(ll_vals, weights=ll_weights))

        # Section 4: ``worst_task_f1`` and ``f1_imbalance_index`` surface
        # cross-task disparity that a mean alone hides. The imbalance index
        # is ``(max - min) / max`` clamped to ``[0, 1]``; 0 means perfectly
        # balanced, 1 means at least one task is at zero F1.
        f1_vals: list[float] = []
        f1_tasks: list[str] = []
        for task, metrics in per_task.items():
            if not isinstance(metrics, dict):
                continue
            f1 = metrics.get("f1")
            if not isinstance(f1, (int, float)) or not np.isfinite(f1):
                f1 = metrics.get("f1_macro")
            if isinstance(f1, (int, float)) and np.isfinite(f1):
                f1_vals.append(float(f1))
                f1_tasks.append(task)
        if f1_vals:
            worst_idx = int(np.argmin(f1_vals))
            best = float(np.max(f1_vals))
            worst = float(f1_vals[worst_idx])
            agg["worst_task_f1"] = worst
            agg["worst_task_f1_name"] = f1_tasks[worst_idx]  # type: ignore[assignment]
            if best > 0:
                agg["f1_imbalance_index"] = float(
                    np.clip((best - worst) / best, 0.0, 1.0)
                )
            else:
                agg["f1_imbalance_index"] = 0.0

        agg["num_tasks"] = float(len(per_task))
        return agg


__all__ = [
    "MetricsEngine",
    "MetricsEngineConfig",
    "compute_classification_metrics",
    "compute_metrics_from_preds",
    "compute_multilabel_metrics",
]
