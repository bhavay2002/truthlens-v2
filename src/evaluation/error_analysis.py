from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config.task_config import get_task_type

logger = logging.getLogger(__name__)


# =========================================================
# CORE HELPERS
# =========================================================

def _to_numpy(x):
    return np.asarray(x)


def _top_k_indices(arr: np.ndarray, k: int = 10, *, largest: bool = True) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.size == 0:
        return np.array([], dtype=int)
    k = min(k, arr.size)
    if largest:
        return np.argsort(-arr)[:k]
    return np.argsort(arr)[:k]


def _binary_positive_proba(probs: np.ndarray) -> np.ndarray:
    """Coerce a (N,), (N, 1), or (N, 2) probability array to per-sample P(class=1)."""
    arr = np.asarray(probs, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr.reshape(-1)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr[:, 1]
    raise ValueError(f"Unexpected probability shape for binary task: {arr.shape}")


# =========================================================
# BINARY ERROR ANALYSIS
# =========================================================

def analyze_binary_errors(
    y_true,
    y_pred,
    probs: Optional[np.ndarray] = None,
    texts: Optional[List[str]] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]

    results: Dict[str, Any] = {
        "false_positives": int(len(fp_idx)),
        "false_negatives": int(len(fn_idx)),
    }

    if probs is not None:
        positive_proba = _binary_positive_proba(probs)

        if fp_idx.size:
            fp_conf = positive_proba[fp_idx]
            fp_hard = fp_idx[_top_k_indices(fp_conf, k=top_k, largest=True)]
            results["top_false_positives"] = _build_samples(fp_hard, texts, positive_proba)

        if fn_idx.size:
            fn_conf = positive_proba[fn_idx]
            # HIGH E12: ``largest=False`` on raw confidences picks the lowest
            # P(class=1) directly. The previous ``largest=True`` on
            # ``1.0 - fn_conf`` is mathematically equivalent but the float32
            # subtraction occasionally re-orders ties under EPS rounding.
            fn_hard = fn_idx[_top_k_indices(fn_conf, k=top_k, largest=False)]
            results["top_false_negatives"] = _build_samples(fn_hard, texts, positive_proba)

    return results


# =========================================================
# MULTICLASS ERROR ANALYSIS
# =========================================================

def analyze_multiclass_errors(
    y_true,
    y_pred,
    probs: Optional[np.ndarray] = None,
    texts: Optional[List[str]] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    incorrect = np.where(y_true != y_pred)[0]
    results: Dict[str, Any] = {"total_errors": int(len(incorrect))}

    if incorrect.size:
        pairs = list(zip(y_true[incorrect].tolist(), y_pred[incorrect].tolist()))
        pair_counts = pd.Series(pairs).value_counts().to_dict()
        results["confusion_pairs"] = {
            f"{k[0]}->{k[1]}": int(v) for k, v in pair_counts.items()
        }

    # Per-class error rate
    classes = np.unique(np.concatenate([y_true, y_pred]))
    error_rate_per_class: Dict[str, float] = {}
    for cls in classes:
        cls_mask = y_true == cls
        if cls_mask.any():
            error_rate_per_class[str(int(cls))] = float(
                (y_pred[cls_mask] != cls).mean()
            )
    if error_rate_per_class:
        results["error_rate_per_class"] = error_rate_per_class

    if probs is not None and incorrect.size:
        probs_arr = np.asarray(probs, dtype=float)
        if probs_arr.ndim == 2:
            confidence = np.max(probs_arr, axis=1)
            wrong_conf = confidence[incorrect]
            hard_idx = incorrect[_top_k_indices(wrong_conf, k=top_k, largest=True)]
            results["hard_examples"] = _build_samples(hard_idx, texts, confidence)

    return results


# =========================================================
# MULTILABEL ERROR ANALYSIS
# =========================================================

def analyze_multilabel_errors(
    y_true,
    y_pred,
    probs: Optional[np.ndarray] = None,
    texts: Optional[List[str]] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    errors = (y_true != y_pred).astype(int)
    per_label_errors = errors.sum(axis=0)

    results: Dict[str, Any] = {
        "per_label_error_count": per_label_errors.tolist(),
        "total_error_labels": int(errors.sum()),
    }

    sample_errors = errors.sum(axis=1)
    if sample_errors.size:
        hard_idx = _top_k_indices(sample_errors, k=top_k, largest=True)
        results["hard_samples"] = _build_samples(hard_idx, texts, sample_errors)

    return results


# =========================================================
# SAMPLE BUILDER
# =========================================================

def _build_samples(indices, texts, scores) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for idx in indices:
        sample: Dict[str, Any] = {
            "index": int(idx),
            "score": float(scores[idx]) if scores is not None else None,
        }
        if texts is not None and 0 <= int(idx) < len(texts):
            sample["text"] = texts[int(idx)]
        samples.append(sample)
    return samples


# =========================================================
# MAIN API
# =========================================================

def error_analysis(
    y_true,
    y_pred,
    *,
    probs: Optional[np.ndarray] = None,
    texts: Optional[List[str]] = None,
    task: Optional[str] = None,
    task_type: Optional[str] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    if task_type is None:
        task_type = get_task_type(task) if task else None

    logger.info("[ERROR ANALYSIS] task=%s type=%s", task, task_type)

    if task_type == "binary":
        return analyze_binary_errors(y_true, y_pred, probs, texts, top_k)
    if task_type == "multiclass":
        return analyze_multiclass_errors(y_true, y_pred, probs, texts, top_k)
    if task_type == "multilabel":
        return analyze_multilabel_errors(y_true, y_pred, probs, texts, top_k)

    # Fall back: infer from shapes
    y_true_arr = _to_numpy(y_true)
    if y_true_arr.ndim == 2:
        return analyze_multilabel_errors(y_true, y_pred, probs, texts, top_k)
    if len(np.unique(y_true_arr)) <= 2:
        return analyze_binary_errors(y_true, y_pred, probs, texts, top_k)
    return analyze_multiclass_errors(y_true, y_pred, probs, texts, top_k)


# =========================================================
# CLASS WRAPPER
# =========================================================

class ErrorAnalyzer:
    """Stateless OO wrapper used by :class:`Evaluator`."""

    def __init__(self, *, top_k: int = 10):
        self.top_k = top_k

    def analyze(self, collected: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(collected, dict):
            return {}

        y_true = collected.get("y_true")
        y_pred = collected.get("y_pred")
        if y_true is None or y_pred is None:
            return {}

        try:
            return error_analysis(
                y_true=y_true,
                y_pred=y_pred,
                probs=collected.get("y_proba"),
                task=collected.get("task"),
                task_type=collected.get("task_type"),
                top_k=self.top_k,
            )
        except (TypeError, ValueError) as exc:
            logger.warning("ErrorAnalyzer.analyze failed: %s", exc)
            return {}


__all__ = [
    "ErrorAnalyzer",
    "analyze_binary_errors",
    "analyze_multiclass_errors",
    "analyze_multilabel_errors",
    "error_analysis",
]
