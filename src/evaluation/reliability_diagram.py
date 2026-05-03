from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Optional

import matplotlib

matplotlib.use("Agg", force=False)  # headless friendly
import matplotlib.pyplot as plt
import numpy as np

from src.config.task_config import get_task_type

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# BINNING (vectorized)
# =========================================================

def _bin_stats(confidence: np.ndarray, correctness: np.ndarray, n_bins: int) -> Dict[str, np.ndarray]:
    confidence = np.asarray(confidence, dtype=float).reshape(-1)
    correctness = np.asarray(correctness, dtype=float).reshape(-1)

    if confidence.shape != correctness.shape:
        raise ValueError("confidence and correctness must have the same shape")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(confidence, bins, right=False) - 1, 0, n_bins - 1)

    counts = np.bincount(bin_ids, minlength=n_bins).astype(float)
    sum_acc = np.bincount(bin_ids, weights=correctness, minlength=n_bins)
    sum_conf = np.bincount(bin_ids, weights=confidence, minlength=n_bins)

    safe_counts = np.where(counts > 0, counts, 1.0)
    acc = np.where(counts > 0, sum_acc / safe_counts, 0.0)
    conf = np.where(counts > 0, sum_conf / safe_counts, 0.0)

    return {
        "accuracy": acc,
        "confidence": conf,
        "counts": counts,
        "bin_centers": (bins[:-1] + bins[1:]) / 2,
    }


# =========================================================
# TASK-SPECIFIC RELIABILITY
# =========================================================

def _binary_reliability(y_true, probs, n_bins):
    probs = np.asarray(probs, dtype=float).reshape(-1)
    preds = (probs >= 0.5).astype(int)
    correctness = (preds == np.asarray(y_true)).astype(float)
    return _bin_stats(probs, correctness, n_bins)


def _multiclass_reliability(y_true, probs, n_bins):
    probs = np.asarray(probs, dtype=float)
    preds = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    correctness = (preds == np.asarray(y_true)).astype(float)
    return _bin_stats(confidence, correctness, n_bins)


def _per_class_reliability(y_true, probs, n_bins):
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true)
    n_classes = probs.shape[1]

    return {
        f"class_{c}": _bin_stats(probs[:, c], (y_true == c).astype(float), n_bins)
        for c in range(n_classes)
    }


def _multilabel_reliability(y_true, probs, n_bins):
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true)
    n_labels = probs.shape[1]

    return {
        f"label_{i}": _bin_stats(probs[:, i], y_true[:, i].astype(float), n_bins)
        for i in range(n_labels)
    }


# =========================================================
# PLOTTING
# =========================================================

def _plot_curve(bin_data, title="Reliability Diagram", save_path=None):
    # Section 7: skip the matplotlib figure construction entirely when there
    # is nowhere to save the result. The previous code paid the full
    # ``plt.subplots`` + ``ax.plot`` + ``plt.close`` cost on every reliability
    # call (one per task, sometimes per-class) regardless of whether the
    # figure was wanted, which dominated headless eval runs.
    if not save_path:
        return None

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
    ax.plot(bin_data["confidence"], bin_data["accuracy"], marker="o", label="Model")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend()

    try:
        fig.savefig(save_path, bbox_inches="tight")
    finally:
        plt.close(fig)

    return None  # we close the figure to free memory; consumers shouldn't keep handles


# =========================================================
# MAIN API
# =========================================================

def reliability_diagram(
    y_true,
    probs,
    *,
    task: Optional[str] = None,
    task_type: Optional[str] = None,
    n_bins: int = 10,
    mode: Literal["global", "per_class"] = "global",
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    if task_type is None:
        task_type = get_task_type(task) if task else None

    logger.info("[RELIABILITY] task=%s type=%s", task, task_type)

    if task_type == "binary":
        stats = _binary_reliability(y_true, probs, n_bins)
        _plot_curve(stats, "Binary Reliability", save_path)
        return {"global": stats}

    if task_type == "multiclass":
        global_stats = _multiclass_reliability(y_true, probs, n_bins)
        _plot_curve(global_stats, "Multiclass Reliability", save_path)
        result = {"global": global_stats}
        if mode == "per_class":
            result["per_class"] = _per_class_reliability(y_true, probs, n_bins)
        return result

    if task_type == "multilabel":
        return {"per_label": _multilabel_reliability(y_true, probs, n_bins)}

    raise ValueError(f"Unsupported task_type: {task_type}")


# =========================================================
# CLASS WRAPPER (used by src.evaluation.calibration)
# =========================================================

class ReliabilityDiagram:
    """OO wrapper around :func:`reliability_diagram`."""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def compute(self, probs, y_true, task_type: Optional[str] = None,
                save_path: Optional[str] = None, mode: str = "global"):
        try:
            return reliability_diagram(
                y_true=y_true,
                probs=probs,
                task_type=task_type,
                n_bins=self.n_bins,
                save_path=save_path,
                mode=mode,
            )
        except (TypeError, ValueError) as exc:
            logger.debug("ReliabilityDiagram fallback: %s", exc)
            try:
                stats = _multiclass_reliability(y_true, probs, self.n_bins)
                return {"global": stats}
            except Exception as e:  # pragma: no cover
                logger.warning("ReliabilityDiagram.compute fallback failed: %s", e)
                return {}


__all__ = ["ReliabilityDiagram", "reliability_diagram"]
