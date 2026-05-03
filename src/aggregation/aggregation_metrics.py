from __future__ import annotations

import logging
from typing import Dict, Any, List

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# SAFE UTILS
# =========================================================

def _safe_array(x):
    arr = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)


def _normalize(p):
    p = _safe_array(p)
    p = p + EPS
    return p / (np.sum(p) + EPS)


# =========================================================
# BASIC STATISTICS (UPGRADED)
# =========================================================

def compute_basic_stats(values: np.ndarray) -> Dict[str, float]:

    values = _safe_array(values)

    if values.size == 0:
        return {}

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


# =========================================================
# HISTOGRAM
# =========================================================

def compute_histogram(values: np.ndarray, bins: int = 10) -> Dict[str, Any]:

    values = _safe_array(values)

    hist, bin_edges = np.histogram(values, bins=bins, range=(0.0, 1.0))

    return {
        "counts": hist.tolist(),
        "bin_edges": bin_edges.tolist(),
    }


# =========================================================
# CALIBRATION
# =========================================================

def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:

    probs = _safe_array(probs)
    labels = _safe_array(labels).astype(int)

    if probs.ndim == 2:
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
    else:
        confidences = probs
        predictions = (probs >= 0.5).astype(int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):

        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])

        if not np.any(mask):
            continue

        acc = np.mean(predictions[mask] == labels[mask])
        conf = np.mean(confidences[mask])

        ece += np.abs(acc - conf) * (np.sum(mask) / len(confidences))

    return float(ece)


def classwise_ece(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:

    probs = _safe_array(probs)
    labels = _safe_array(labels).astype(int)

    if probs.ndim != 2:
        return {}

    results = {}

    for c in range(probs.shape[1]):
        binary_labels = (labels == c).astype(int)
        results[f"class_{c}"] = expected_calibration_error(
            probs[:, c], binary_labels
        )

    return results


# =========================================================
# BRIER SCORE
# =========================================================

def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:

    probs = _safe_array(probs)
    labels = _safe_array(labels).astype(int)

    if probs.ndim == 2:
        one_hot = np.eye(probs.shape[1])[labels]
        return float(np.mean((probs - one_hot) ** 2))

    return float(np.mean((probs - labels) ** 2))


# =========================================================
# UNCERTAINTY
# =========================================================

def compute_entropy(probs: np.ndarray) -> np.ndarray:

    probs = _safe_array(probs)

    return -np.sum(probs * np.log(probs + EPS), axis=1)


def uncertainty_statistics(probs: np.ndarray) -> Dict[str, float]:

    entropy = compute_entropy(probs)

    return {
        "mean_entropy": float(np.mean(entropy)),
        "p95_entropy": float(np.percentile(entropy, 95)),
        "p99_entropy": float(np.percentile(entropy, 99)),
    }


# =========================================================
# DRIFT DETECTION
# =========================================================

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = _normalize(p)
    q = _normalize(q)
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = _normalize(p)
    q = _normalize(q)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def population_stability_index(expected, actual, bins=10):

    expected = _safe_array(expected)
    actual = _safe_array(actual)

    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))

    psi = 0.0

    for i in range(bins):

        e_mask = (expected >= breakpoints[i]) & (expected < breakpoints[i + 1])
        a_mask = (actual >= breakpoints[i]) & (actual < breakpoints[i + 1])

        e_ratio = np.sum(e_mask) / len(expected)
        a_ratio = np.sum(a_mask) / len(actual)

        psi += (a_ratio - e_ratio) * np.log((a_ratio + EPS) / (e_ratio + EPS))

    return float(psi)


def compute_distribution_shift(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 20,
) -> Dict[str, float]:

    ref_hist, _ = np.histogram(reference, bins=bins, range=(0, 1), density=True)
    cur_hist, _ = np.histogram(current, bins=bins, range=(0, 1), density=True)

    ref_hist = _normalize(ref_hist)
    cur_hist = _normalize(cur_hist)

    return {
        "kl": kl_divergence(ref_hist, cur_hist),
        "js": js_divergence(ref_hist, cur_hist),
        "psi": population_stability_index(reference, current),
    }


# =========================================================
# TASK METRICS
# =========================================================

def compute_task_metrics(scores: Dict[str, float]) -> Dict[str, Any]:

    values = np.array(list(scores.values()), dtype=np.float32)

    return {
        "stats": compute_basic_stats(values),
    }


def compute_batch_metrics(batch_scores: List[Dict[str, float]]) -> Dict[str, Any]:

    if not batch_scores:
        return {}

    # BATCH-METRIC: collecting keys only from batch_scores[0] silently
    # drops any keys that appear in later samples but not the first.
    # Build the union of all keys across every sample so no metric is
    # lost, then default missing entries to 0.0.
    keys: set = set()
    for sample in batch_scores:
        keys.update(sample.keys())

    aggregated: Dict[str, List[float]] = {k: [] for k in keys}

    for sample in batch_scores:
        for k in keys:
            aggregated[k].append(sample.get(k, 0.0))

    results = {}

    for k, vals in aggregated.items():

        arr = np.array(vals, dtype=np.float32)

        results[k] = {
            "stats": compute_basic_stats(arr),
            "histogram": compute_histogram(arr),
        }

    return results


# =========================================================
# SYSTEM METRICS COLLECTOR
# =========================================================

class AggregationMetrics:

    def __init__(self):
        self.history: List[Dict[str, float]] = []

    def update(self, scores: Dict[str, float]) -> None:
        self.history.append(scores)

    def summarize(self) -> Dict[str, Any]:
        return compute_batch_metrics(self.history)

    def reset(self) -> None:
        self.history.clear()

    def size(self) -> int:
        return len(self.history)