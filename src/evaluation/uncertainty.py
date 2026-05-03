from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional

import numpy as np

from src.config.task_config import get_task_type

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# VALIDATION
# =========================================================

def _validate_probs(probs: Iterable, *, allow_1d: bool = False) -> np.ndarray:
    """Validate a probability array. Always returns a 2D ``(N, K)`` numpy array.

    ``allow_1d`` lets callers (e.g. binary tasks with positive-class probability)
    pass a 1D array, which is reshaped to ``(N, 1)``. The default is strict 2D.
    """
    arr = np.asarray(probs, dtype=float)

    if arr.size == 0:
        raise ValueError("probs cannot be empty")

    if arr.ndim == 1:
        if not allow_1d:
            raise ValueError("probs must be 2D (got shape (N,))")
        return arr.reshape(-1, 1)

    if arr.ndim != 2:
        raise ValueError(f"probs must be 2D (got shape {arr.shape})")

    return arr


# =========================================================
# ENTROPY
# =========================================================

def predictive_entropy(probs: Iterable) -> np.ndarray:
    """Shannon entropy along the class axis. Strictly requires 2D input."""
    arr = _validate_probs(probs, allow_1d=False)
    arr = np.clip(arr, EPS, 1.0)
    return -np.sum(arr * np.log(arr), axis=-1)


def normalized_entropy(probs: Iterable) -> np.ndarray:
    arr = _validate_probs(probs)
    n_classes = arr.shape[-1]
    return predictive_entropy(arr) / (np.log(max(n_classes, 2)) + EPS)


# =========================================================
# CONFIDENCE
# =========================================================

def confidence_scores(probs: Iterable) -> np.ndarray:
    """Top-class confidence."""
    arr = _validate_probs(probs, allow_1d=True)
    return np.max(arr, axis=-1)


def margin_confidence(probs: Iterable) -> np.ndarray:
    arr = _validate_probs(probs)
    sorted_probs = np.sort(arr, axis=1)
    if sorted_probs.shape[1] < 2:
        return np.zeros(sorted_probs.shape[0], dtype=float)
    return sorted_probs[:, -1] - sorted_probs[:, -2]


def confidence_weighted_entropy(probs: Iterable) -> np.ndarray:
    arr = _validate_probs(probs)
    return predictive_entropy(arr) * (1.0 - confidence_scores(arr))


# =========================================================
# MULTILABEL
# =========================================================

def multilabel_uncertainty(probs: Iterable) -> Dict[str, np.ndarray]:
    arr = _validate_probs(probs)
    arr_clipped = np.clip(arr, EPS, 1.0 - EPS)

    label_entropy = -(
        arr_clipped * np.log(arr_clipped)
        + (1.0 - arr_clipped) * np.log(1.0 - arr_clipped)
    )

    # Per-sample confidence under multilabel = max(p, 1-p) per label, then mean.
    # Section 8: clip ``per_label_conf`` to the same ``[EPS, 1-EPS]`` window
    # as ``label_entropy`` above so log/conf statistics computed downstream
    # don't see 0.0 / 1.0 sneaking in for one but not the other.
    per_label_conf = np.clip(np.maximum(arr, 1.0 - arr), EPS, 1.0 - EPS)

    return {
        "label_entropy": label_entropy,
        "mean_entropy": np.mean(label_entropy, axis=1),
        "confidence": np.mean(per_label_conf, axis=1),
    }


# =========================================================
# VARIANCE / MUTUAL INFORMATION (MC SAMPLES)
# =========================================================

def predictive_variance(prob_samples: Iterable) -> np.ndarray:
    arr = np.asarray(prob_samples)
    if arr.ndim != 3:
        raise ValueError("Expected (T, N, C)")
    return np.var(arr, axis=0).mean(axis=1)


def mutual_information(prob_samples: Iterable) -> np.ndarray:
    arr = np.asarray(prob_samples)
    if arr.ndim != 3:
        raise ValueError("Expected (T, N, C)")

    mean_probs = np.mean(arr, axis=0)

    entropy_mean = predictive_entropy(mean_probs)
    entropy_expected = np.mean(
        np.stack([predictive_entropy(p) for p in arr]),
        axis=0,
    )

    mi = entropy_mean - entropy_expected
    # HIGH E11: normalize by ``log(K)`` so MI stays comparable across batches
    # and deployments. Dividing by ``max(|mi|)`` (the previous behavior) made
    # any cross-batch MI threshold meaningless because each batch was rescaled
    # to the same [-1, 1] range regardless of the model's actual uncertainty.
    n_classes = int(arr.shape[-1])
    denom = float(np.log(max(n_classes, 2)))
    return mi / denom


# =========================================================
# ENERGY
# =========================================================

def energy_score(logits: Iterable) -> np.ndarray:
    arr = np.asarray(logits, dtype=float)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    max_logits = np.max(arr, axis=1, keepdims=True)
    stabilized = arr - max_logits
    logsumexp = np.log(np.sum(np.exp(stabilized), axis=1) + EPS) + max_logits.squeeze()
    energy = -logsumexp

    # Section 8: explicit ``> EPS`` guard. The previous ``or 1.0`` fired only
    # for the exact value ``0.0`` and let tiny floating-point ``std`` values
    # produce huge z-scores. Treat anything below EPS as effectively zero.
    raw_std = float(np.std(energy))
    std = raw_std if raw_std > EPS else 1.0
    return (energy - float(np.mean(energy))) / std


# =========================================================
# DRIFT SIGNAL
# =========================================================

def uncertainty_drift(entropy: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(entropy, dtype=float)
    return {
        "entropy_shift": float(np.mean(arr)),
        "entropy_spread": float(np.std(arr)),
        "high_uncertainty_ratio": float(np.mean(arr > 0.8)),
    }


# =========================================================
# MAIN STATS
# =========================================================

def uncertainty_statistics(
    probs: Iterable,
    *,
    task: Optional[str] = None,
    logits: Optional[Iterable] = None,
    prob_samples: Optional[Iterable] = None,
    explanation_scores: Optional[Iterable] = None,
) -> Dict[str, float]:
    arr = _validate_probs(probs, allow_1d=False)
    task_type = get_task_type(task) if task else None

    if task_type == "multilabel":
        ml = multilabel_uncertainty(arr)
        entropy = ml["mean_entropy"]
        confidence = ml["confidence"]
    else:
        entropy = predictive_entropy(arr)
        confidence = confidence_scores(arr)

    weighted_entropy = predictive_entropy(arr) * (1.0 - confidence)

    stats = {
        "mean_entropy": float(np.mean(entropy)),
        "std_entropy": float(np.std(entropy)),
        "min_entropy": float(np.min(entropy)),
        "max_entropy": float(np.max(entropy)),
        "p95_entropy": float(np.percentile(entropy, 95)),
        "p99_entropy": float(np.percentile(entropy, 99)),
        "mean_confidence": float(np.mean(confidence)),
        "std_confidence": float(np.std(confidence)),
        "min_confidence": float(np.min(confidence)),
        "max_confidence": float(np.max(confidence)),
        "mean_weighted_entropy": float(np.mean(weighted_entropy)),
    }

    if arr.shape[1] > 1 and task_type != "multilabel":
        stats["mean_margin"] = float(np.mean(margin_confidence(arr)))

    if logits is not None:
        try:
            energy = energy_score(logits)
            stats["mean_energy"] = float(np.mean(energy))
            stats["std_energy"] = float(np.std(energy))
        except (TypeError, ValueError) as exc:
            logger.debug("energy_score skipped: %s", exc)

    if prob_samples is not None:
        try:
            mi = mutual_information(prob_samples)
            stats["mean_mutual_information"] = float(np.mean(mi))
        except ValueError as exc:
            logger.debug("mutual_information skipped: %s", exc)

    stats.update(uncertainty_drift(entropy))

    if explanation_scores is not None:
        explanation_arr = np.asarray(explanation_scores, dtype=float)
        if explanation_arr.shape == entropy.shape and explanation_arr.size > 1:
            with np.errstate(invalid="ignore"):
                corr = np.corrcoef(entropy, explanation_arr)[0, 1]
            stats["uncertainty_explanation_corr"] = (
                float(corr) if np.isfinite(corr) else 0.0
            )

    return stats


# =========================================================
# PER-SAMPLE
# =========================================================

def uncertainty_per_sample(
    probs: Iterable,
    *,
    task: Optional[str] = None,
    logits: Optional[Iterable] = None,
    prob_samples: Optional[Iterable] = None,
) -> Dict[str, np.ndarray]:
    arr = _validate_probs(probs, allow_1d=False)
    task_type = get_task_type(task) if task else None

    if task_type == "multilabel":
        ml = multilabel_uncertainty(arr)
        entropy = ml["mean_entropy"]
        confidence = ml["confidence"]
    else:
        entropy = predictive_entropy(arr)
        confidence = confidence_scores(arr)

    result: Dict[str, np.ndarray] = {
        "entropy": entropy,
        "confidence": confidence,
        "weighted_entropy": predictive_entropy(arr) * (1.0 - confidence),
    }

    if arr.shape[1] > 1 and task_type != "multilabel":
        result["margin"] = margin_confidence(arr)

    if logits is not None:
        try:
            result["energy"] = energy_score(logits)
        except (TypeError, ValueError) as exc:
            logger.debug("energy_score skipped: %s", exc)

    if prob_samples is not None:
        try:
            result["mutual_information"] = mutual_information(prob_samples)
        except ValueError as exc:
            logger.debug("mutual_information skipped: %s", exc)

    return result


__all__ = [
    "confidence_scores",
    "confidence_weighted_entropy",
    "energy_score",
    "margin_confidence",
    "multilabel_uncertainty",
    "mutual_information",
    "normalized_entropy",
    "predictive_entropy",
    "predictive_variance",
    "uncertainty_drift",
    "uncertainty_per_sample",
    "uncertainty_statistics",
]
