"""
File: explanation_calibrator.py
Module: Explainability Calibration
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-12

# SCALE-3: use a *fixed* reference entropy so confidence is not
# confounded by the number of tokens in a particular explanation.
# A uniform distribution over 512 tokens (BERT / RoBERTa max) has the
# highest expected entropy; anchoring to that value means a peaked
# 5-token explanation and a peaked 100-token explanation with the same
# relative concentration receive the same confidence score.
_MAX_ENTROPY_REF = float(np.log(512.0))


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_scores(scores: List[float]) -> np.ndarray:
    """
    L1 normalization (probability distribution).
    """

    arr = np.asarray(scores, dtype=float)

    if arr.size == 0:
        return arr

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    arr = np.abs(arr)
    total = float(np.sum(arr))

    if total <= 0:
        return np.zeros_like(arr)

    return arr / (total + EPS)


# =========================================================
# ENTROPY
# =========================================================

def compute_entropy(probs: np.ndarray) -> float:
    if probs.size == 0:
        return 0.0

    probs = np.clip(probs, EPS, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def compute_confidence(probs: np.ndarray) -> float:
    """Confidence derived from entropy.

    SCALE-3: normalise against ``_MAX_ENTROPY_REF`` (log 512) rather
    than ``log(N)`` so that explanations with different token counts are
    compared on the same scale. A 5-token peaked explanation and a
    100-token peaked explanation with the same relative concentration
    will now produce the same confidence instead of artificially
    differing because their N values differ.
    """

    if probs.size == 0:
        return 0.0

    entropy = compute_entropy(probs)
    normalized_entropy = entropy / (_MAX_ENTROPY_REF + EPS)
    return float(np.clip(1.0 - normalized_entropy, 0.0, 1.0))


# =========================================================
# METHOD-AWARE NORMALIZATION
# =========================================================

def calibrate_by_method(
    scores: List[float],
    method: Optional[str],
) -> np.ndarray:
    """FAITH-3: drop arbitrary per-method ``power 0.8`` / ``power 1.2``
    shaping. Those exponents had no theoretical basis, were irreversible,
    and changed the relative ordering of low-importance tokens. All methods
    now share a single L1 normalisation step. Method-specific calibration,
    if ever needed, should be a learned monotone transform fitted on a
    labelled faithfulness dataset, not a hardcoded power.
    """

    # ``method`` is retained in the signature for backward compatibility
    # but no longer drives any shape transformation.
    _ = method
    return normalize_scores(scores)


# =========================================================
# MAIN CALIBRATION PIPELINE
# =========================================================

def calibrate_explanation(
    scores: List[float],
    method: Optional[str] = None,
) -> Dict[str, object]:
    """
    FINAL CONTRACT:

    Returns:
    {
        "scores": np.ndarray,
        "confidence": float,
        "entropy": float,
    }
    """

    if not scores:
        return {
            "scores": np.array([], dtype=float),
            "confidence": 0.0,
            "entropy": 0.0,
        }

    # -------------------------
    # normalize + method calibration
    # -------------------------
    calibrated = calibrate_by_method(scores, method)

    # -------------------------
    # entropy + confidence
    # -------------------------
    ent = compute_entropy(calibrated)
    conf = compute_confidence(calibrated)

    return {
        "scores": calibrated,          #  numpy array
        "confidence": conf,            #  float
        "entropy": ent,                # float
    }