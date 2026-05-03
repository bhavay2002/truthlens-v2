"""
File: drift_detection.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
from scipy.stats import entropy, wasserstein_distance

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# CONFIG
# =========================================================

@dataclass
class DriftConfig:
    kl_threshold: float = 0.1
    js_threshold: float = 0.1
    psi_threshold: float = 0.2
    wasserstein_threshold: float = 0.1


# =========================================================
# CORE FUNCTIONS
# =========================================================

def _normalize(p):
    p = np.asarray(p)
    p = p + EPS
    return p / np.sum(p)


def kl_divergence(p, q):
    p = _normalize(p)
    q = _normalize(q)
    return float(entropy(p, q))


def js_divergence(p, q):
    p = _normalize(p)
    q = _normalize(q)
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)


def population_stability_index(expected, actual, bins=10):

    expected = np.asarray(expected)
    actual = np.asarray(actual)

    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))

    psi = 0.0

    for i in range(bins):
        e_mask = (expected >= breakpoints[i]) & (expected < breakpoints[i + 1])
        a_mask = (actual >= breakpoints[i]) & (actual < breakpoints[i + 1])

        e_ratio = np.sum(e_mask) / len(expected)
        a_ratio = np.sum(a_mask) / len(actual)

        psi += (a_ratio - e_ratio) * np.log((a_ratio + EPS) / (e_ratio + EPS))

    return float(psi)


# =========================================================
# MAIN CLASS
# =========================================================

class DriftDetector:

    def __init__(self, config: Optional[DriftConfig] = None):
        self.config = config or DriftConfig()
        self.baseline: Dict[str, Any] = {}

        logger.info("DriftDetector initialized")

    # =====================================================
    # SET BASELINE
    # =====================================================

    def set_baseline(
        self,
        *,
        probabilities: Dict[str, np.ndarray],
        entropy_values: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Store baseline distributions.
        """

        self.baseline = {
            "probabilities": probabilities,
            "entropy": entropy_values or {},
        }

        logger.info("Baseline set for drift detection")

    # =====================================================
    # DETECT DRIFT
    # =====================================================

    def detect(
        self,
        *,
        probabilities: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:

        if not self.baseline:
            raise RuntimeError("Baseline not set")

        results = {}

        for task, probs in probabilities.items():

            base_probs = self.baseline["probabilities"].get(task)

            if base_probs is None:
                continue

            # flatten distributions
            p = base_probs.flatten()
            q = probs.flatten()

            # normalize histograms
            hist_p, _ = np.histogram(p, bins=20, range=(0, 1), density=True)
            hist_q, _ = np.histogram(q, bins=20, range=(0, 1), density=True)

            kl = kl_divergence(hist_p, hist_q)
            js = js_divergence(hist_p, hist_q)
            psi = population_stability_index(p, q)
            wdist = wasserstein_distance(p, q)

            drift_flag = (
                kl > self.config.kl_threshold
                or js > self.config.js_threshold
                or psi > self.config.psi_threshold
                or wdist > self.config.wasserstein_threshold
            )

            results[task] = {
                "kl_divergence": kl,
                "js_divergence": js,
                "psi": psi,
                "wasserstein": wdist,
                "drift_detected": drift_flag,
            }

        return results

    # =====================================================
    # ENTROPY DRIFT
    # =====================================================

    def detect_entropy_drift(
        self,
        *,
        entropy_values: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:

        results = {}

        for task, ent in entropy_values.items():

            base = self.baseline.get("entropy", {}).get(task)

            if base is None:
                continue

            psi = population_stability_index(base, ent)
            wdist = wasserstein_distance(base, ent)

            drift_flag = (
                psi > self.config.psi_threshold
                or wdist > self.config.wasserstein_threshold
            )

            results[task] = {
                "psi": psi,
                "wasserstein": wdist,
                "drift_detected": drift_flag,
            }

        return results