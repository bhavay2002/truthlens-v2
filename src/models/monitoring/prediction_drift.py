from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# UTILS
# =========================================================

def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def _normalize_rows(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    s = np.sum(p, axis=1, keepdims=True)
    s = np.clip(s, EPS, None)
    return p / s


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, EPS, None))


# =========================================================
# DISTANCES / DIVERGENCES
# =========================================================

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = _normalize_rows(p)
    q = _normalize_rows(q)
    return float(np.mean(np.sum(p * (_safe_log(p) - _safe_log(q)), axis=1)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = _normalize_rows(p)
    q = _normalize_rows(q)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    p = _normalize_rows(p)
    q = _normalize_rows(q)
    return float(0.5 * np.mean(np.sum(np.abs(p - q), axis=1)))


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = _normalize_rows(p)
    q = _normalize_rows(q)
    return float(
        np.mean(
            np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2, axis=1))
        )
    )


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    n = min(len(a_sorted), len(b_sorted))
    if n == 0:
        return 0.0
    a_sorted = a_sorted[:n]
    b_sorted = b_sorted[:n]
    return float(np.mean(np.abs(a_sorted - b_sorted)))


# =========================================================
# HISTOGRAM PSI
# =========================================================

def _hist_probs(x: np.ndarray, bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(x, bins=bins, density=False)
    probs = hist.astype(float) / (np.sum(hist) + EPS)
    probs = np.clip(probs, EPS, None)
    probs = probs / np.sum(probs)
    return probs, edges


def psi_1d(ref: np.ndarray, cur: np.ndarray, bins: int = 20) -> float:
    p, edges = _hist_probs(ref, bins=bins)
    # align bins for current
    cur_hist, _ = np.histogram(cur, bins=edges, density=False)
    q = cur_hist.astype(float) / (np.sum(cur_hist) + EPS)
    q = np.clip(q, EPS, None)
    q = q / np.sum(q)
    return float(np.sum((p - q) * (_safe_log(p) - _safe_log(q))))


def psi_matrix(ref: np.ndarray, cur: np.ndarray, bins: int = 20) -> float:
    ref = _ensure_2d(ref)
    cur = _ensure_2d(cur)
    k = min(ref.shape[1], cur.shape[1])
    vals = []
    for i in range(k):
        vals.append(psi_1d(ref[:, i], cur[:, i], bins=bins))
    return float(np.mean(vals)) if vals else 0.0


# =========================================================
# ENTROPY / CONFIDENCE
# =========================================================

def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    probs = _normalize_rows(_ensure_2d(probs))
    return -np.sum(probs * _safe_log(probs), axis=1)


def confidence_scores(probs: np.ndarray) -> np.ndarray:
    probs = _normalize_rows(_ensure_2d(probs))
    return np.max(probs, axis=1)


# =========================================================
# STATE
# =========================================================

@dataclass
class DriftState:
    ref_probs: Optional[np.ndarray] = None
    ref_confidence: Optional[np.ndarray] = None
    ref_entropy: Optional[np.ndarray] = None
    ref_hist_edges: Optional[np.ndarray] = None
    count: int = 0


# =========================================================
# DETECTOR
# =========================================================

class PredictionDriftDetector:
    """
    Drift detector for model predictions (probabilities).

    Tracks:
        - KL / JS divergence
        - Total variation / Hellinger
        - PSI (per-dimension histogram)
        - Wasserstein on confidence
        - Entropy / confidence shifts
    """

    def __init__(
        self,
        threshold: float = 0.1,
        psi_threshold: float = 0.2,
        bins: int = 20,
    ) -> None:
        self.threshold = float(threshold)
        self.psi_threshold = float(psi_threshold)
        self.bins = int(bins)
        self.state = DriftState()

    # =====================================================
    # FIT / RESET
    # =====================================================

    def fit(self, probs: np.ndarray) -> None:
        probs = _normalize_rows(_ensure_2d(probs))

        self.state.ref_probs = probs.copy()
        self.state.ref_confidence = confidence_scores(probs)
        self.state.ref_entropy = predictive_entropy(probs)

        # store global edges based on flattened distribution for consistency
        flat = probs.ravel()
        _, edges = np.histogram(flat, bins=self.bins, density=False)
        self.state.ref_hist_edges = edges

        self.state.count = probs.shape[0]

        logger.info("[DRIFT] baseline fitted (n=%d)", self.state.count)

    def reset(self) -> None:
        self.state = DriftState()

    # =====================================================
    # UPDATE / COMPUTE
    # =====================================================

    def update(self, probs: np.ndarray) -> Dict[str, float]:
        if self.state.ref_probs is None:
            raise RuntimeError("Call fit() before update().")

        cur = _normalize_rows(_ensure_2d(probs))
        ref = self.state.ref_probs

        # align shapes by columns
        k = min(ref.shape[1], cur.shape[1])
        ref_k = ref[:, :k]
        cur_k = cur[:, :k]

        # core divergences
        kl = kl_divergence(ref_k, cur_k)
        js = js_divergence(ref_k, cur_k)
        tv = total_variation(ref_k, cur_k)
        hel = hellinger_distance(ref_k, cur_k)

        # PSI (per-dimension)
        psi_val = psi_matrix(ref_k, cur_k, bins=self.bins)

        # confidence / entropy
        ref_conf = self.state.ref_confidence
        cur_conf = confidence_scores(cur_k)

        ref_ent = self.state.ref_entropy
        cur_ent = predictive_entropy(cur_k)

        conf_shift = float(np.mean(cur_conf) - np.mean(ref_conf))
        ent_shift = float(np.mean(cur_ent) - np.mean(ref_ent))

        conf_wass = wasserstein_1d(ref_conf, cur_conf)
        ent_wass = wasserstein_1d(ref_ent, cur_ent)

        # aggregate score
        score = float(np.mean([js, tv, hel, psi_val]))

        drift_flag = bool(
            (score > self.threshold)
            or (psi_val > self.psi_threshold)
        )

        result = {
            "kl_divergence": float(kl),
            "js_divergence": float(js),
            "total_variation": float(tv),
            "hellinger": float(hel),
            "psi": float(psi_val),
            "confidence_shift": float(conf_shift),
            "entropy_shift": float(ent_shift),
            "confidence_wasserstein": float(conf_wass),
            "entropy_wasserstein": float(ent_wass),
            "drift_score": float(score),
            "drift_detected": drift_flag,
        }

        return result