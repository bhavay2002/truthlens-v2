from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

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


def _safe_cov(x: np.ndarray) -> np.ndarray:
    if x.shape[0] < 2:
        return np.eye(x.shape[1])
    cov = np.cov(x, rowvar=False)
    cov = cov + EPS * np.eye(cov.shape[0])
    return cov


def _mean(x: np.ndarray) -> np.ndarray:
    return np.mean(x, axis=0)


def _normalize(x: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0) + EPS
    return (x - mean) / std


# =========================================================
# DISTANCES
# =========================================================

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + EPS)
    b_norm = b / (np.linalg.norm(b) + EPS)
    return float(1.0 - np.dot(a_norm, b_norm))


def mahalanobis_distance(
    mean_a: np.ndarray,
    mean_b: np.ndarray,
    cov: np.ndarray,
) -> float:
    diff = mean_a - mean_b
    inv_cov = np.linalg.pinv(cov)
    return float(np.sqrt(diff.T @ inv_cov @ diff))


def frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
) -> float:
    diff = mu1 - mu2
    covmean = _sqrtm(sigma1 @ sigma2)

    return float(
        diff @ diff
        + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    )


def _sqrtm(mat: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 0, None)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T


# =========================================================
# MMD (KERNEL DRIFT)
# =========================================================

def _rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    x_norm = np.sum(x**2, axis=1).reshape(-1, 1)
    y_norm = np.sum(y**2, axis=1).reshape(1, -1)
    dist = x_norm + y_norm - 2 * np.dot(x, y.T)
    return np.exp(-gamma * dist)


def mmd_rbf(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    k_xx = _rbf_kernel(x, x, gamma)
    k_yy = _rbf_kernel(y, y, gamma)
    k_xy = _rbf_kernel(x, y, gamma)

    return float(
        k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    )


# =========================================================
# STATE
# =========================================================

@dataclass
class EmbeddingDriftState:
    ref_embeddings: Optional[np.ndarray] = None
    ref_mean: Optional[np.ndarray] = None
    ref_cov: Optional[np.ndarray] = None


# =========================================================
# DETECTOR
# =========================================================

class EmbeddingDriftDetector:
    """
    Drift detection on embedding distributions.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        use_normalization: bool = True,
    ) -> None:
        self.threshold = threshold
        self.use_normalization = use_normalization
        self.state = EmbeddingDriftState()

    # =====================================================
    # FIT
    # =====================================================

    def fit(self, embeddings: np.ndarray) -> None:
        x = _ensure_2d(embeddings)

        if self.use_normalization:
            x = _normalize(x)

        self.state.ref_embeddings = x
        self.state.ref_mean = _mean(x)
        self.state.ref_cov = _safe_cov(x)

        logger.info("[EMBED DRIFT] baseline fitted")

    def reset(self) -> None:
        self.state = EmbeddingDriftState()

    # =====================================================
    # UPDATE
    # =====================================================

    def update(self, embeddings: np.ndarray) -> Dict[str, float]:
        if self.state.ref_embeddings is None:
            raise RuntimeError("Call fit() first")

        x = _ensure_2d(embeddings)

        if self.use_normalization:
            x = _normalize(x)

        ref = self.state.ref_embeddings

        mu_ref = self.state.ref_mean
        cov_ref = self.state.ref_cov

        mu_cur = _mean(x)
        cov_cur = _safe_cov(x)

        # Distances
        euclid = euclidean_distance(mu_ref, mu_cur)
        cosine = cosine_distance(mu_ref, mu_cur)
        mahal = mahalanobis_distance(mu_ref, mu_cur, cov_ref)
        frechet = frechet_distance(mu_ref, cov_ref, mu_cur, cov_cur)
        mmd = mmd_rbf(ref, x)

        score = float(np.mean([euclid, cosine, mahal, mmd]))

        drift = score > self.threshold

        return {
            "euclidean": euclid,
            "cosine": cosine,
            "mahalanobis": mahal,
            "frechet": frechet,
            "mmd": mmd,
            "drift_score": score,
            "drift_detected": drift,
        }