from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from src.visualization.visualize import plot_feature_importance
from src.features.base.numerics import EPS

logger = logging.getLogger(__name__)

FeatureVector = Dict[str, float]

# =========================================================
# SAFE MATRIX CONVERSION (FIXED)
# =========================================================

def _dict_to_matrix(features: List[FeatureVector]) -> Tuple[np.ndarray, List[str]]:
    """
    Robust conversion: union of all keys + safe numeric handling.
    """

    if not features:
        raise ValueError("Feature list cannot be empty")

    all_keys = set()
    for f in features:
        all_keys.update(f.keys())

    keys = sorted(all_keys)
    index = {k: i for i, k in enumerate(keys)}

    X = np.zeros((len(features), len(keys)), dtype=np.float32)

    for i, f in enumerate(features):
        row = X[i]
        for k, v in f.items():
            j = index.get(k)
            if j is None:
                continue

            if isinstance(v, (int, float)):
                val = float(v)
                if np.isfinite(val):
                    row[j] = val

    return X, keys


# =========================================================
# MAIN CLASS
# =========================================================

@dataclass
class FeatureStatistics:

    # Audit fix §1.7 — the cache used to be unkeyed: the FIRST feature
    # list ever passed in was returned for every subsequent call,
    # silently corrupting any pipeline that reused a ``FeatureStatistics``
    # instance across batches. The cache is now keyed on
    # ``(id(features), len(features), len(features[0]) if features else 0)``
    # which catches the common "same list passed back-to-back to several
    # stat methods" pattern without false hits across batches.
    _cached_matrix: Optional[np.ndarray] = field(default=None, init=False)
    _cached_keys: Optional[List[str]] = field(default=None, init=False)
    _cached_signature: Optional[Tuple[int, int, int]] = field(default=None, init=False)

    # -----------------------------------------------------
    # CACHE MATRIX (BIG PERFORMANCE WIN)
    # -----------------------------------------------------

    def _get_matrix(self, features: List[FeatureVector]) -> Tuple[np.ndarray, List[str]]:

        signature = (
            id(features),
            len(features),
            len(features[0]) if features else 0,
        )

        if (
            self._cached_signature == signature
            and self._cached_matrix is not None
            and self._cached_keys is not None
        ):
            return self._cached_matrix, self._cached_keys

        X, keys = _dict_to_matrix(features)

        self._cached_matrix = X
        self._cached_keys = keys
        self._cached_signature = signature

        return X, keys

    # =====================================================
    # BASIC STATS
    # =====================================================

    def compute_basic_statistics(
        self,
        features: List[FeatureVector],
    ) -> Dict[str, Dict[str, float]]:

        X, keys = self._get_matrix(features)

        stats: Dict[str, Dict[str, float]] = {}

        for i, name in enumerate(keys):
            col = X[:, i]

            stats[name] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "min": float(np.min(col)),
                "max": float(np.max(col)),
                "median": float(np.median(col)),
            }

        return stats

    # =====================================================
    # VARIANCE
    # =====================================================

    def compute_variance(self, features: List[FeatureVector]) -> Dict[str, float]:

        X, keys = self._get_matrix(features)

        var = np.var(X, axis=0)

        return {k: float(v) for k, v in zip(keys, var)}

    # =====================================================
    # CONSTANT FEATURES
    # =====================================================

    def detect_constant_features(
        self,
        features: List[FeatureVector],
        tolerance: float = 1e-12,
    ) -> List[str]:

        X, keys = self._get_matrix(features)

        var = np.var(X, axis=0)

        constant = [keys[i] for i, v in enumerate(var) if v < tolerance]

        return constant

    # =====================================================
    # SKEWNESS
    # =====================================================

    def compute_skewness(self, features: List[FeatureVector]) -> Dict[str, float]:

        X, keys = self._get_matrix(features)

        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)

        std[std < EPS] = 1.0

        skew = np.mean(((X - mean) / std) ** 3, axis=0)

        return {k: float(v) for k, v in zip(keys, skew)}

    # =====================================================
    # CORRELATION (FIXED)
    # =====================================================

    def compute_correlation_matrix(
        self,
        features: List[FeatureVector],
    ) -> Tuple[np.ndarray, List[str]]:

        X, keys = self._get_matrix(features)

        std = np.std(X, axis=0)
        valid = std > EPS

        if not np.any(valid):
            return np.zeros((len(keys), len(keys))), keys

        X_valid = X[:, valid]

        corr = np.corrcoef(X_valid, rowvar=False)

        # rebuild full matrix
        full_corr = np.zeros((len(keys), len(keys)))

        idx = np.where(valid)[0]
        for i, ii in enumerate(idx):
            for j, jj in enumerate(idx):
                full_corr[ii, jj] = corr[i, j]

        return full_corr, keys

    # =====================================================
    # TOP FEATURES (NEW)
    # =====================================================

    def top_k_variance(
        self,
        features: List[FeatureVector],
        k: int = 20,
    ) -> List[Tuple[str, float]]:

        var = self.compute_variance(features)

        return sorted(var.items(), key=lambda x: x[1], reverse=True)[:k]

    # =====================================================
    # DATASET SUMMARY
    # =====================================================

    def dataset_summary(
        self,
        features: List[FeatureVector],
    ) -> Dict[str, float]:

        X, _ = self._get_matrix(features)

        var = np.var(X, axis=0)

        return {
            "num_samples": float(X.shape[0]),
            "num_features": float(X.shape[1]),
            "mean_variance": float(np.mean(var)),
            "max_variance": float(np.max(var)),
            "min_variance": float(np.min(var)),
        }

    # =====================================================
    # DRIFT DETECTION (NEW 🔥)
    # =====================================================

    def compare_distributions(
        self,
        features_a: List[FeatureVector],
        features_b: List[FeatureVector],
    ) -> Dict[str, float]:
        """
        Mean shift between two datasets (simple drift signal)
        """

        Xa, keys = _dict_to_matrix(features_a)
        Xb, _ = _dict_to_matrix(features_b)

        mean_a = np.mean(Xa, axis=0)
        mean_b = np.mean(Xb, axis=0)

        shift = np.abs(mean_a - mean_b)

        return {k: float(v) for k, v in zip(keys, shift)}

    # =====================================================
    # VISUALIZATION
    # =====================================================

    def save_variance_plot(
        self,
        features: List[FeatureVector],
        *,
        output_path: str | Path,
        top_k: int = 25,
    ) -> Path:

        variances = self.compute_variance(features)

        plot_feature_importance(
            features=list(variances.keys()),
            scores=list(variances.values()),
            top_k=min(top_k, len(variances)),
            save_path=output_path,
        )

        return Path(output_path)