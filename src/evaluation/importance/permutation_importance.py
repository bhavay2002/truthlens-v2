# src/features/importance/permutation_importance.py
"""Permutation-importance utilities — OFFLINE-ONLY.

Audit fix §8 — used exclusively by
``src/evaluation/advanced_analysis.py``. Permutation importance
requires K full-dataset re-scoring passes per feature, which makes
it inappropriate for any inference path. Do not import from
``src/inference/`` or ``api/app.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
from src.features.base.numerics import EPS

logger = logging.getLogger(__name__)

MetricFn = Callable[[np.ndarray, np.ndarray], float]

# =========================================================
# DEFAULT METRICS
# =========================================================

def accuracy_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def mse_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


# =========================================================
# PERMUTATION IMPORTANCE
# =========================================================

@dataclass
class PermutationImportance:

    model: object
    metric: MetricFn = accuracy_metric

    n_repeats: int = 5
    random_seed: int = 42

    normalize: bool = True
    use_proba: bool = False

    _baseline_cache: Optional[float] = None

    # -----------------------------------------------------

    def _predict(self, X: np.ndarray) -> np.ndarray:

        if self.use_proba and hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)

        if hasattr(self.model, "predict"):
            return self.model.predict(X)

        raise RuntimeError("Model must implement predict()")

    # -----------------------------------------------------

    def _baseline(self, X: np.ndarray, y: np.ndarray) -> float:

        if self._baseline_cache is not None:
            return self._baseline_cache

        pred = self._predict(X)
        score = self.metric(y, pred)

        if not np.isfinite(score):
            raise ValueError("Baseline metric invalid")

        self._baseline_cache = score

        logger.info("Baseline score: %.6f", score)
        return score

    # -----------------------------------------------------
    # CORE COMPUTE
    # -----------------------------------------------------

    def compute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, float]:

        if self.n_repeats <= 0:
            raise ValueError("n_repeats must be > 0")

        baseline = self._baseline(X, y)

        rng = np.random.default_rng(self.random_seed)

        importances: Dict[str, float] = {}

        for j, name in enumerate(feature_names):

            scores = []

            col = X[:, j].copy()

            for _ in range(self.n_repeats):

                shuffled = rng.permutation(col)

                X[:, j] = shuffled

                pred = self._predict(X)
                score = self.metric(y, pred)

                if not np.isfinite(score):
                    raise ValueError(f"Invalid score for feature {name}")

                scores.append(baseline - score)

            # restore original column
            X[:, j] = col

            mean_imp = float(np.mean(scores))

            if self.normalize:
                mean_imp = mean_imp / (abs(baseline) + EPS)

            importances[name] = mean_imp

        return importances

    # -----------------------------------------------------
    # VARIANCE (IMPORTANT)
    # -----------------------------------------------------

    def compute_with_variance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Tuple[float, float]]:

        baseline = self._baseline(X, y)

        rng = np.random.default_rng(self.random_seed)

        results = {}

        for j, name in enumerate(feature_names):

            scores = []
            col = X[:, j].copy()

            for _ in range(self.n_repeats):

                X[:, j] = rng.permutation(col)

                pred = self._predict(X)
                score = self.metric(y, pred)

                scores.append(baseline - score)

            X[:, j] = col

            arr = np.array(scores)

            mean = float(arr.mean())
            std = float(arr.std())

            if self.normalize:
                mean = mean / (abs(baseline) + EPS)
                std = std / (abs(baseline) + EPS)

            results[name] = (mean, std)

        return results

    # -----------------------------------------------------
    # GROUP PERMUTATION (ADVANCED)
    # -----------------------------------------------------

    def group_permutation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        groups: Dict[str, List[str]],
    ) -> Dict[str, float]:

        baseline = self._baseline(X, y)

        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        rng = np.random.default_rng(self.random_seed)

        results = {}

        for group, feats in groups.items():

            indices = [name_to_idx[f] for f in feats if f in name_to_idx]

            if not indices:
                results[group] = 0.0
                continue

            scores = []

            original = X[:, indices].copy()

            for _ in range(self.n_repeats):

                for idx in indices:
                    X[:, idx] = rng.permutation(X[:, idx])

                pred = self._predict(X)
                score = self.metric(y, pred)

                scores.append(baseline - score)

            X[:, indices] = original

            val = float(np.mean(scores))

            if self.normalize:
                val = val / (abs(baseline) + EPS)

            results[group] = val

        return results

    # -----------------------------------------------------

    def rank_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> List[Tuple[str, float]]:

        scores = self.compute(X, y, feature_names)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # -----------------------------------------------------

    def top_k(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        k: int = 20,
    ) -> List[Tuple[str, float]]:

        return self.rank_features(X, y, feature_names)[:k]