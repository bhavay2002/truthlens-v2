# src/features/importance/feature_ablation.py
"""Feature ablation utilities — OFFLINE-ONLY.

Audit fix §8 — this module is consumed exclusively by
``src/evaluation/advanced_analysis.py`` for post-hoc explainability
runs. It is **not** imported by ``src/inference/``, ``api/app.py``,
or any model forward path. Do not register it on the live request
pipeline; running an ablation sweep at request time would multiply
latency by the number of features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

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
# FEATURE ABLATION
# =========================================================

@dataclass
class FeatureAblation:

    # CRIT-E-ADVANCED-BROKEN fix: model is now optional so callers that
    # supply a predict_fn closure (e.g. advanced_analysis wrappers) can
    # construct the class without an sklearn-compatible object.
    model: Optional[object] = None
    metric: MetricFn = accuracy_metric
    normalize: bool = True
    bootstrap_runs: int = 0  # >0 enables variance estimation

    # CRIT-E-ADVANCED-BROKEN fix: accept a raw callable as an alternative
    # to a model object.
    predict_fn: Optional[Callable] = None

    _baseline_cache: float | None = None

    # -----------------------------------------------------

    def _predict(self, X: np.ndarray) -> np.ndarray:
        # CRIT-E-ADVANCED-BROKEN: prefer predict_fn over model.
        if self.predict_fn is not None:
            return np.asarray(self.predict_fn(X))
        if self.model is not None and hasattr(self.model, "predict"):
            return self.model.predict(X)
        raise RuntimeError("FeatureAblation requires model or predict_fn")

    # -----------------------------------------------------

    def _baseline_score(self, X: np.ndarray, y: np.ndarray) -> float:

        if self._baseline_cache is not None:
            return self._baseline_cache

        pred = self._predict(X)
        score = self.metric(y, pred)

        self._baseline_cache = score

        logger.info("Baseline score: %.6f", score)
        return score

    # -----------------------------------------------------
    # FAST SINGLE FEATURE ABLATION
    # -----------------------------------------------------

    def single_feature_ablation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, float]:

        # CRIT-E-ADVANCED-BROKEN fix: callers (e.g. advanced_analysis
        # wrappers) may pass text lists or 1-D arrays.  Column ablation is
        # only meaningful for 2-D tabular arrays; return zero importance with
        # a warning rather than crashing with an unpack / indexing TypeError.
        X = np.asarray(X) if not isinstance(X, np.ndarray) else X
        if X.ndim != 2:
            logger.warning(
                "single_feature_ablation: X has shape %s (expected 2-D); "
                "column ablation is not applicable — returning zero importance "
                "for all %d features.",
                X.shape,
                len(feature_names),
            )
            return {name: 0.0 for name in feature_names}

        baseline = self._baseline_score(X, y)

        n_samples, n_features = X.shape

        results: Dict[str, float] = {}

        for i, name in enumerate(feature_names):

            X_ablate = X.copy()
            X_ablate[:, i] = 0.0

            pred = self._predict(X_ablate)
            score = self.metric(y, pred)

            impact = baseline - score

            # 🔥 normalization (important)
            if self.normalize:
                impact = impact / (abs(baseline) + 1e-8)

            results[name] = float(impact)

        return results

    # -----------------------------------------------------
    # GROUP ABLATION
    # -----------------------------------------------------

    def group_ablation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        groups: Dict[str, List[str]],
    ) -> Dict[str, float]:

        baseline = self._baseline_score(X, y)

        name_to_idx = {name: i for i, name in enumerate(feature_names)}

        results: Dict[str, float] = {}

        for group_name, group_features in groups.items():

            indices = [name_to_idx[f] for f in group_features if f in name_to_idx]

            if not indices:
                results[group_name] = 0.0
                continue

            X_ablate = X.copy()
            X_ablate[:, indices] = 0.0

            pred = self._predict(X_ablate)
            score = self.metric(y, pred)

            impact = baseline - score

            if self.normalize:
                impact = impact / (abs(baseline) + 1e-8)

            results[group_name] = float(impact)

        return results

    # -----------------------------------------------------
    # BOOTSTRAP VARIANCE (RESEARCH)
    # -----------------------------------------------------

    def bootstrap_ablation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Tuple[float, float]]:

        if self.bootstrap_runs <= 0:
            raise ValueError("bootstrap_runs must be > 0")

        n = len(X)
        all_scores = {f: [] for f in feature_names}

        for _ in range(self.bootstrap_runs):

            idx = np.random.choice(n, n, replace=True)

            X_sample = X[idx]
            y_sample = y[idx]

            scores = self.single_feature_ablation(X_sample, y_sample, feature_names)

            for k, v in scores.items():
                all_scores[k].append(v)

        results = {}

        for k, values in all_scores.items():
            arr = np.array(values)
            results[k] = (float(arr.mean()), float(arr.std()))

        return results

    # -----------------------------------------------------
    # RANKING
    # -----------------------------------------------------

    def rank_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> List[Tuple[str, float]]:

        scores = self.single_feature_ablation(X, y, feature_names)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return ranked

    # -----------------------------------------------------

    def top_k(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        k: int = 20,
    ) -> List[Tuple[str, float]]:

        return self.rank_features(X, y, feature_names)[:k]