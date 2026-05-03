# src/features/importance/shap_importance.py
"""SHAP-based importance utilities — OFFLINE-ONLY.

Audit fix §8 — used exclusively by
``src/evaluation/advanced_analysis.py`` for explainability reports.
SHAP value computation is exponential in the number of features
under exact attribution and several seconds per sample under the
sampling approximation, so this module must not be wired into any
inference path. Do not import from ``src/inference/`` or
``api/app.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from src.features.base.numerics import EPS

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False
    shap = None
    logger.warning("SHAP not available")


# =========================================================
# SHAP IMPORTANCE
# =========================================================

@dataclass
class ShapImportance:

    # CRIT-E-ADVANCED-BROKEN fix: model is now optional; callers that rely
    # on compute_with_function supply a predict_fn instead of a model object.
    model: Optional[object] = None
    max_samples: Optional[int] = 1000
    batch_size: int = 128
    random_seed: int = 42

    use_interactions: bool = False

    _explainer: Optional[object] = field(default=None, init=False)

    # -----------------------------------------------------
    # EXPLAINER (SMART)
    # -----------------------------------------------------

    def _create_explainer(self, X: np.ndarray):

        if not SHAP_AVAILABLE:
            raise RuntimeError("SHAP required")

        # CRIT-E-ADVANCED-BROKEN fix: model is now Optional; guard before
        # attribute access so ShapImportance() / ShapImportance(model=None)
        # doesn't crash here — callers should use compute_with_function when
        # they have only a predict_fn and no model object.
        if self.model is None:
            raise RuntimeError(
                "ShapImportance.model is None; use compute_with_function "
                "to pass a predict_fn instead of a model object."
            )

        if hasattr(self.model, "predict_proba"):
            try:
                return shap.TreeExplainer(self.model)
            except Exception:
                pass

        try:
            return shap.LinearExplainer(self.model, X)
        except Exception:
            pass

        # fallback
        background = X[: min(len(X), 100)]

        predict_fn = (
            self.model.predict_proba
            if hasattr(self.model, "predict_proba")
            else self.model.predict
        )

        return shap.KernelExplainer(predict_fn, background)

    # -----------------------------------------------------

    def _get_explainer(self, X: np.ndarray):
        if self._explainer is None:
            self._explainer = self._create_explainer(X)
        return self._explainer

    # -----------------------------------------------------
    # SAMPLING
    # -----------------------------------------------------

    def _sample(self, X: np.ndarray):

        if self.max_samples is None or len(X) <= self.max_samples:
            return X.astype(np.float32)

        rng = np.random.default_rng(self.random_seed)
        idx = rng.choice(len(X), self.max_samples, replace=False)

        logger.info("Sampling %d rows for SHAP", self.max_samples)
        return X[idx].astype(np.float32)

    # -----------------------------------------------------
    # BATCH SHAP
    # -----------------------------------------------------

    def _compute_shap(self, explainer, X: np.ndarray):

        values_list = []

        for i in range(0, len(X), self.batch_size):

            batch = X[i : i + self.batch_size]

            if self.use_interactions and hasattr(explainer, "shap_interaction_values"):
                vals = explainer.shap_interaction_values(batch)
            else:
                vals = explainer(batch)

            values_list.append(vals)

        return values_list

    # -----------------------------------------------------
    # PROCESS VALUES
    # -----------------------------------------------------

    def _process(self, shap_values):

        values = getattr(shap_values, "values", shap_values)

        if isinstance(values, list):
            values = np.stack(values, axis=0)

        values = np.asarray(values, dtype=np.float32)

        # multi-class
        if values.ndim == 3:
            values = values[:, :, -1]

        values = np.nan_to_num(values)

        return values

    # -----------------------------------------------------
    # MAIN
    # -----------------------------------------------------

    def compute(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, float]:

        if not SHAP_AVAILABLE:
            return {name: 0.0 for name in feature_names}

        X_sample = self._sample(X)

        try:
            explainer = self._get_explainer(X_sample)

            shap_batches = self._compute_shap(explainer, X_sample)

            values = np.vstack([
                self._process(v) for v in shap_batches
            ])

        except Exception as e:
            logger.warning("SHAP failed: %s", e)
            return {name: 0.0 for name in feature_names}

        # -------------------------
        # IMPORTANCE
        # -------------------------

        abs_vals = np.abs(values)

        mean_vals = abs_vals.mean(axis=0)
        std_vals = abs_vals.std(axis=0)

        # 🔥 L1 normalization (better)
        total = mean_vals.sum() + EPS
        mean_vals = mean_vals / total

        return {
            name: float(score)
            for name, score in zip(feature_names, mean_vals)
        }

    # -----------------------------------------------------
    # CRIT-E-ADVANCED-BROKEN fix: entry point for callers that supply a
    # predict_fn closure instead of an sklearn-compatible model object.
    # advanced_analysis.shap_importance() calls this method.
    # -----------------------------------------------------

    def compute_with_function(
        self,
        predict_fn: Callable,
        X,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute SHAP importance using an external predict_fn.

        Temporarily installs ``predict_fn`` wrapped in a thin sklearn-
        compatible adapter so the regular ``compute`` path can be reused.
        When ``X`` is not a 2-D numeric array (e.g. a list of strings),
        SHAP column-level attribution is not applicable; the method returns
        zero importance for all features and logs a warning.

        Args:
            predict_fn: callable(X) -> ndarray of predictions / probabilities.
            X: feature matrix or raw inputs.
            feature_names: column labels.  Auto-generated when ``None`` and
                ``X`` is a 2-D array; an empty list is used for non-tabular
                inputs.
        """
        arr = np.asarray(X) if not isinstance(X, np.ndarray) else X
        if arr.ndim != 2:
            logger.warning(
                "compute_with_function: X has shape %s (expected 2-D); "
                "SHAP attribution is not applicable to non-tabular inputs — "
                "returning zero importance.",
                arr.shape,
            )
            return {name: 0.0 for name in (feature_names or [])}

        names = feature_names or [f"feature_{i}" for i in range(arr.shape[1])]

        # Install a thin adapter so _create_explainer / _get_explainer work.
        class _Adapter:
            def predict(self_, a: np.ndarray) -> np.ndarray:  # noqa: N805
                return np.asarray(predict_fn(a))

            def predict_proba(self_, a: np.ndarray) -> np.ndarray:  # noqa: N805
                return np.asarray(predict_fn(a)).astype(float)

        saved_model = self.model
        saved_explainer = self._explainer
        try:
            self.model = _Adapter()
            self._explainer = None
            return self.compute(arr, names)
        finally:
            self.model = saved_model
            self._explainer = saved_explainer

    # -----------------------------------------------------
    # VARIANCE (RESEARCH)
    # -----------------------------------------------------

    def compute_with_variance(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Tuple[float, float]]:

        X_sample = self._sample(X)

        explainer = self._get_explainer(X_sample)

        shap_vals = explainer(X_sample)

        values = self._process(shap_vals)

        abs_vals = np.abs(values)

        mean = abs_vals.mean(axis=0)
        std = abs_vals.std(axis=0)

        total = mean.sum() + EPS
        mean = mean / total
        std = std / total

        return {
            name: (float(m), float(s))
            for name, m, s in zip(feature_names, mean, std)
        }

    # -----------------------------------------------------
    # GROUPING
    # -----------------------------------------------------

    def group_importance(
        self,
        X: np.ndarray,
        feature_names: List[str],
        groups: Dict[str, List[str]],
    ) -> Dict[str, float]:

        base_scores = self.compute(X, feature_names)

        results = {}

        for g, feats in groups.items():
            val = sum(base_scores.get(f, 0.0) for f in feats)
            results[g] = float(val)

        return results

    # -----------------------------------------------------

    def rank_features(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ) -> List[Tuple[str, float]]:

        scores = self.compute(X, feature_names)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # -----------------------------------------------------

    def top_k(
        self,
        X: np.ndarray,
        feature_names: List[str],
        k: int = 20,
    ) -> List[Tuple[str, float]]:

        return self.rank_features(X, feature_names)[:k]