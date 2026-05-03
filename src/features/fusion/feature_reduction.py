# src/features/fusion/feature_reduction.py
"""
Unified feature-reduction module.

Audit fix #8 — merges what used to live in two parallel modules:

    src/features/feature_pruning.py            (FeaturePruner)
    src/features/fusion/feature_selection.py   (Variance/Correlation/TopK
                                                /Composite selectors,
                                                FeatureSelectionPipeline)

Both modules implemented their own _dict_to_matrix, their own variance
threshold, and their own correlation prune — at *different* default
thresholds (0.95 in both, but with different tie-breaking) — and were
imported separately by FeatureEngineeringPipeline.  The split obscured
which module owned which step and made it impossible to reason about
the order of operations.

This module provides a single canonical implementation:

    * VarianceThresholdSelector   — drops constant / low-variance columns
    * CorrelationSelector         — drops one of every highly-correlated
                                    pair  (default 0.9 — see audit task 8)
    * TopKSelector                — keeps the K highest-scoring columns
    * CompositeSelector           — chains the above
    * FeatureSelectionPipeline    — name-aware fit/transform on
                                    List[Dict[str, float]] with JSON
                                    persistence of the kept-name list
    * FeaturePruner               — variance + correlation prune in one
                                    fit() call (default correlation
                                    threshold 0.9, per audit task 8)
    * FeatureReductionPipeline    — the recommended end-to-end entry
                                    point: fit on the training feature
                                    matrix, persist kept-names, transform
                                    every subsequent (training + inference)
                                    matrix.

The two original modules are kept as thin re-export shims for
backward compatibility — every existing import site continues to work.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.feature_selection import mutual_info_classif
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Mutual information disabled.")


FeatureVector = Dict[str, float]
EPS = 1e-8

# Audit task 8 — single source of truth for the correlation prune
# threshold used across the project.  0.9 is more aggressive than the
# previous 0.95 and is the value the audit explicitly calls for.
DEFAULT_CORRELATION_THRESHOLD = 0.9
DEFAULT_VARIANCE_THRESHOLD = 1e-6


# =========================================================
# UTILITIES
# =========================================================

def _dict_to_matrix(features: List[FeatureVector]) -> Tuple[np.ndarray, List[str]]:
    """
    Build a dense (N, D) float32 matrix from a list of feature dicts.

    Keys are the *union* of every dict's keys (sorted for determinism)
    so the column order is stable across calls regardless of which
    samples happen to be missing which features.  Missing values are
    filled with 0.0 and non-finite values are sanitized.
    """
    if not features:
        raise ValueError("Feature list cannot be empty")

    keys = sorted({k for f in features for k in f.keys()})
    name_to_idx = {k: i for i, k in enumerate(keys)}

    matrix = np.zeros((len(features), len(keys)), dtype=np.float32)

    for i, f in enumerate(features):
        row = matrix[i]
        for k, v in f.items():
            j = name_to_idx.get(k)
            if j is None:
                continue
            if not np.isfinite(v):
                v = 0.0
            row[j] = float(v)

    return matrix, keys


def _matrix_to_dicts(
    matrix: np.ndarray, keys: List[str]
) -> List[FeatureVector]:
    """Convert a (N, D) matrix back into N feature dicts."""
    return [{k: float(v) for k, v in zip(keys, row)} for row in matrix]


# =========================================================
# SELECTORS  (matrix-level — fit/transform on (N, D) ndarray)
# =========================================================

@dataclass
class VarianceThresholdSelector:
    threshold: float = 0.0
    selected_indices: List[int] = field(default_factory=list)
    scores_: Optional[np.ndarray] = None
    fitted: bool = False

    def fit(self, X: np.ndarray, y=None) -> None:
        var = np.var(X, axis=0)
        self.scores_ = var
        self.selected_indices = [i for i, v in enumerate(var) if v > self.threshold]
        self.fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check()
        return X[:, self.selected_indices]

    def _check(self):
        if not self.fitted:
            raise RuntimeError("Selector not fitted")


@dataclass
class CorrelationSelector:
    threshold: float = DEFAULT_CORRELATION_THRESHOLD
    # Audit fix §6.3 — chunk size for the column-block correlation
    # computation. ``np.corrcoef(X, rowvar=False)`` on a 100k×800
    # matrix materialises an 800×800 correlation matrix (cheap, 5MB)
    # *plus* a transient 800×100k transposed copy of X — which is the
    # actual memory blow-up. The chunked path computes
    # ``Z.T @ Z`` (where Z is column-mean-centred / std-scaled) one
    # ``chunk_size`` block of columns at a time so peak memory stays
    # at O(chunk_size × N) instead of O(D × N).
    chunk_size: int = 256
    selected_indices: List[int] = field(default_factory=list)
    fitted: bool = False

    def fit(self, X: np.ndarray, y=None) -> None:

        # Single-column input is trivially uncorrelated.
        if X.shape[1] <= 1:
            self.selected_indices = list(range(X.shape[1]))
            self.fitted = True
            return

        n, d = X.shape

        # Small-matrix fast path: identical to the original behaviour
        # but without the transposed copy, since we already pay for
        # the full corrcoef anyway.
        if d <= self.chunk_size or n <= 1:
            corr = np.corrcoef(X, rowvar=False)
            corr = np.nan_to_num(corr)
            upper = np.triu(np.abs(corr), k=1)

            to_drop: Set[int] = set()
            for i in range(upper.shape[0]):
                if i in to_drop:
                    continue
                for j in np.where(upper[i] > self.threshold)[0]:
                    to_drop.add(int(j))

            self.selected_indices = sorted(set(range(d)) - to_drop)
            self.fitted = True
            return

        # ---------- chunked path ----------
        # Standardise once (column-wise mean / std) so each
        # chunk-vs-chunk product is a correlation by construction.
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        stds_safe = np.where(stds > 0, stds, 1.0)
        Z = (X - means) / stds_safe
        Z = np.ascontiguousarray(Z, dtype=np.float32)
        # ``corr[i, j] = (Z[:, i] @ Z[:, j]) / N``. Constant-column
        # entries get a real std of 0 → we mapped that to 1 above and
        # correct here by zeroing the corresponding rows/cols
        # implicitly (their Z column is all zeros so all dot products
        # are zero too — i.e. uncorrelated, which is the correct
        # semantic for "constant feature").

        to_drop: Set[int] = set()
        chunk = max(1, int(self.chunk_size))

        for i_start in range(0, d, chunk):
            i_end = min(d, i_start + chunk)
            block_i = Z[:, i_start:i_end]

            for j_start in range(i_start, d, chunk):
                j_end = min(d, j_start + chunk)
                block_j = Z[:, j_start:j_end]

                # (chunk × N) @ (N × chunk) → (chunk × chunk)
                corr_block = (block_i.T @ block_j) / float(n)
                corr_block = np.nan_to_num(corr_block)

                # Restrict to strictly-upper-triangular pairs in the
                # global index space.
                rows, cols = np.where(np.abs(corr_block) > self.threshold)
                for r, c in zip(rows, cols):
                    gi = int(i_start + r)
                    gj = int(j_start + c)
                    if gj <= gi:
                        continue
                    if gi in to_drop:
                        continue
                    to_drop.add(gj)

        self.selected_indices = sorted(set(range(d)) - to_drop)
        self.fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check()
        return X[:, self.selected_indices]

    def _check(self):
        if not self.fitted:
            raise RuntimeError("Selector not fitted")


@dataclass
class TopKSelector:
    k: int = 50
    method: str = "variance"

    selected_indices: List[int] = field(default_factory=list)
    scores_: Optional[np.ndarray] = None
    fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:

        if self.method == "variance":
            scores = np.var(X, axis=0)

        elif self.method == "mutual_info":
            if not SKLEARN_AVAILABLE:
                raise RuntimeError("sklearn required for mutual_info")
            if y is None:
                raise ValueError("Labels required")
            scores = mutual_info_classif(X, y)

        else:
            raise ValueError("Invalid method")

        self.scores_ = scores
        ranked = np.argsort(scores)[::-1]
        self.selected_indices = ranked[: self.k].tolist()
        self.fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check()
        return X[:, self.selected_indices]

    def _check(self):
        if not self.fitted:
            raise RuntimeError("Selector not fitted")


@dataclass
class CompositeSelector:
    """Chain selectors; each one is fitted on the *output* of the previous."""

    selectors: List[object]
    fitted: bool = False

    def fit(self, X: np.ndarray, y=None) -> None:
        for sel in self.selectors:
            if hasattr(sel, "fit"):
                if y is not None:
                    sel.fit(X, y)
                else:
                    sel.fit(X)
                X = sel.transform(X)
        self.fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("CompositeSelector not fitted")
        for sel in self.selectors:
            X = sel.transform(X)
        return X


# =========================================================
# NAME-AWARE PIPELINE  (operates on List[Dict[str, float]])
# =========================================================

@dataclass
class FeatureSelectionPipeline:
    """
    Wrap a matrix-level selector with name awareness.  fit() learns the
    set of *kept feature names*; transform() projects new samples down
    to those names regardless of input order or extra/missing keys.
    """

    selector: object

    feature_order: List[str] = field(default_factory=list)
    selected_keys: List[str] = field(default_factory=list)

    _name_to_idx: Dict[str, int] = field(default_factory=dict, init=False)
    fitted: bool = False

    # -----------------------------------------------------

    def fit(
        self,
        features: List[FeatureVector],
        labels: Optional[List[int]] = None,
    ) -> None:

        X, keys = _dict_to_matrix(features)
        self.feature_order = keys
        self._name_to_idx = {k: i for i, k in enumerate(keys)}

        y = np.array(labels) if labels is not None else None

        if hasattr(self.selector, "fit"):
            self.selector.fit(X, y)

        selected_idx = getattr(self.selector, "selected_indices", None)
        if selected_idx is None:
            raise ValueError("Selector must expose selected_indices")

        self.selected_keys = [keys[i] for i in selected_idx]
        self.fitted = True

        logger.info(
            "FeatureSelection fitted | original=%d selected=%d",
            len(keys),
            len(self.selected_keys),
        )

    # -----------------------------------------------------

    def transform(
        self,
        features: List[FeatureVector],
        *,
        return_array: bool = True,
    ):
        if not self.fitted:
            raise RuntimeError("Pipeline not fitted")

        X = np.zeros((len(features), len(self.feature_order)), dtype=np.float32)

        for i, f in enumerate(features):
            for k, v in f.items():
                j = self._name_to_idx.get(k)
                if j is None:
                    continue
                if not np.isfinite(v):
                    v = 0.0
                X[i, j] = float(v)

        X = self.selector.transform(X)

        if return_array:
            return X

        return _matrix_to_dicts(X, self.selected_keys)

    # -----------------------------------------------------

    def fit_transform(
        self,
        features: List[FeatureVector],
        labels: Optional[List[int]] = None,
        *,
        return_array: bool = True,
    ):
        self.fit(features, labels)
        return self.transform(features, return_array=return_array)

    # -----------------------------------------------------
    # PERSISTENCE  (kept-name list as JSON)
    # -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        data = {
            "feature_order": self.feature_order,
            "selected_keys": self.selected_keys,
        }
        Path(path).write_text(json.dumps(data))
        logger.info("FeatureSelection saved → %s", path)

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text())
        self.feature_order = data["feature_order"]
        self.selected_keys = data["selected_keys"]
        self._name_to_idx = {k: i for i, k in enumerate(self.feature_order)}
        self.fitted = True
        logger.info("FeatureSelection loaded ← %s", path)

    def get_selected_features(self) -> List[str]:
        return self.selected_keys


# =========================================================
# COMPATIBILITY: FeaturePruner  (variance + correlation in one call)
# =========================================================

@dataclass
class FeaturePruner:
    """
    Single-call variance + correlation prune.  Default thresholds are
    aligned with audit task 8 (correlation ≥ 0.9 ⇒ drop one of the pair).

    Kept as a stable name because it is referenced from
    FeatureEngineeringPipeline and from saved training configs.
    """

    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD

    removed_features_: Set[str] = field(default_factory=set, init=False)
    kept_features_: List[str] = field(default_factory=list, init=False)

    # -----------------------------------------------------

    def fit(self, features: List[FeatureVector]) -> None:

        X, keys = _dict_to_matrix(features)
        logger.info("Starting feature pruning | features=%d", len(keys))

        # ---------- 1. variance ----------
        vt = VarianceThresholdSelector(threshold=self.variance_threshold)
        vt.fit(X)
        removed_low_variance = [
            keys[i] for i in range(len(keys)) if i not in vt.selected_indices
        ]
        X = vt.transform(X)
        keys = [keys[i] for i in vt.selected_indices]
        logger.info("Removed low variance features: %d", len(removed_low_variance))

        # ---------- 2. correlation ----------
        if X.shape[1] > 1:
            cs = CorrelationSelector(threshold=self.correlation_threshold)
            cs.fit(X)
            removed_corr = [
                keys[i] for i in range(len(keys)) if i not in cs.selected_indices
            ]
            X = cs.transform(X)
            keys = [keys[i] for i in cs.selected_indices]
            logger.info("Removed correlated features: %d", len(removed_corr))
        else:
            removed_corr = []

        self.kept_features_ = keys
        self.removed_features_ = set(removed_low_variance + removed_corr)

        logger.info(
            "Feature pruning complete | kept=%d removed=%d",
            len(self.kept_features_),
            len(self.removed_features_),
        )

    # -----------------------------------------------------

    def transform(self, features: List[FeatureVector]) -> List[FeatureVector]:

        if not self.kept_features_:
            raise RuntimeError("FeaturePruner must be fitted first")

        return [
            {k: float(f.get(k, 0.0)) for k in self.kept_features_}
            for f in features
        ]

    def fit_transform(
        self, features: List[FeatureVector]
    ) -> List[FeatureVector]:
        self.fit(features)
        return self.transform(features)

    def get_removed_features(self) -> List[str]:
        return sorted(self.removed_features_)

    def get_kept_features(self) -> List[str]:
        return list(self.kept_features_)


# =========================================================
# RECOMMENDED END-TO-END PIPELINE  (audit task 8)
#
# Drop-in replacement for "FeaturePruner THEN FeatureSelectionPipeline"
# that:
#   * runs variance threshold + correlation prune at 0.9 on the
#     *training* feature matrix
#   * persists the kept-name list to JSON next to the model checkpoint
#   * applies the saved kept-name set at inference time, regardless of
#     whether the inference-time feature dict happens to expose extra
#     or missing keys.
# =========================================================

@dataclass
class FeatureReductionPipeline:

    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD

    kept_features_: List[str] = field(default_factory=list, init=False)
    removed_features_: List[str] = field(default_factory=list, init=False)
    fitted: bool = False

    # -----------------------------------------------------

    def fit(self, features: List[FeatureVector]) -> None:

        pruner = FeaturePruner(
            variance_threshold=self.variance_threshold,
            correlation_threshold=self.correlation_threshold,
        )
        pruner.fit(features)

        self.kept_features_ = pruner.get_kept_features()
        self.removed_features_ = pruner.get_removed_features()
        self.fitted = True

        logger.info(
            "FeatureReductionPipeline fitted | kept=%d removed=%d "
            "variance_threshold=%g correlation_threshold=%g",
            len(self.kept_features_),
            len(self.removed_features_),
            self.variance_threshold,
            self.correlation_threshold,
        )

    # -----------------------------------------------------

    def transform(
        self,
        features: List[FeatureVector],
        *,
        return_array: bool = False,
    ):
        if not self.fitted:
            raise RuntimeError("FeatureReductionPipeline not fitted")

        if return_array:
            X = np.zeros(
                (len(features), len(self.kept_features_)),
                dtype=np.float32,
            )
            for i, f in enumerate(features):
                for j, k in enumerate(self.kept_features_):
                    v = f.get(k, 0.0)
                    if not np.isfinite(v):
                        v = 0.0
                    X[i, j] = float(v)
            return X

        return [
            {k: float(f.get(k, 0.0)) for k in self.kept_features_}
            for f in features
        ]

    def fit_transform(
        self,
        features: List[FeatureVector],
        *,
        return_array: bool = False,
    ):
        self.fit(features)
        return self.transform(features, return_array=return_array)

    # -----------------------------------------------------
    # PERSISTENCE
    # -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        payload = {
            "schema": "feature_reduction.v1",
            "variance_threshold": self.variance_threshold,
            "correlation_threshold": self.correlation_threshold,
            "kept_features": list(self.kept_features_),
            "removed_features": list(self.removed_features_),
        }
        Path(path).write_text(json.dumps(payload, indent=2))
        logger.info(
            "FeatureReductionPipeline saved → %s (kept=%d)",
            path,
            len(self.kept_features_),
        )

    def load(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text())
        if payload.get("schema") != "feature_reduction.v1":
            raise ValueError(
                f"Unrecognised schema: {payload.get('schema')!r}"
            )
        self.variance_threshold = float(payload["variance_threshold"])
        self.correlation_threshold = float(payload["correlation_threshold"])
        self.kept_features_ = list(payload["kept_features"])
        self.removed_features_ = list(payload.get("removed_features", []))
        self.fitted = True
        logger.info(
            "FeatureReductionPipeline loaded ← %s (kept=%d)",
            path,
            len(self.kept_features_),
        )

    def get_kept_features(self) -> List[str]:
        return list(self.kept_features_)

    def get_removed_features(self) -> List[str]:
        return list(self.removed_features_)


__all__ = [
    "DEFAULT_CORRELATION_THRESHOLD",
    "DEFAULT_VARIANCE_THRESHOLD",
    "VarianceThresholdSelector",
    "CorrelationSelector",
    "TopKSelector",
    "CompositeSelector",
    "FeatureSelectionPipeline",
    "FeaturePruner",
    "FeatureReductionPipeline",
]
