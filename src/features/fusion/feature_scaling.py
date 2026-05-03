from __future__ import annotations

"""
Feature scaling.

Defines :class:`FeatureScalingPipeline` (alias :data:`FeatureScaler`), the
per-feature numeric scaler used by the feature-engineering pipeline
(training + inference). Replaces the per-extractor hard-coded
``value / 20.0`` style normalizations and the in-extractor sum-to-one
passes that previously polluted the layer.

The previous version of this file also contained ``HybridTruthLensModel``
— the multi-head transformer + engineered-feature fusion model. That class
has been moved to :mod:`src.models.architectures.hybrid_truthlens_model`
where it belongs. A deprecation re-export is preserved at the bottom of
this module so any straggling imports of
``src.features.fusion.feature_scaling.HybridTruthLensModel`` keep working
with a one-shot warning.
"""

import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# =========================================================
# FEATURE SCALER
# =========================================================

EPS = 1e-8

ScalingMethod = str  # "standard" | "minmax" | "robust"


@dataclass
class FeatureScalingPipeline:
    """
    Per-feature scaler with persistent state.

    Designed for the feature-engineering layer where every sample is a
    `Dict[str, float]` whose keys may vary slightly between calls (extractors
    can be added/removed). Only features observed at `fit` time are scaled at
    `transform` time; unknown keys pass through with a one-shot warning.

    Methods
    -------
    fit(features)
        Learn per-feature statistics from a list of feature dicts.
    transform(features, return_array=False)
        Apply the learned scaling. With return_array=True, returns a numpy
        matrix (samples × features) plus the column order is `feature_names_`.
    fit_transform(features, return_array=False)
    save(path) / load(path)
        Persist learned state to JSON (no pickle, safe to ship).

    Parameters
    ----------
    method : "standard" | "minmax" | "robust"
        - standard : (x - mean) / std
        - minmax   : (x - min) / (max - min) → [0, 1]
        - robust   : (x - median) / IQR
    clip : optional (low, high)
        If set, scaled values are clipped post-transform.
    """

    method: ScalingMethod = "standard"
    clip: Optional[Tuple[float, float]] = None

    # --- learned state --------------------------------------------------
    feature_names_: List[str] = field(default_factory=list)
    stats_: Dict[str, Dict[str, float]] = field(default_factory=dict)
    fitted_: bool = False

    # internal: whether we've already warned about an unseen key (per key)
    _warned_unknown: set = field(default_factory=set)

    # =====================================================
    # FIT
    # =====================================================

    def fit(self, features: Sequence[Dict[str, float]]) -> "FeatureScalingPipeline":
        if not features:
            raise ValueError("FeatureScalingPipeline.fit: empty feature list")

        if self.method not in {"standard", "minmax", "robust"}:
            raise ValueError(f"Unknown scaling method: {self.method}")

        # Collect column-major arrays per feature key
        keys: List[str] = sorted({k for fd in features for k in fd.keys()})
        if not keys:
            raise ValueError("FeatureScalingPipeline.fit: no feature keys present")

        n = len(features)
        matrix = np.zeros((n, len(keys)), dtype=np.float64)
        for i, fd in enumerate(features):
            for j, key in enumerate(keys):
                v = fd.get(key, 0.0)
                if v is None or not np.isfinite(v):
                    v = 0.0
                matrix[i, j] = float(v)

        self.feature_names_ = keys
        self.stats_ = {}

        if self.method == "standard":
            mean = matrix.mean(axis=0)
            std = matrix.std(axis=0)
            std = np.where(std < EPS, 1.0, std)  # guard against zero-variance
            for j, k in enumerate(keys):
                self.stats_[k] = {"mean": float(mean[j]), "std": float(std[j])}

        elif self.method == "minmax":
            mn = matrix.min(axis=0)
            mx = matrix.max(axis=0)
            rng = mx - mn
            rng = np.where(rng < EPS, 1.0, rng)
            for j, k in enumerate(keys):
                self.stats_[k] = {"min": float(mn[j]), "range": float(rng[j])}

        elif self.method == "robust":
            med = np.median(matrix, axis=0)
            q75 = np.percentile(matrix, 75, axis=0)
            q25 = np.percentile(matrix, 25, axis=0)
            iqr = q75 - q25
            iqr = np.where(iqr < EPS, 1.0, iqr)
            for j, k in enumerate(keys):
                self.stats_[k] = {"median": float(med[j]), "iqr": float(iqr[j])}

        self.fitted_ = True
        self._warned_unknown.clear()
        logger.info(
            "FeatureScalingPipeline fitted | method=%s features=%d samples=%d",
            self.method, len(keys), n,
        )
        return self

    # =====================================================
    # TRANSFORM
    # =====================================================

    def _scale_value(self, key: str, value: float) -> float:
        s = self.stats_.get(key)
        if s is None:
            if key not in self._warned_unknown:
                logger.warning(
                    "FeatureScalingPipeline: unseen feature at transform time: %s",
                    key,
                )
                self._warned_unknown.add(key)
            return float(value)

        if not np.isfinite(value):
            value = 0.0

        if self.method == "standard":
            out = (value - s["mean"]) / (s["std"] + EPS)
        elif self.method == "minmax":
            out = (value - s["min"]) / (s["range"] + EPS)
        elif self.method == "robust":
            out = (value - s["median"]) / (s["iqr"] + EPS)
        else:
            out = value

        if self.clip is not None:
            lo, hi = self.clip
            out = float(np.clip(out, lo, hi))

        return float(out)

    # -----------------------------------------------------

    def transform(
        self,
        features: Sequence[Dict[str, float]],
        return_array: bool = False,
    ) -> Union[List[Dict[str, float]], np.ndarray]:

        if not self.fitted_:
            raise RuntimeError(
                "FeatureScalingPipeline.transform: scaler is not fitted. "
                "Call fit(features) first or load() a persisted scaler."
            )

        if not features:
            if return_array:
                return np.zeros((0, len(self.feature_names_)), dtype=np.float32)
            return []

        if return_array:
            n = len(features)
            mat = np.zeros((n, len(self.feature_names_)), dtype=np.float32)
            for i, fd in enumerate(features):
                for j, key in enumerate(self.feature_names_):
                    mat[i, j] = self._scale_value(key, fd.get(key, 0.0))
            return mat

        out: List[Dict[str, float]] = []
        for fd in features:
            scaled = {k: self._scale_value(k, v) for k, v in fd.items()}
            out.append(scaled)
        return out

    # -----------------------------------------------------

    def fit_transform(
        self,
        features: Sequence[Dict[str, float]],
        return_array: bool = False,
    ) -> Union[List[Dict[str, float]], np.ndarray]:
        self.fit(features)
        return self.transform(features, return_array=return_array)

    # =====================================================
    # PERSISTENCE
    # =====================================================

    def save(self, path: str) -> None:
        if not self.fitted_:
            raise RuntimeError("Cannot save unfitted FeatureScalingPipeline")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "method": self.method,
            "clip": list(self.clip) if self.clip is not None else None,
            "feature_names": self.feature_names_,
            "stats": self.stats_,
            "version": 1,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("FeatureScalingPipeline saved | path=%s features=%d",
                    path, len(self.feature_names_))

    # -----------------------------------------------------

    @classmethod
    def load(cls, path: str) -> "FeatureScalingPipeline":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        scaler = cls(
            method=payload.get("method", "standard"),
            clip=tuple(payload["clip"]) if payload.get("clip") else None,
        )
        scaler.feature_names_ = list(payload.get("feature_names", []))
        scaler.stats_ = dict(payload.get("stats", {}))
        scaler.fitted_ = bool(scaler.feature_names_ and scaler.stats_)
        if not scaler.fitted_:
            raise RuntimeError(f"FeatureScalingPipeline.load: invalid file {path}")
        logger.info("FeatureScalingPipeline loaded | path=%s features=%d",
                    path, len(scaler.feature_names_))
        return scaler

    # =====================================================
    # INTROSPECTION
    # =====================================================

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        return dict(self.stats_)

    def __repr__(self) -> str:
        n = len(self.feature_names_) if self.fitted_ else 0
        return f"FeatureScalingPipeline(method={self.method!r}, fitted={self.fitted_}, features={n})"


# Backward-compat alias.
FeatureScaler = FeatureScalingPipeline


# =========================================================
# PER-SECTION SCALER  (§10.4 multi-task scaling fix)
# =========================================================

class FeatureSectionScaler:
    """Maintains one :class:`FeatureScalingPipeline` per feature section.

    §10.4 — A single scaler fitted across all features mis-scales sections
    whose value ranges differ by orders of magnitude (e.g. raw log-magnitude
    token counts vs. normalised [0, 1] lexicon ratios).  This wrapper splits
    each sample dict by section using :func:`partition_feature_sections`,
    fits an independent scaler per section, and reassembles the scaled dicts
    before returning.  Unknown keys (section "other") fall through to a
    single catch-all scaler.

    Parameters
    ----------
    method : str
        Scaling method forwarded to every :class:`FeatureScalingPipeline`
        (``"standard"``, ``"minmax"``, or ``"robust"``).
    clip : tuple or None
        Optional ``(low, high)`` clip forwarded to every child scaler.
    """

    def __init__(
        self,
        method: str = "standard",
        clip: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.method = method
        self.clip = clip
        self._scalers: Dict[str, "FeatureScalingPipeline"] = {}
        self.fitted_: bool = False

    # ------------------------------------------------------------------

    def _split_by_section(
        self, features: Sequence[Dict[str, float]]
    ) -> Dict[str, List[Dict[str, float]]]:
        """Return ``{section: [per-sample dict, ...]}`` for all samples."""
        # Import here to avoid circular import at module load time.
        from src.features.pipelines.feature_pipeline import partition_feature_sections

        section_samples: Dict[str, List[Dict[str, float]]] = {}
        for fd in features:
            sections = partition_feature_sections(fd)
            for sec, sec_dict in sections.items():
                section_samples.setdefault(sec, []).append(sec_dict)
        return section_samples

    # ------------------------------------------------------------------

    def fit(
        self, features: Sequence[Dict[str, float]]
    ) -> "FeatureSectionScaler":
        if not features:
            raise ValueError("FeatureSectionScaler.fit: empty feature list")

        section_samples = self._split_by_section(features)
        self._scalers = {}

        for sec, samples in section_samples.items():
            # Skip sections where every sample dict is empty (no features
            # for that section in this dataset).
            if not any(samples):
                continue
            scaler = FeatureScalingPipeline(method=self.method, clip=self.clip)
            try:
                scaler.fit(samples)
                self._scalers[sec] = scaler
            except Exception as exc:
                logger.warning(
                    "FeatureSectionScaler: skipping section %r — fit failed: %s",
                    sec, exc,
                )

        self.fitted_ = True
        logger.info(
            "FeatureSectionScaler fitted | method=%s sections=%d samples=%d",
            self.method, len(self._scalers), len(features),
        )
        return self

    # ------------------------------------------------------------------

    def transform(
        self, features: Sequence[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        if not self.fitted_:
            raise RuntimeError(
                "FeatureSectionScaler.transform: call fit() first."
            )

        from src.features.pipelines.feature_pipeline import partition_feature_sections

        out: List[Dict[str, float]] = []
        for fd in features:
            sections = partition_feature_sections(fd)
            merged: Dict[str, float] = {}
            for sec, sec_dict in sections.items():
                if not sec_dict:
                    continue
                scaler = self._scalers.get(sec)
                if scaler is None:
                    # Section unseen at fit time — pass through unscaled.
                    merged.update(sec_dict)
                else:
                    scaled_list = scaler.transform([sec_dict])
                    if scaled_list:
                        merged.update(scaled_list[0])
            out.append(merged)
        return out

    # ------------------------------------------------------------------

    def fit_transform(
        self, features: Sequence[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        self.fit(features)
        return self.transform(features)

    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        if not self.fitted_:
            raise RuntimeError("Cannot save unfitted FeatureSectionScaler")

        import json as _json

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "method": self.method,
            "clip": list(self.clip) if self.clip is not None else None,
            "sections": {},
        }
        for sec, scaler in self._scalers.items():
            payload["sections"][sec] = {
                "feature_names": scaler.feature_names_,
                "stats": scaler.stats_,
            }
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(payload, f, indent=2)
        logger.info(
            "FeatureSectionScaler saved | path=%s sections=%d",
            path, len(self._scalers),
        )

    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str) -> "FeatureSectionScaler":
        import json as _json

        with open(path, "r", encoding="utf-8") as f:
            payload = _json.load(f)

        obj = cls(
            method=payload.get("method", "standard"),
            clip=tuple(payload["clip"]) if payload.get("clip") else None,
        )
        for sec, sec_data in payload.get("sections", {}).items():
            scaler = FeatureScalingPipeline(method=obj.method, clip=obj.clip)
            scaler.feature_names_ = list(sec_data.get("feature_names", []))
            scaler.stats_ = dict(sec_data.get("stats", {}))
            scaler.fitted_ = bool(scaler.feature_names_ and scaler.stats_)
            if scaler.fitted_:
                obj._scalers[sec] = scaler

        obj.fitted_ = bool(obj._scalers)
        if not obj.fitted_:
            raise RuntimeError(
                f"FeatureSectionScaler.load: no valid sections in {path}"
            )
        logger.info(
            "FeatureSectionScaler loaded | path=%s sections=%d",
            path, len(obj._scalers),
        )
        return obj

    def __repr__(self) -> str:
        n = len(self._scalers) if self.fitted_ else 0
        return (
            f"FeatureSectionScaler(method={self.method!r}, "
            f"fitted={self.fitted_}, sections={n})"
        )


# =========================================================
# DEPRECATION SHIM  —  HybridTruthLensModel moved to
#   src.models.architectures.hybrid_truthlens_model
# =========================================================

def __getattr__(name: str) -> Any:
    """PEP 562 module-level ``__getattr__`` so that

        from src.features.fusion.feature_scaling import HybridTruthLensModel

    keeps working with a one-shot ``DeprecationWarning``. New code MUST
    import from :mod:`src.models.architectures` instead.
    """
    if name == "HybridTruthLensModel":
        warnings.warn(
            "HybridTruthLensModel has moved to "
            "src.models.architectures.hybrid_truthlens_model; importing it "
            "from src.features.fusion.feature_scaling is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        from src.models.architectures.hybrid_truthlens_model import (
            HybridTruthLensModel as _Hybrid,
        )
        return _Hybrid
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
