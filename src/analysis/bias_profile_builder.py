from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from src.analysis.feature_schema import get_schema

logger = logging.getLogger(__name__)


# =========================================================
# CONSTANTS
# =========================================================

EPS = 1e-9


# =========================================================
# CONFIG
# =========================================================

@dataclass(slots=True)
class BiasProfileConfig:

    # Section weights
    bias_weight: float = 1.0
    emotion_weight: float = 1.0
    narrative_weight: float = 1.0
    discourse_weight: float = 1.0
    argument_weight: float = 0.6
    ideology_weight: float = 0.6

    # Normalization
    normalize_values: bool = True
    normalization_method: str = "minmax"  # minmax | zscore | robust

    clip_values: bool = True
    clip_range: tuple[float, float] = (0.0, 1.0)

    # Advanced
    global_normalization: bool = True
    apply_softmax_to_ideology: bool = True

    # Score aggregation
    aggregation_method: str = "mean"  # mean | weighted_mean


# =========================================================
# BUILDER
# =========================================================

class BiasProfileBuilder:

    PROFILE_SECTIONS = (
        "bias",
        "emotion",
        "narrative",
        "discourse",
        "argument",
        "ideology",
    )

    # Feature-pipeline key prefixes routed by `build_from_feature_dict`.
    # Keep these in sync with the emit prefixes used by the extractors in
    # `src/features/<group>/`.
    _FEATURE_PREFIX_TO_SECTION = {
        "disc_": "discourse",
        "arg_": "argument",
        "emotion_": "emotion",
        "bias_": "bias",
        "narrative_": "narrative",
        "framing_": "bias",
        "ideology_": "ideology",
    }

    def __init__(self, config: BiasProfileConfig | None = None):
        self.config = config or BiasProfileConfig()
        logger.info("BiasProfileBuilder initialized")

    # =====================================================
    # FEATURE-PIPELINE ENTRY (audit task 2)
    # =====================================================

    def build_from_feature_dict(
        self,
        features: Dict[str, float],
        ideology: Dict[str, float] | None = None,
    ) -> Dict[str, Any]:
        """Build a profile from a flat feature dict (FeaturePipeline output).

        Routes prefixed keys to their section using
        :attr:`_FEATURE_PREFIX_TO_SECTION`. Unprefixed keys are dropped.
        Provides the missing wiring between the feature-engineering layer
        (``src/features/<group>/*``) — including ``DiscourseFeatures``
        (``disc_*``) and ``ArgumentStructureFeatures`` (``arg_*``) — and
        the bias profile.

        Parameters
        ----------
        features : dict
            Flat ``{feature_name: float}`` map (typically the output of
            ``FeaturePipeline.extract(ctx)`` or one row of
            ``BatchFeaturePipeline``).
        ideology : dict, optional
            Ideology distribution (already a probability vector). Passed
            through unchanged because it is produced by an analysis-side
            classifier rather than by a prefixed feature extractor.
        """
        sections: Dict[str, Dict[str, float]] = {
            s: {} for s in self.PROFILE_SECTIONS
        }
        for k, v in (features or {}).items():
            for prefix, section in self._FEATURE_PREFIX_TO_SECTION.items():
                if k.startswith(prefix):
                    sections[section][k] = v
                    break

        return self.build_profile(
            bias=sections["bias"],
            emotion=sections["emotion"],
            narrative=sections["narrative"],
            discourse=sections["discourse"],
            argument=sections["argument"],
            ideology=ideology if ideology is not None else sections["ideology"],
        )

    # =====================================================
    # MAIN ENTRY
    # =====================================================

    def build_profile(
        self,
        *,
        bias: Dict[str, float],
        emotion: Dict[str, float],
        narrative: Dict[str, float],
        discourse: Dict[str, float],
        ideology: Dict[str, float],
        argument: Dict[str, float] | None = None,
    ) -> Dict[str, Any]:

        profile = {
            "metadata": {
                "created_at": int(time.time()),
                "sections": list(self.PROFILE_SECTIONS),
            }
        }

        # ---- Process each section ----
        # `argument` is keyword-default-None for backward-compat with callers
        # that predate the discourse/argument feature wiring (audit task 2).
        for section_name, data in {
            "bias": bias,
            "emotion": emotion,
            "narrative": narrative,
            "discourse": discourse,
            "argument": argument or {},
            "ideology": ideology,
        }.items():
            profile[section_name] = self._process_section(data)

        # ---- Ideology calibration ----
        if self.config.apply_softmax_to_ideology:
            profile["ideology"] = self._softmax(profile["ideology"])

        # ---- Global normalization ----
        if self.config.global_normalization:
            profile = self._global_normalize(profile)

        # ---- Metrics ----
        profile["metrics"] = self._compute_metrics(profile)

        # ---- Final score ----
        profile["bias_score"] = self._compute_bias_score(profile)

        return profile

    # =====================================================
    # SECTION PROCESSING
    # =====================================================

    def _process_section(self, data: Dict[str, Any]) -> Dict[str, float]:

        data = self._sanitize(data)

        if self.config.normalize_values:
            data = self._normalize(data)

        if self.config.clip_values:
            data = self._clip(data)

        return data

    # =====================================================
    # SANITIZATION
    # =====================================================

    def _sanitize(self, data: Dict[str, Any]) -> Dict[str, float]:

        cleaned = {}

        for k, v in data.items():
            try:
                v = float(v)
                if not np.isfinite(v):
                    v = 0.0
            except Exception:
                v = 0.0

            cleaned[k] = v

        return cleaned

    # =====================================================
    # NORMALIZATION
    # =====================================================

    def _normalize(self, data: Dict[str, float]) -> Dict[str, float]:

        if not data:
            return data

        values = np.array(list(data.values()), dtype=np.float32)

        if self.config.normalization_method == "zscore":
            mean, std = values.mean(), values.std()
            if std < EPS:
                return data
            # NUM-A4: previously emitted raw z-scores in (-inf, +inf),
            # which were then floored to 0 by the downstream [0, 1] clip
            # — making the lower half of the distribution silently
            # collapse to zero. Map z-scores monotonically into (0, 1)
            # via tanh so both tails are preserved before clipping.
            z = (values - mean) / (std + EPS)
            norm = 0.5 * (np.tanh(z) + 1.0)

        elif self.config.normalization_method == "robust":
            median = np.median(values)
            iqr = np.percentile(values, 75) - np.percentile(values, 25)
            if iqr < EPS:
                return data
            # NUM-A3: same fix as zscore — previously this produced
            # negative recentered scores that got floored to 0 by the
            # [0, 1] clip. Squash with tanh into (0, 1) explicitly.
            r = (values - median) / (iqr + EPS)
            norm = 0.5 * (np.tanh(r) + 1.0)

        else:  # minmax
            min_v, max_v = values.min(), values.max()
            if max_v - min_v < EPS:
                return data
            norm = (values - min_v) / (max_v - min_v + EPS)

        return dict(zip(data.keys(), norm.astype(float)))

    # =====================================================
    # GLOBAL NORMALIZATION (FIXED)
    # =====================================================

    def _global_normalize(self, profile: Dict[str, Any]) -> Dict[str, Any]:

        # PERF-A6: collect the section value arrays once, concatenate,
        # then derive global min/max in a single pass instead of
        # re-walking each section dict. The per-section dicts are then
        # rewritten via the existing min/scale.
        section_values: Dict[str, np.ndarray] = {}

        for section in self.PROFILE_SECTIONS:
            section_dict = profile.get(section, {})
            if section_dict:
                section_values[section] = np.fromiter(
                    section_dict.values(), dtype=np.float32, count=len(section_dict)
                )

        if not section_values:
            return profile

        arr = np.concatenate(list(section_values.values()))
        min_v, max_v = arr.min(), arr.max()

        if max_v - min_v < EPS:
            return profile

        scale = max_v - min_v + EPS

        for section in self.PROFILE_SECTIONS:
            section_dict = profile.get(section, {})
            if not section_dict:
                continue
            profile[section] = {
                k: float((v - min_v) / scale)
                for k, v in section_dict.items()
            }

        return profile

    # =====================================================
    # SOFTMAX (STABLE)
    # =====================================================

    def _softmax(self, data: Dict[str, float]) -> Dict[str, float]:

        if not data:
            return data

        values = np.array(list(data.values()), dtype=np.float32)

        values = values - np.max(values)  # stability
        exp = np.exp(values)

        # NUM-A2: drop the `+ EPS` in the denominator. After the
        # max-subtraction trick `exp` always contains at least one
        # element equal to 1.0, so `exp.sum() >= 1.0 > 0` and the EPS
        # only introduced an unnecessary downward bias on the
        # probabilities (and broke the property that the result sums to
        # exactly 1.0).
        probs = exp / exp.sum()

        return dict(zip(data.keys(), probs.astype(float)))

    # =====================================================
    # CLIPPING
    # =====================================================

    def _clip(self, data: Dict[str, float]) -> Dict[str, float]:

        low, high = self.config.clip_range

        return {
            k: float(np.clip(v, low, high))
            for k, v in data.items()
        }

    # =====================================================
    # METRICS (IMPROVED)
    # =====================================================

    def _compute_metrics(self, profile: Dict[str, Any]) -> Dict[str, float]:

        ideology_vals = np.array(
            list(profile.get("ideology", {}).values()), dtype=np.float32
        )

        if ideology_vals.size > 0:
            p = ideology_vals / (ideology_vals.sum() + EPS)
            entropy = float(-np.sum(p * np.log(p + EPS)))
            dominance = float(np.max(p))
        else:
            entropy = dominance = 0.0

        bias_vals = np.array(
            list(profile.get("bias", {}).values()), dtype=np.float32
        )

        variance = float(np.var(bias_vals)) if bias_vals.size else 0.0

        return {
            "bias_variance": variance,
            "ideology_entropy": entropy,
            "ideology_dominance": dominance,
        }

    # =====================================================
    # FINAL SCORE (FIXED WEIGHTING)
    # =====================================================

    def _compute_bias_score(self, profile: Dict[str, Any]) -> float:

        weights = {
            "bias": self.config.bias_weight,
            "emotion": self.config.emotion_weight,
            "narrative": self.config.narrative_weight,
            "discourse": self.config.discourse_weight,
            "argument": self.config.argument_weight,
            "ideology": self.config.ideology_weight,
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for section, weight in weights.items():
            values = list(profile.get(section, {}).values())

            if not values:
                continue

            section_mean = float(np.mean(values))

            weighted_sum += section_mean * weight
            total_weight += weight

        if total_weight <= EPS:
            return 0.0

        score = weighted_sum / (total_weight + EPS)

        return float(np.clip(score, 0.0, 1.0))


# =========================================================
# VECTORIZATION (STABLE)
# =========================================================

def bias_profile_vector(profile: Dict[str, Any]) -> np.ndarray:
    # Section 7: this helper has no in-tree callers (only a stale
    # comment reference in feature_schema.py). It also encodes a
    # questionable mapping (`narrative` → "framing" schema) that is
    # almost certainly a bug for any external caller. Kept to avoid
    # breaking any downstream import, but emit a DeprecationWarning so
    # the few-or-zero remaining callers surface themselves. Prefer
    # `FeatureMerger.to_vector` or `FullAnalysisOutput.to_vector` for
    # serializing analysis output.
    import warnings
    warnings.warn(
        "bias_profile_vector is deprecated and slated for removal; "
        "use FullAnalysisOutput.to_vector or FeatureMerger.to_vector "
        "instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    sections = {
        "bias": "framing",
        "emotion": "emotion_target",
        "narrative": "framing",
        "discourse": "discourse_coherence",
        "ideology": "ideology",
    }

    values: List[float] = []

    for section, schema_name in sections.items():
        keys = get_schema(schema_name)
        data = profile.get(section, {})

        for k in keys:
            values.append(float(data.get(k, 0.0)))

    if not values:
        raise ValueError("profile contains no values")

    return np.array(values, dtype=np.float32)