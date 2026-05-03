from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.feature_context import FeatureContext
from src.analysis.feature_schema import (
    PROPAGANDA_PATTERN_KEYS,
    make_vector,
    validate_features,
)
from src.analysis._text_features import safe_normalized_entropy

logger = logging.getLogger(__name__)

EPS = 1e-8
MAX_CLIP = 1.0


# =========================================================
# CONFIG
# =========================================================

@dataclass(slots=True)
class PropagandaPatternConfig:

    fear_weight_emotion: float = 0.35
    fear_weight_rhetoric: float = 0.35
    fear_weight_narrative: float = 0.30

    scapegoat_weight_rhetoric: float = 0.55
    scapegoat_weight_argument: float = 0.45

    polarization_weight_narrative: float = 0.60
    polarization_weight_rhetoric: float = 0.40

    emotion_amplification_weight: float = 0.6
    rhetoric_amplification_weight: float = 0.4

    narrative_claim_weight: float = 0.5
    narrative_evidence_weight: float = 0.5

    clip_outputs: bool = True
    clip_range: tuple[float, float] = (0.0, 1.0)

    enable_validation: bool = True
    enable_debug_metadata: bool = False


# =========================================================
# DETECTOR
# =========================================================

class PropagandaPatternDetector(BaseAnalyzer):
    """
    Aggregates upstream analyzer outputs into a small set of propaganda
    pattern intensity signals. Each per-pattern score is an independent
    bounded intensity in [0, 1] — they are NOT renormalized into a
    probability distribution because that would distort their absolute
    magnitudes (a text strong on every pattern would get the same
    distribution as one weak on every pattern). A separate
    `propaganda_diversity` entropy score captures how evenly the
    patterns are spread.

    Inherits from :class:`BaseAnalyzer` so it picks up validation,
    fallback, and caching for free, but it does NOT use the standard
    ``analyze(ctx)`` contract: it consumes upstream feature dicts
    instead of a FeatureContext. The orchestrator invokes
    :meth:`analyze` directly with keyword arguments.
    """

    name = "propaganda_pattern"
    expected_keys = set(PROPAGANDA_PATTERN_KEYS)
    use_cache = False  # inputs are upstream features, not the ctx itself

    def __init__(self, config: PropagandaPatternConfig | None = None):
        self.config = config or PropagandaPatternConfig()

    # --------------------------------------------------------
    # Detector-specific public API.
    # --------------------------------------------------------

    def analyze(  # type: ignore[override]
        self,
        ctx: FeatureContext | None = None,
        *,
        emotion_features: Dict[str, float] | None = None,
        narrative_features: Dict[str, float] | None = None,
        rhetorical_features: Dict[str, float] | None = None,
        argument_features: Dict[str, float] | None = None,
        information_features: Dict[str, float] | None = None,
    ) -> Dict[str, float]:
        # `ctx` is accepted for interface compatibility with BaseAnalyzer
        # but is ignored — this detector composes upstream features.
        del ctx

        emotion = emotion_features or {}
        narrative = narrative_features or {}
        rhetoric = rhetorical_features or {}
        argument = argument_features or {}
        info = information_features or {}

        # -------------------------
        # CORE SIGNALS (independent intensities in [0, 1])
        # -------------------------
        raw = {
            "fear": self._fear(emotion, narrative, rhetoric),
            "scapegoating": self._scapegoating(rhetoric, argument),
            "polarization": self._polarization(narrative, rhetoric),
            "amplification": self._amplification(emotion, rhetoric),
            "imbalance": self._imbalance(argument, info),
        }

        # -------------------------
        # DERIVED METRICS
        # -------------------------
        # Intensity: bounded mean of raw signals — preserves the absolute
        # magnitude of how strong propaganda cues are overall.
        intensity = sum(raw.values()) / max(len(raw), 1)

        # Diversity: normalized entropy of the signal distribution.
        diversity = self._entropy(raw)

        features = {
            "fear_propaganda_score": self._safe(raw["fear"]),
            "scapegoating_score": self._safe(raw["scapegoating"]),
            "polarization_score": self._safe(raw["polarization"]),
            "emotional_amplification_score": self._safe(raw["amplification"]),
            "narrative_imbalance_score": self._safe(raw["imbalance"]),
            "propaganda_intensity": self._safe(intensity),
            "propaganda_diversity": self._safe(diversity),
        }

        # -------------------------
        # CLIP
        # -------------------------
        if self.config.clip_outputs:
            features = self._clip(features)

        # -------------------------
        # VALIDATION
        # -------------------------
        if self.config.enable_validation:
            validate_features(features, PROPAGANDA_PATTERN_KEYS)

        return features

    # =========================================================
    # SIGNALS (CONFIG-DRIVEN)
    # =========================================================

    def _fear(
        self,
        emotion: Dict[str, float],
        narrative: Dict[str, float],
        rhetoric: Dict[str, float],
    ) -> float:

        e = self._get(emotion, "emotion_expression_ratio")
        r = self._get(rhetoric, "rhetoric_fear_appeal_score")

        # F10: previously `_get(*keys)` returned the FIRST present
        # value. Both narrative cues are commonly emitted at once, so
        # we lost half the signal whenever `conflict_intensity` was
        # present. Use the mean of all present cues instead.
        n = self._mean_present(
            narrative, "conflict_intensity", "polarization_ratio"
        )

        return (
            e * self.config.fear_weight_emotion
            + r * self.config.fear_weight_rhetoric
            + n * self.config.fear_weight_narrative
        )

    def _scapegoating(
        self,
        rhetoric: Dict[str, float],
        argument: Dict[str, float],
    ) -> float:

        r = self._get(rhetoric, "rhetoric_scapegoating_score")
        a = self._get(argument, "argument_contrast_ratio")

        return (
            r * self.config.scapegoat_weight_rhetoric
            + a * self.config.scapegoat_weight_argument
        )

    def _polarization(
        self,
        narrative: Dict[str, float],
        rhetoric: Dict[str, float],
    ) -> float:

        n = self._get(narrative, "polarization_ratio")
        r = self._get(rhetoric, "rhetoric_loaded_language_score")

        return (
            n * self.config.polarization_weight_narrative
            + r * self.config.polarization_weight_rhetoric
        )

    def _amplification(
        self,
        emotion: Dict[str, float],
        rhetoric: Dict[str, float],
    ) -> float:

        vals = [
            self._get(emotion, "emotion_expression_ratio"),
            self._get(emotion, "dominant_emotion_strength"),
        ]

        e = sum(vals) / max(len(vals), 1)
        r = self._get(rhetoric, "rhetoric_emotional_appeal_score")

        return (
            e * self.config.emotion_amplification_weight
            + r * self.config.rhetoric_amplification_weight
        )

    def _imbalance(
        self,
        argument: Dict[str, float],
        info: Dict[str, float],
    ) -> float:

        claim = self._get(argument, "argument_claim_ratio")
        evidence = self._get(info, "factual_density")

        # Bounded ratio in [0, 1]: 1 means all claim and no evidence.
        denom = claim + evidence
        if denom <= EPS:
            return 0.0
        return claim / (denom + EPS)

    # =========================================================
    # UTILS
    # =========================================================

    def _get(
        self,
        features: Dict[str, Any],
        *keys: str,
        default: float = 0.0,
    ) -> float:

        for k in keys:
            v = features.get(k)
            if isinstance(v, (int, float)) and np.isfinite(v):
                return float(v)

        return default

    def _mean_present(
        self,
        features: Dict[str, Any],
        *keys: str,
        default: float = 0.0,
    ) -> float:
        """F10: mean over keys that are actually present and finite.

        Returns ``default`` when no requested key is present. Use this
        when several upstream features express the same underlying
        signal and we want to combine, not pick, them.
        """
        vals = []
        for k in keys:
            v = features.get(k)
            if isinstance(v, (int, float)) and np.isfinite(v):
                vals.append(float(v))
        if not vals:
            return default
        return sum(vals) / len(vals)

    def _entropy(self, dist: Dict[str, float]) -> float:

        # NUM-A1: this detector's previously-local guarded entropy is now
        # the shared `safe_normalized_entropy` helper. Behavior is
        # identical (same n<=1 / max_entropy<EPS / sum<EPS guards).
        return safe_normalized_entropy(dist.values())

    def _clip(self, features: Dict[str, float]) -> Dict[str, float]:

        low, high = self.config.clip_range

        return {k: float(np.clip(v, low, high)) for k, v in features.items()}

    def _safe(self, v: float) -> float:
        if not isinstance(v, (int, float)) or not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))


# =========================================================
# VECTOR
# =========================================================

def propaganda_pattern_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, PROPAGANDA_PATTERN_KEYS)
