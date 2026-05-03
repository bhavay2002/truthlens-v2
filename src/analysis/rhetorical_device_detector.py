from __future__ import annotations

import logging
import re
from typing import Dict, Set

import numpy as np

from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.feature_context import FeatureContext
from src.analysis._text_features import (
    term_ratio,
    phrase_match_count,
    normalize_lexicon_terms,
)
from src.analysis.feature_schema import RHETORICAL_DEVICE_KEYS, make_vector

logger = logging.getLogger(__name__)

EPS = 1e-8
MAX_CLIP = 1.0


class RhetoricalDeviceDetector(BaseAnalyzer):

    name = "rhetorical_devices"
    expected_keys = set(RHETORICAL_DEVICE_KEYS)

    # -----------------------------------------------------

    EXAGGERATION_TERMS = {...}
    LOADED_LANGUAGE_TERMS = {...}
    EMOTIONAL_APPEAL_TERMS = {...}
    FEAR_APPEAL_TERMS = {...}
    INTENSIFIERS = {...}

    SCAPEGOAT_PATTERNS = {...}
    FALSE_DILEMMA_PATTERNS = {...}

    RHETORICAL_PUNCT_PATTERN = re.compile(r"[!?]+")

    # =========================================================

    def __init__(self):

        self.exaggeration = normalize_lexicon_terms(self.EXAGGERATION_TERMS)
        self.loaded = normalize_lexicon_terms(self.LOADED_LANGUAGE_TERMS)
        self.emotional = normalize_lexicon_terms(self.EMOTIONAL_APPEAL_TERMS)
        self.fear = normalize_lexicon_terms(self.FEAR_APPEAL_TERMS)
        self.intensifiers = normalize_lexicon_terms(self.INTENSIFIERS)

        self.scapegoat_patterns = normalize_lexicon_terms(self.SCAPEGOAT_PATTERNS)
        self.false_dilemma_patterns = normalize_lexicon_terms(self.FALSE_DILEMMA_PATTERNS)

        logger.info("RhetoricalDeviceDetector initialized (final)")

    # =========================================================

    def analyze(self, ctx: FeatureContext) -> Dict[str, float]:

        # 🔥 lazy-safe
        ctx.ensure_tokens()

        if ctx.safe_n_tokens() == 0:
            return self._empty()

        n_tokens = ctx.safe_n_tokens()

        # -----------------------------------------------------
        # RAW SCORES
        # -----------------------------------------------------

        raw = {
            "exaggeration": self._score(ctx, self.exaggeration),
            "loaded": self._score(ctx, self.loaded),
            "emotional": self._score(ctx, self.emotional),
            "fear": self._score(ctx, self.fear),
            "intensifier": self._score(ctx, self.intensifiers),
            "scapegoat": self._pattern(ctx, self.scapegoat_patterns),
            "false_dilemma": self._pattern(ctx, self.false_dilemma_patterns),
        }

        # -----------------------------------------------------
        # NORMALIZATION
        # -----------------------------------------------------

        dist = self._normalize(raw)

        # -----------------------------------------------------
        # GLOBAL INTENSITY
        # -----------------------------------------------------

        intensity = sum(raw.values()) / (len(raw) + EPS)

        # -----------------------------------------------------
        # DIVERSITY
        # -----------------------------------------------------

        diversity = self._entropy(dist)

        # -----------------------------------------------------
        # PUNCTUATION (FIXED)
        # -----------------------------------------------------

        punctuation = self._punctuation(ctx)

        # -----------------------------------------------------
        # OUTPUT
        # -----------------------------------------------------

        return {
            "rhetoric_exaggeration_score": self._safe(dist["exaggeration"]),
            "rhetoric_loaded_language_score": self._safe(dist["loaded"]),
            "rhetoric_emotional_appeal_score": self._safe(dist["emotional"]),
            "rhetoric_fear_appeal_score": self._safe(dist["fear"]),
            "rhetoric_intensifier_ratio": self._safe(dist["intensifier"]),
            "rhetoric_scapegoating_score": self._safe(dist["scapegoat"]),
            "rhetoric_false_dilemma_score": self._safe(dist["false_dilemma"]),
            "rhetoric_punctuation_score": self._safe(punctuation),
            "rhetoric_intensity": self._safe(intensity),
            "rhetoric_diversity": self._safe(diversity),
        }

    # =========================================================

    def _score(self, ctx: FeatureContext, lexicon: Set[str]) -> float:

        n_tokens = ctx.safe_n_tokens()

        token_score = term_ratio(
            ctx.safe_counts(),
            n_tokens,
            lexicon,
        )

        phrase_hits = phrase_match_count(
            ctx.text_lower or "",
            lexicon,
        )

        phrase_score = phrase_hits / (n_tokens + EPS)

        # 🔥 weighted fusion (prevents double counting)
        return 0.7 * token_score + 0.3 * phrase_score

    # =========================================================

    def _pattern(self, ctx: FeatureContext, patterns: Set[str]) -> float:

        hits = phrase_match_count(
            ctx.text_lower or "",
            patterns,
        )

        return hits / (ctx.safe_n_tokens() + EPS)

    # =========================================================

    def _punctuation(self, ctx: FeatureContext) -> float:

        text = ctx.text_lower or ""

        matches = self.RHETORICAL_PUNCT_PATTERN.findall(text)

        if not matches:
            return 0.0

        # 🔥 counts intensity correctly (!!! > !)
        score = sum(len(m) for m in matches)

        return score / (ctx.safe_n_tokens() + EPS)

    # =========================================================

    def _normalize(self, scores: Dict[str, float]) -> Dict[str, float]:

        values = np.array(list(scores.values()), dtype=np.float32)

        total = float(values.sum())

        if total < EPS:
            return {k: 0.0 for k in scores}

        norm = values / (total + EPS)

        return dict(zip(scores.keys(), norm.astype(float)))

    # =========================================================

    def _entropy(self, dist: Dict[str, float]) -> float:

        values = np.array(list(dist.values()), dtype=np.float32)

        if values.sum() < EPS:
            return 0.0

        probs = values / (values.sum() + EPS)

        entropy = -np.sum(probs * np.log(probs + EPS))
        max_entropy = np.log(len(probs))

        return float(entropy / (max_entropy + EPS))

    # =========================================================

    def _safe(self, value: float) -> float:
        if not np.isfinite(value):
            return 0.0
        return float(np.clip(value, 0.0, MAX_CLIP))

    # =========================================================

    def _empty(self) -> Dict[str, float]:
        return {
            "rhetoric_exaggeration_score": 0.0,
            "rhetoric_loaded_language_score": 0.0,
            "rhetoric_emotional_appeal_score": 0.0,
            "rhetoric_fear_appeal_score": 0.0,
            "rhetoric_intensifier_ratio": 0.0,
            "rhetoric_scapegoating_score": 0.0,
            "rhetoric_false_dilemma_score": 0.0,
            "rhetoric_punctuation_score": 0.0,
            "rhetoric_intensity": 0.0,
            "rhetoric_diversity": 0.0,
        }


# =========================================================
# VECTOR
# =========================================================

def rhetorical_feature_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, RHETORICAL_DEVICE_KEYS)