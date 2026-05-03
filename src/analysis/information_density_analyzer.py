from __future__ import annotations

import logging
from typing import Dict, Set

import numpy as np

from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.feature_context import FeatureContext
from src.analysis._text_features import (
    term_ratio,
    phrase_match_count,
    cached_phrase_match_count,
    normalize_lexicon_terms,
    safe_normalized_entropy,
)
from src.analysis.feature_schema import INFORMATION_DENSITY_KEYS, make_vector

logger = logging.getLogger(__name__)


# =========================================================
# CONSTANTS
# =========================================================

EPS = 1e-8
MAX_CLIP = 1.0


# =========================================================
# ANALYZER
# =========================================================

class InformationDensityAnalyzer(BaseAnalyzer):

    name = "information_density"
    expected_keys = set(INFORMATION_DENSITY_KEYS)

    # -----------------------------------------------------
    # LEXICONS (KEEP FULL SETS)
    # -----------------------------------------------------

    FACTUAL_TERMS: Set[str] = {...}
    OPINION_TERMS: Set[str] = {...}
    CLAIM_TERMS: Set[str] = {...}
    RHETORICAL_TERMS: Set[str] = {...}
    EMOTIONAL_TERMS: Set[str] = {...}
    MODAL_TERMS: Set[str] = {...}

    # =========================================================

    def __init__(self):

        self.factual = normalize_lexicon_terms(self.FACTUAL_TERMS)
        self.opinion = normalize_lexicon_terms(self.OPINION_TERMS)
        self.claim = normalize_lexicon_terms(self.CLAIM_TERMS)
        self.rhetorical = normalize_lexicon_terms(self.RHETORICAL_TERMS)
        self.emotion = normalize_lexicon_terms(self.EMOTIONAL_TERMS)
        self.modal = normalize_lexicon_terms(self.MODAL_TERMS)

        logger.info("InformationDensityAnalyzer initialized (final)")

    # =========================================================

    def analyze(self, ctx: FeatureContext) -> Dict[str, float]:

        # 🔥 ensure lazy context
        ctx.ensure_tokens()

        if ctx.safe_n_tokens() == 0:
            return self._empty()

        n_tokens = ctx.safe_n_tokens()

        # -----------------------------------------------------
        # RAW DENSITIES
        # -----------------------------------------------------

        raw = {
            "factual": self._density(ctx, self.factual),
            "opinion": self._density(ctx, self.opinion),
            "claim": self._density(ctx, self.claim),
            "rhetorical": self._density(ctx, self.rhetorical),
            "emotion": self._density(ctx, self.emotion),
            "modal": self._density(ctx, self.modal),
        }

        # -----------------------------------------------------
        # NORMALIZE (CRITICAL)
        # -----------------------------------------------------

        dist = self._normalize(raw)

        # -----------------------------------------------------
        # FEATURES
        # -----------------------------------------------------

        features = {
            "factual_density": self._safe(dist["factual"]),
            "opinion_density": self._safe(dist["opinion"]),
            "claim_density": self._safe(dist["claim"]),
            "rhetorical_density": self._safe(dist["rhetorical"]),
            "emotion_density": self._safe(dist["emotion"]),
            "modal_density": self._safe(dist["modal"]),
        }

        # -----------------------------------------------------
        # PUNCTUATION SIGNAL
        # -----------------------------------------------------

        features["rhetorical_punctuation_density"] = self._punctuation(ctx)

        # -----------------------------------------------------
        # INFORMATION VS EMOTION RATIO
        # -----------------------------------------------------

        features.update(self._information_emotion_ratio(dist))

        # -----------------------------------------------------
        # DIVERSITY (ENTROPY)
        # -----------------------------------------------------

        features["information_diversity"] = self._entropy(dist)

        return features

    # =========================================================

    def _density(self, ctx: FeatureContext, lexicon: Set[str]) -> float:

        n_tokens = ctx.safe_n_tokens()

        # token signal
        token_score = term_ratio(
            ctx.safe_counts(),
            n_tokens,
            lexicon,
        )

        # phrase signal — PERF-A2: shared per-ctx phrase-hit cache.
        phrase_hits = cached_phrase_match_count(ctx, lexicon)

        phrase_score = phrase_hits / (n_tokens + EPS)

        # 🔥 weighted fusion (prevents double counting)
        combined = 0.7 * token_score + 0.3 * phrase_score

        return self._safe(combined)

    # =========================================================

    def _normalize(self, scores: Dict[str, float]) -> Dict[str, float]:

        values = np.array(list(scores.values()), dtype=np.float32)

        total = float(values.sum())

        if total < EPS:
            return {k: 0.0 for k in scores}

        norm = values / (total + EPS)

        return dict(zip(scores.keys(), norm.astype(float)))

    # =========================================================

    def _punctuation(self, ctx: FeatureContext) -> float:

        # PERF-A1: shared punctuation-count cache (paid once per ctx).
        count = ctx.punct_count("!") + ctx.punct_count("?")

        return self._safe(count / (ctx.safe_n_tokens() + EPS))

    # =========================================================

    def _information_emotion_ratio(self, dist: Dict[str, float]) -> Dict[str, float]:

        factual = dist["factual"]
        emotion = dist["emotion"]

        ratio = factual / (factual + emotion + EPS)

        return {
            "information_emotion_ratio": self._safe(ratio),
        }

    # =========================================================

    def _entropy(self, dist: Dict[str, float]) -> float:

        # NUM-A1: shared safe normalized entropy helper.
        return self._safe(safe_normalized_entropy(dist.values()))

    # =========================================================

    def _safe(self, value: float) -> float:

        if not np.isfinite(value):
            return 0.0

        return float(np.clip(value, 0.0, MAX_CLIP))

    # =========================================================

    def _empty(self) -> Dict[str, float]:

        return {
            "factual_density": 0.0,
            "opinion_density": 0.0,
            "claim_density": 0.0,
            "rhetorical_density": 0.0,
            "emotion_density": 0.0,
            "modal_density": 0.0,
            "rhetorical_punctuation_density": 0.0,
            "information_emotion_ratio": 0.0,
            "information_diversity": 0.0,
        }


# =========================================================
# VECTOR CONVERSION
# =========================================================

def information_density_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, INFORMATION_DENSITY_KEYS)