from __future__ import annotations

import logging
from typing import Dict, Set

import numpy as np

from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.feature_context import FeatureContext
from src.analysis._text_features import (
    term_ratio,
    phrase_match_count,
    normalize_lexicon_terms,
)
from src.analysis.feature_schema import FRAMING_KEYS, make_vector

logger = logging.getLogger(__name__)


# =========================================================
# CONSTANTS
# =========================================================

EPS = 1e-8
MAX_CLIP = 1.0


# =========================================================
# ANALYZER
# =========================================================

class FramingAnalyzer(BaseAnalyzer):

    name = "framing"
    expected_keys = set(FRAMING_KEYS)

    # -----------------------------------------------------
    # LEXICONS (KEEP YOUR FULL LISTS HERE)
    # -----------------------------------------------------

    CONFLICT_TERMS: Set[str] = {...}
    ECONOMIC_TERMS: Set[str] = {...}
    MORAL_TERMS: Set[str] = {...}
    HUMAN_INTEREST_TERMS: Set[str] = {...}
    SECURITY_TERMS: Set[str] = {...}

    BASE_KEYS = [
        "frame_conflict_score",
        "frame_economic_score",
        "frame_moral_score",
        "frame_human_interest_score",
        "frame_security_score",
    ]

    # =========================================================

    def __init__(self):

        self.conflict = normalize_lexicon_terms(self.CONFLICT_TERMS)
        self.economic = normalize_lexicon_terms(self.ECONOMIC_TERMS)
        self.moral = normalize_lexicon_terms(self.MORAL_TERMS)
        self.human = normalize_lexicon_terms(self.HUMAN_INTEREST_TERMS)
        self.security = normalize_lexicon_terms(self.SECURITY_TERMS)

        logger.info("FramingAnalyzer initialized (final)")

    # =========================================================

    def analyze(self, ctx: FeatureContext) -> Dict[str, float]:

        # 🔥 ensure lazy features are ready
        ctx.ensure_tokens()

        if ctx.safe_n_tokens() == 0:
            return self._empty_features()

        scores = {
            "frame_conflict_score": self._score(ctx, self.conflict),
            "frame_economic_score": self._score(ctx, self.economic),
            "frame_moral_score": self._score(ctx, self.moral),
            "frame_human_interest_score": self._score(ctx, self.human),
            "frame_security_score": self._score(ctx, self.security),
        }

        # -----------------------------------------------------
        # RELATIVE NORMALIZATION (CRITICAL)
        # -----------------------------------------------------

        scores = self._normalize(scores)

        # -----------------------------------------------------
        # HIGH-LEVEL FEATURES
        # -----------------------------------------------------

        scores.update(self._frame_dominance(scores))
        scores.update(self._frame_diversity(scores))

        return scores

    # =========================================================

    def _score(self, ctx: FeatureContext, lexicon: Set[str]) -> float:

        # token-level signal
        token_score = term_ratio(
            ctx.safe_counts(),
            ctx.safe_n_tokens(),
            lexicon,
        )

        # phrase-level signal
        phrase_hits = phrase_match_count(
            ctx.text_lower or "",
            lexicon,
        )

        phrase_score = phrase_hits / (ctx.safe_n_tokens() + EPS)

        # 🔥 weighted fusion (prevents double counting)
        combined = 0.7 * token_score + 0.3 * phrase_score

        return self._safe(combined)

    # =========================================================

    def _normalize(self, scores: Dict[str, float]) -> Dict[str, float]:

        values = np.array(list(scores.values()), dtype=np.float32)

        total = float(values.sum())

        if total < EPS:
            return {k: 0.0 for k in scores}

        normalized = values / (total + EPS)

        return dict(zip(scores.keys(), normalized.astype(float)))

    # =========================================================

    def _frame_dominance(self, scores: Dict[str, float]) -> Dict[str, float]:

        if not scores:
            return {"frame_dominance_score": 0.0}

        return {
            "frame_dominance_score": self._safe(max(scores.values()))
        }

    # =========================================================

    def _frame_diversity(self, scores: Dict[str, float]) -> Dict[str, float]:

        values = np.array(list(scores.values()), dtype=np.float32)

        if values.sum() < EPS:
            return {"frame_diversity_score": 0.0}

        probs = values / (values.sum() + EPS)

        # entropy-based diversity (robust)
        entropy = -np.sum(probs * np.log(probs + EPS))

        max_entropy = np.log(len(values))
        diversity = entropy / (max_entropy + EPS)

        return {
            "frame_diversity_score": self._safe(diversity)
        }

    # =========================================================

    def _safe(self, value: float) -> float:

        if not np.isfinite(value):
            return 0.0

        return float(np.clip(value, 0.0, MAX_CLIP))

    # =========================================================

    def _empty_features(self) -> Dict[str, float]:

        return {
            "frame_conflict_score": 0.0,
            "frame_economic_score": 0.0,
            "frame_moral_score": 0.0,
            "frame_human_interest_score": 0.0,
            "frame_security_score": 0.0,
            "frame_dominance_score": 0.0,
            "frame_diversity_score": 0.0,
        }


# =========================================================
# VECTOR CONVERSION
# =========================================================

def framing_feature_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, FRAMING_KEYS)