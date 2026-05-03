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
from src.analysis.feature_schema import IDEOLOGICAL_LANGUAGE_KEYS, make_vector

logger = logging.getLogger(__name__)


# =========================================================
# CONSTANTS
# =========================================================

EPS = 1e-8
MAX_CLIP = 1.0


# =========================================================
# ANALYZER
# =========================================================

class IdeologicalLanguageDetector(BaseAnalyzer):

    name = "ideological_language"
    expected_keys = set(IDEOLOGICAL_LANGUAGE_KEYS)

    # -----------------------------------------------------
    # LEXICONS (KEEP FULL LISTS)
    # -----------------------------------------------------

    LIBERTY_TERMS: Set[str] = {...}
    EQUALITY_TERMS: Set[str] = {...}
    TRADITION_TERMS: Set[str] = {...}
    ELITE_TERMS: Set[str] = {...}
    IDEOLOGY_PHRASES: Set[str] = {...}

    # =========================================================

    def __init__(self):

        self.liberty = normalize_lexicon_terms(self.LIBERTY_TERMS)
        self.equality = normalize_lexicon_terms(self.EQUALITY_TERMS)
        self.tradition = normalize_lexicon_terms(self.TRADITION_TERMS)
        self.elite = normalize_lexicon_terms(self.ELITE_TERMS)
        self.phrases = normalize_lexicon_terms(self.IDEOLOGY_PHRASES)

        logger.info("IdeologicalLanguageDetector initialized (final)")

    # =========================================================

    def analyze(self, ctx: FeatureContext) -> Dict[str, float]:

        # 🔥 ensure lazy computation
        ctx.ensure_tokens()

        if ctx.safe_n_tokens() == 0:
            return self._empty_features()

        n_tokens = ctx.safe_n_tokens()

        # -----------------------------------------------------
        # TOKEN SIGNALS
        # -----------------------------------------------------

        raw_scores = {
            "liberty": term_ratio(ctx.safe_counts(), n_tokens, self.liberty),
            "equality": term_ratio(ctx.safe_counts(), n_tokens, self.equality),
            "tradition": term_ratio(ctx.safe_counts(), n_tokens, self.tradition),
            "elite": term_ratio(ctx.safe_counts(), n_tokens, self.elite),
        }

        # -----------------------------------------------------
        # PHRASE SIGNAL
        # -----------------------------------------------------

        phrase_hits = phrase_match_count(
            ctx.text_lower or "",
            self.phrases
        )

        phrase_score = phrase_hits / (n_tokens + EPS)

        # 🔥 controlled fusion (prevents overpowering)
        for k in raw_scores:
            raw_scores[k] += 0.2 * phrase_score

        # -----------------------------------------------------
        # NORMALIZATION (CRITICAL)
        # -----------------------------------------------------

        scores = self._normalize(raw_scores)

        # -----------------------------------------------------
        # BALANCE (CENTERED)
        # -----------------------------------------------------

        balance = scores["liberty"] - scores["equality"]

        # map [-1,1] → [0,1]
        balance_norm = (balance + 1.0) / 2.0

        # -----------------------------------------------------
        # DIVERSITY (ENTROPY)
        # -----------------------------------------------------

        diversity = self._entropy(scores)

        # -----------------------------------------------------
        # OUTPUT
        # -----------------------------------------------------

        return {
            "liberty_language_ratio": self._safe(scores["liberty"]),
            "equality_language_ratio": self._safe(scores["equality"]),
            "tradition_language_ratio": self._safe(scores["tradition"]),
            "anti_elite_language_ratio": self._safe(scores["elite"]),
            "liberty_vs_equality_balance": self._safe(balance_norm),
            "ideology_phrase_density": self._safe(phrase_score),
            "ideology_diversity": self._safe(diversity),
        }

    # =========================================================

    def _normalize(self, scores: Dict[str, float]) -> Dict[str, float]:

        values = np.array(list(scores.values()), dtype=np.float32)

        total = float(values.sum())

        if total < EPS:
            return {k: 0.0 for k in scores}

        norm = values / (total + EPS)

        return dict(zip(scores.keys(), norm.astype(float)))

    # =========================================================

    def _entropy(self, scores: Dict[str, float]) -> float:

        values = np.array(list(scores.values()), dtype=np.float32)

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

    def _empty_features(self) -> Dict[str, float]:

        return {
            "liberty_language_ratio": 0.0,
            "equality_language_ratio": 0.0,
            "tradition_language_ratio": 0.0,
            "anti_elite_language_ratio": 0.0,
            "liberty_vs_equality_balance": 0.5,
            "ideology_phrase_density": 0.0,
            "ideology_diversity": 0.0,
        }


# =========================================================
# VECTOR CONVERSION
# =========================================================

def ideological_language_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, IDEOLOGICAL_LANGUAGE_KEYS)