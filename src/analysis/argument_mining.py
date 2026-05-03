from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.feature_context import FeatureContext
from src.analysis._text_features import (
    term_ratio,
    phrase_match_count,
    cached_phrase_match_count,
    normalize_lexicon_terms,
)
from src.analysis.feature_schema import ARGUMENT_MINING_KEYS, make_vector
from src.analysis.spacy_loader import get_doc

logger = logging.getLogger(__name__)


# =========================================================
# CONSTANTS
# =========================================================

EPS = 1e-8
MAX_RATIO_CLIP = 1.0


# =========================================================
# ANALYZER
# =========================================================

class ArgumentMiningAnalyzer(BaseAnalyzer):

    CLAIM_MARKERS = {
        "therefore","thus","hence","consequently","so",
        "accordingly","as a result","for this reason",
        "it follows","this proves","this shows",
        "this demonstrates","clearly","obviously",
        "undoubtedly","in conclusion","overall",
        "ultimately","in summary"
    }

    PREMISE_MARKERS = {
        "because","since","given","as",
        "due to","based on","in light of",
        "for the reason that","seeing that"
    }

    SUPPORT_MARKERS = {
        "for example","for instance","to illustrate",
        "as evidence","data shows","studies show",
        "research indicates","statistics show",
        "according to","analysis shows"
    }

    CONTRAST_MARKERS = {
        "however","but","although","though",
        "nevertheless","on the other hand",
        "in contrast","despite","whereas"
    }

    REBUTTAL_MARKERS = {
        "however","nonetheless","still",
        "despite this","even so","that said",
        "in spite of this","contrary to this"
    }

    # -----------------------------------------------------

    def __init__(self):
        self.support_phrases = normalize_lexicon_terms(self.SUPPORT_MARKERS)
        self.rebuttal_phrases = normalize_lexicon_terms(self.REBUTTAL_MARKERS)

    # =========================================================

    def analyze(self, ctx: FeatureContext) -> Dict[str, float]:

        if not ctx.text:
            return self._empty_features()

        #  Shared spaCy doc (CRITICAL optimization)
        doc = get_doc(ctx, task="syntax")

        # Section 4: BaseAnalyzer auto-calls ensure_tokens() in
        # _validate_context, but use the safe accessors so a degraded
        # context (or a future caller that bypasses BaseAnalyzer) still
        # short-circuits cleanly instead of dividing by None.
        n_tokens = ctx.safe_n_tokens()

        if n_tokens == 0:
            return self._empty_features()

        token_counts = ctx.safe_counts()

        text_lower = ctx.text_lower or ctx.text.lower()

        features: Dict[str, float] = {}

        # -----------------------------------------------------
        # TOKEN RATIOS
        # -----------------------------------------------------

        features["argument_claim_ratio"] = self._safe_ratio(
            term_ratio(token_counts, n_tokens, self.CLAIM_MARKERS)
        )

        features["argument_premise_ratio"] = self._safe_ratio(
            term_ratio(token_counts, n_tokens, self.PREMISE_MARKERS)
        )

        features["argument_contrast_ratio"] = self._safe_ratio(
            term_ratio(token_counts, n_tokens, self.CONTRAST_MARKERS)
        )

        # -----------------------------------------------------
        # PHRASE FEATURES
        # -----------------------------------------------------

        features["argument_support_ratio"] = self._phrase_ratio(
            ctx, n_tokens, self.support_phrases
        )

        features["argument_rebuttal_ratio"] = self._phrase_ratio(
            ctx, n_tokens, self.rebuttal_phrases
        )

        # -----------------------------------------------------
        # STRUCTURAL FEATURES (spaCy)
        # -----------------------------------------------------

        density = self._argument_density(doc)
        features.update(density)

        # -----------------------------------------------------
        # COMPLEXITY
        # -----------------------------------------------------

        features["argument_complexity"] = self._safe_ratio(
            density["argument_clause_density"]
            + density["argument_verb_density"]
        )

        return features

    # =========================================================

    def _phrase_ratio(
        self,
        ctx: FeatureContext,
        n_tokens: int,
        phrases: set,
    ) -> float:

        if n_tokens <= 0:
            return 0.0

        # PERF-A2: route through the shared per-ctx phrase-hit cache.
        hits = cached_phrase_match_count(ctx, phrases)

        return self._safe_ratio(hits / (n_tokens + EPS))

    # =========================================================

    def _argument_density(self, doc) -> Dict[str, float]:

        total = max(len(doc), 1)

        verbs = sum(1 for t in doc if t.pos_ == "VERB")
        clauses = sum(
            1 for t in doc if t.dep_ in {"ccomp", "xcomp", "advcl"}
        )

        verb_density = verbs / total
        clause_density = clauses / total

        return {
            "argument_verb_density": self._safe_ratio(verb_density),
            "argument_clause_density": self._safe_ratio(clause_density),
        }

    # =========================================================

    def _safe_ratio(self, value: float) -> float:

        if not np.isfinite(value):
            return 0.0

        return float(np.clip(value, 0.0, MAX_RATIO_CLIP))

    # =========================================================

    def _empty_features(self) -> Dict[str, float]:

        return {
            "argument_claim_ratio": 0.0,
            "argument_premise_ratio": 0.0,
            "argument_support_ratio": 0.0,
            "argument_contrast_ratio": 0.0,
            "argument_rebuttal_ratio": 0.0,
            "argument_verb_density": 0.0,
            "argument_clause_density": 0.0,
            "argument_complexity": 0.0,
        }


# =========================================================
# VECTOR CONVERSION (MODEL-READY)
# =========================================================

def argument_feature_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, ARGUMENT_MINING_KEYS)