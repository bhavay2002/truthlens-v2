from __future__ import annotations

import logging
import re
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
from src.analysis.feature_schema import CONTEXT_OMISSION_KEYS, make_vector
from src.analysis.spacy_loader import get_doc

logger = logging.getLogger(__name__)


# =========================================================
# CONSTANTS
# =========================================================

EPS = 1e-8
MAX_CLIP = 1.0


# =========================================================
# DETECTOR
# =========================================================

class ContextOmissionDetector(BaseAnalyzer):

    name = "context_omission"
    expected_keys = set(CONTEXT_OMISSION_KEYS)

    VAGUE_REFERENCES = {
        "they","people","many","some","others",
        "experts","critics","sources","analysts",
        "officials","insiders","observers",
        "commentators","reportedly","allegedly",
        "authorities","investigators","researchers",
        "witnesses","participants","leaders",
        "lawmakers","politicians","administration",
        "supporters","opponents","activists",
        "many believe","some claim","others argue",
        "it is said","it is believed","rumor","speculation"
    }

    ATTRIBUTION_MARKERS = {
        "according","according to","reported","reports",
        "stated","claimed","said","noted",
        "explained","announced","revealed","confirmed",
        "suggested","told","wrote","indicated"
    }

    EVIDENCE_MARKERS = {
        "data","study","studies","report","research",
        "analysis","evidence","statistics","survey",
        "experiment","findings","results",
        "research suggests","data indicates"
    }

    UNCERTAINTY_MARKERS = {
        "allegedly","reportedly","apparently",
        "possibly","potentially","likely",
        "rumor","speculation","suggests",
        "appears","seems","may","might","could"
    }

    # F15: drop apostrophes from the quote pattern. Including ASCII `'`
    # and the curly apostrophes ‘ ’ caused contractions ("don't",
    # "it's") and possessives ("Alice's") to dominate the count and
    # inflate `context_quote_ratio` for any normal English prose.
    # Restrict to actual double-quote glyphs.
    QUOTE_PATTERN = re.compile(r"[\"“”]")

    def __init__(self):
        self.vague_phrases = normalize_lexicon_terms(self.VAGUE_REFERENCES)
        self.evidence_phrases = normalize_lexicon_terms(self.EVIDENCE_MARKERS)

    # =========================================================

    def analyze(self, ctx: FeatureContext) -> Dict[str, float]:

        # Section 4: use safe accessor so the analyzer never reads
        # `ctx.n_tokens` while it's still None (BaseAnalyzer normally
        # warms it via ensure_tokens, but defend against direct callers).
        n_tokens = ctx.safe_n_tokens()

        if n_tokens == 0:
            return self._empty_features()

        features: Dict[str, float] = {}

        # 🔥 shared spaCy doc
        doc = get_doc(ctx, task="ner")

        token_counts = ctx.safe_counts()
        text_lower = ctx.text_lower or ""

        # -----------------------------------------------------
        # TOKEN RATIOS
        # -----------------------------------------------------

        features["context_vague_reference_ratio"] = self._safe(
            term_ratio(token_counts, n_tokens, self.VAGUE_REFERENCES)
        )

        features["context_attribution_ratio"] = self._safe(
            term_ratio(token_counts, n_tokens, self.ATTRIBUTION_MARKERS)
        )

        features["context_uncertainty_ratio"] = self._safe(
            term_ratio(token_counts, n_tokens, self.UNCERTAINTY_MARKERS)
        )

        # -----------------------------------------------------
        # PHRASE FEATURES
        # -----------------------------------------------------

        features["context_evidence_ratio"] = self._phrase_ratio(
            ctx, n_tokens, self.evidence_phrases
        )

        # -----------------------------------------------------
        # QUOTES
        # -----------------------------------------------------

        features["context_quote_ratio"] = self._quote_ratio(
            text_lower, n_tokens
        )

        # -----------------------------------------------------
        # ENTITY FEATURES (UPDATED)
        # -----------------------------------------------------

        entity_features = self._entity_features(doc)
        features.update(entity_features)

        # -----------------------------------------------------
        # GROUNDING SCORE (IMPROVED)
        # -----------------------------------------------------

        features["context_grounding_score"] = self._safe(
            0.4 * features["context_evidence_ratio"]
            + 0.3 * features["context_entity_ratio"]
            + 0.2 * (1 - features["context_uncertainty_ratio"])
            + 0.1 * features["context_attribution_ratio"]
        )

        return features

    # =========================================================

    def _phrase_ratio(self, ctx: FeatureContext, n_tokens: int, phrases: set) -> float:
        if n_tokens <= 0:
            return 0.0
        # PERF-A2: shared per-ctx phrase-hit cache.
        hits = cached_phrase_match_count(ctx, phrases)
        return self._safe(hits / (n_tokens + EPS))

    # =========================================================

    def _quote_ratio(self, text_lower: str, n_tokens: int) -> float:
        quotes = len(self.QUOTE_PATTERN.findall(text_lower))
        return self._safe(quotes / (n_tokens + EPS))

    # =========================================================

    def _entity_features(self, doc) -> Dict[str, float]:

        total = max(len(doc), 1)

        entity_count = len(doc.ents)
        entity_ratio = entity_count / total

        entity_types = {ent.label_ for ent in doc.ents}

        # stable diversity scaling
        diversity = np.log1p(len(entity_types)) / np.log1p(20)

        return {
            "context_entity_ratio": self._safe(entity_ratio),
            "context_entity_type_diversity": self._safe(diversity),
        }

    # =========================================================

    def _safe(self, value: float) -> float:
        if not np.isfinite(value):
            return 0.0
        return float(np.clip(value, 0.0, MAX_CLIP))

    # =========================================================

    def _empty_features(self) -> Dict[str, float]:
        return {k: 0.0 for k in CONTEXT_OMISSION_KEYS}


# =========================================================
# VECTOR CONVERSION
# =========================================================

def context_feature_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, CONTEXT_OMISSION_KEYS)