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
from src.analysis.feature_schema import SOURCE_ATTRIBUTION_KEYS, make_vector
from src.analysis.spacy_loader import get_doc

logger = logging.getLogger(__name__)

EPS = 1e-8
MAX_CLIP = 1.0


class SourceAttributionAnalyzer(BaseAnalyzer):

    name = "source_attribution"
    expected_keys = set(SOURCE_ATTRIBUTION_KEYS)

    # -----------------------------------------------------

    EXPERT_TERMS = {...}
    ANONYMOUS_TERMS = {...}
    CREDIBILITY_TERMS = {...}
    ATTRIBUTION_VERBS = {...}

    # F15: paired-quote regex. The previous `[\"“”]+` pattern matched
    # individual quote characters (or runs of them) and `_quote_density`
    # then summed `len(m)` — both inflating the score for stylistic
    # quoting and counting unmatched quotes. We now capture *quoted
    # spans* and emit one count per span, which better matches the
    # metric's intent ("quotation density" → quoted-passage rate).
    QUOTE_PATTERN = re.compile(
        r"\"[^\"]+\"|“[^”]+”"
    )

    # =========================================================

    def __init__(self):

        self.expert = normalize_lexicon_terms(self.EXPERT_TERMS)
        self.anonymous = normalize_lexicon_terms(self.ANONYMOUS_TERMS)
        self.credibility = normalize_lexicon_terms(self.CREDIBILITY_TERMS)
        self.verbs = normalize_lexicon_terms(self.ATTRIBUTION_VERBS)

        logger.info("SourceAttributionAnalyzer initialized (final)")

    # =========================================================

    def analyze(self, ctx: FeatureContext) -> Dict[str, float]:

        # 🔥 lazy-safe
        ctx.ensure_tokens()

        if ctx.safe_n_tokens() == 0:
            return self._empty()

        n_tokens = ctx.safe_n_tokens()

        # shared spaCy
        doc = get_doc(ctx, task="ner")

        # -----------------------------------------------------
        # RAW DENSITIES
        # -----------------------------------------------------

        raw = {
            "expert": self._density(ctx, self.expert),
            "anonymous": self._density(ctx, self.anonymous),
            "credibility": self._density(ctx, self.credibility),
            "verbs": self._density(ctx, self.verbs),
        }

        # -----------------------------------------------------
        # NORMALIZATION
        # -----------------------------------------------------

        dist = self._normalize(raw)

        # -----------------------------------------------------
        # QUOTES
        # -----------------------------------------------------

        quote_ratio = self._quote_density(ctx, n_tokens)

        # -----------------------------------------------------
        # NAMED SOURCES (FIXED → uses shared doc)
        # -----------------------------------------------------

        named_ratio = self._named_source_density(doc)

        # -----------------------------------------------------
        # BALANCE
        # -----------------------------------------------------

        balance = dist["expert"] / (dist["expert"] + dist["anonymous"] + EPS)

        # -----------------------------------------------------
        # INTENSITY
        # -----------------------------------------------------

        intensity = sum(raw.values()) / (len(raw) + EPS)

        # -----------------------------------------------------
        # DIVERSITY
        # -----------------------------------------------------

        diversity = self._entropy(dist)

        # -----------------------------------------------------
        # OUTPUT
        # -----------------------------------------------------

        return {
            "expert_attribution_ratio": self._safe(dist["expert"]),
            "anonymous_source_ratio": self._safe(dist["anonymous"]),
            "credibility_indicator_ratio": self._safe(dist["credibility"]),
            "attribution_verb_ratio": self._safe(dist["verbs"]),
            "quotation_ratio": self._safe(quote_ratio),
            "named_source_ratio": self._safe(named_ratio),
            "source_credibility_balance": self._safe(balance),
            "attribution_intensity": self._safe(intensity),
            "attribution_diversity": self._safe(diversity),
        }

    # =========================================================

    def _density(self, ctx: FeatureContext, lexicon: Set[str]) -> float:

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

        return 0.7 * token_score + 0.3 * phrase_score

    # =========================================================

    def _quote_density(self, ctx: FeatureContext, n_tokens: int) -> float:

        text = ctx.text_lower or ""

        # F15: count of *quoted spans*, not characters.
        matches = self.QUOTE_PATTERN.findall(text)

        if not matches:
            return 0.0

        return len(matches) / (n_tokens + EPS)

    # =========================================================

    def _named_source_density(self, doc) -> float:

        entities = [
            ent for ent in doc.ents
            if ent.label_ in ("PERSON", "ORG")
        ]

        if not entities:
            return 0.0

        return len(entities) / (len(doc) + EPS)

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
            "expert_attribution_ratio": 0.0,
            "anonymous_source_ratio": 0.0,
            "credibility_indicator_ratio": 0.0,
            "attribution_verb_ratio": 0.0,
            "quotation_ratio": 0.0,
            "named_source_ratio": 0.0,
            "source_credibility_balance": 0.5,
            "attribution_intensity": 0.0,
            "attribution_diversity": 0.0,
        }


# =========================================================
# VECTOR
# =========================================================

def source_attribution_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, SOURCE_ATTRIBUTION_KEYS)