from __future__ import annotations

import logging
from typing import Dict, Optional, List, Set

import numpy as np

from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.feature_context import FeatureContext
from src.analysis._text_features import (
    phrase_match_count,
    cached_phrase_match_count,
    normalize_lexicon_terms,
)
from src.analysis.feature_schema import NARRATIVE_CONFLICT_KEYS, make_vector
from src.analysis.spacy_loader import get_doc

logger = logging.getLogger(__name__)


# =========================================================
# CONSTANTS
# =========================================================

EPS = 1e-8
MAX_CLIP = 1.0


# =========================================================
# ANALYZER
# =========================================================

class NarrativeConflictAnalyzer(BaseAnalyzer):

    name = "narrative_conflict"
    expected_keys = set(NARRATIVE_CONFLICT_KEYS)

    CONFLICT_VERBS: Set[str] = {...}
    OPPOSITION_MARKERS: Set[str] = {...}
    POLARIZATION_TERMS: Set[str] = {...}

    # =========================================================

    def __init__(self):

        self.conflict_verbs = normalize_lexicon_terms(self.CONFLICT_VERBS)
        self.opposition = normalize_lexicon_terms(self.OPPOSITION_MARKERS)
        self.polarization = normalize_lexicon_terms(self.POLARIZATION_TERMS)

        logger.info("NarrativeConflictAnalyzer initialized (final)")

    # =========================================================

    def analyze(
        self,
        ctx: FeatureContext,
        hero_entities: Optional[List[str]] = None,
        villain_entities: Optional[List[str]] = None,
        victim_entities: Optional[List[str]] = None,
    ) -> Dict[str, float]:

        # 🔥 ensure lazy + shared NLP
        ctx.ensure_tokens()
        doc = get_doc(ctx, task="syntax")

        if ctx.safe_n_tokens() == 0:
            return self._empty()

        n_tokens = ctx.safe_n_tokens()

        # -----------------------------------------------------
        # CORE SIGNALS
        # -----------------------------------------------------

        conflict = self._conflict_verbs(doc)
        opposition = self._density(ctx, self.opposition)
        polarization = self._density(ctx, self.polarization)

        # -----------------------------------------------------
        # NORMALIZATION
        # -----------------------------------------------------

        total = conflict + opposition + polarization + EPS

        conflict_n = conflict / total
        opposition_n = opposition / total
        polarization_n = polarization / total

        # -----------------------------------------------------
        # ACTOR STRUCTURE
        # -----------------------------------------------------

        actor_score = self._actor_structure(
            ctx,
            hero_entities,
            villain_entities,
            victim_entities,
        )

        # -----------------------------------------------------
        # GLOBAL CONFLICT INTENSITY
        # -----------------------------------------------------

        conflict_intensity = (
            0.4 * conflict_n +
            0.3 * opposition_n +
            0.3 * polarization_n
        )

        # -----------------------------------------------------
        # PUNCTUATION (FIXED)
        # -----------------------------------------------------

        exclaim = self._punctuation(ctx, "!")
        question = self._punctuation(ctx, "?")

        # -----------------------------------------------------
        # OUTPUT
        # -----------------------------------------------------

        return {
            "conflict_verb_ratio": self._safe(conflict_n),
            "opposition_marker_ratio": self._safe(opposition_n),
            "polarization_ratio": self._safe(polarization_n),
            "hero_villain_victim_ratio": self._safe(actor_score),
            "conflict_intensity": self._safe(conflict_intensity),
            "conflict_exclamation_ratio": self._safe(exclaim),
            "conflict_question_ratio": self._safe(question),
        }

    # =========================================================

    def _conflict_verbs(self, doc) -> float:

        verbs = [t for t in doc if t.pos_ == "VERB"]

        if not verbs:
            return 0.0

        count = sum(
            1 for v in verbs if v.lemma_.lower() in self.conflict_verbs
        )

        return count / (len(verbs) + EPS)

    # =========================================================

    def _density(self, ctx: FeatureContext, lexicon: Set[str]) -> float:

        n_tokens = ctx.safe_n_tokens()

        token_hits = sum(
            ctx.safe_counts().get(term, 0)
            for term in lexicon
            if " " not in term
        )

        # PERF-A2: shared per-ctx phrase-hit cache.
        phrase_hits = cached_phrase_match_count(ctx, lexicon)

        # 🔥 weighted fusion (prevents double counting)
        combined = (0.7 * token_hits + 0.3 * phrase_hits)

        return combined / (n_tokens + EPS)

    # =========================================================

    def _actor_structure(
        self,
        ctx: FeatureContext,
        heroes: Optional[List[str]],
        villains: Optional[List[str]],
        victims: Optional[List[str]],
    ) -> float:

        heroes = heroes or []
        villains = villains or []
        victims = victims or []

        text = ctx.text_lower or ""

        hero_mentions = sum(text.count(h.lower()) for h in heroes)
        villain_mentions = sum(text.count(v.lower()) for v in villains)
        victim_mentions = sum(text.count(v.lower()) for v in victims)

        total = hero_mentions + villain_mentions + victim_mentions

        if total == 0:
            return 0.0

        # interaction score
        interaction = (
            min(hero_mentions, villain_mentions) +
            min(villain_mentions, victim_mentions)
        )

        return interaction / (total + EPS)

    # =========================================================

    def _punctuation(self, ctx: FeatureContext, symbol: str) -> float:

        # PERF-A1: read from the shared per-ctx punctuation cache so
        # `text.count(symbol)` is paid once per (ctx, symbol) regardless
        # of how many analyzers ask for it.
        count = ctx.punct_count(symbol)

        return count / (ctx.safe_n_tokens() + EPS)

    # =========================================================

    def _safe(self, value: float) -> float:

        if not np.isfinite(value):
            return 0.0

        return float(np.clip(value, 0.0, MAX_CLIP))

    # =========================================================

    def _empty(self) -> Dict[str, float]:

        return {
            "conflict_verb_ratio": 0.0,
            "opposition_marker_ratio": 0.0,
            "polarization_ratio": 0.0,
            "hero_villain_victim_ratio": 0.0,
            "conflict_intensity": 0.0,
            "conflict_exclamation_ratio": 0.0,
            "conflict_question_ratio": 0.0,
        }


# =========================================================
# VECTOR
# =========================================================

def narrative_conflict_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, NARRATIVE_CONFLICT_KEYS)