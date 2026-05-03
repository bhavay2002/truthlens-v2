from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import numpy as np

from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.feature_context import FeatureContext
from src.analysis._text_features import (
    phrase_match_count,
    cached_phrase_match_count,
    normalize_lexicon_terms,
    safe_normalized_entropy,
)
from src.analysis.feature_schema import NARRATIVE_PROPAGATION_KEYS, make_vector

logger = logging.getLogger(__name__)


# =========================================================
# CONSTANTS
# =========================================================

EPS = 1e-8
MAX_CLIP = 1.0


# =========================================================
# ANALYZER
# =========================================================

class NarrativePropagationAnalyzer(BaseAnalyzer):

    name = "narrative_propagation"
    expected_keys = set(NARRATIVE_PROPAGATION_KEYS)

    # Conflict verbs grouped by sub-category. The keys here directly
    # produce the `<key>_ratio` output features (violent_conflict_ratio,
    # political_conflict_ratio, etc.) so they MUST stay aligned with
    # NARRATIVE_PROPAGATION_KEYS.
    CONFLICT_VERBS: Dict[str, Set[str]] = {
        "violent_conflict": {
            "attack", "attacked", "attacks", "assault", "assaulted",
            "kill", "killed", "killing", "wound", "wounded", "shoot",
            "shot", "bomb", "bombed", "fight", "fought", "strike",
            "struck", "destroy", "destroyed", "invade", "invaded",
            "war", "warfare", "violence", "violent", "clash", "clashed",
        },
        "political_conflict": {
            "oppose", "opposed", "opposes", "denounce", "denounced",
            "condemn", "condemned", "criticize", "criticized", "blame",
            "blamed", "accuse", "accused", "reject", "rejected",
            "dispute", "disputed", "challenge", "challenged",
            "campaign", "rally", "protest", "protested", "lobby",
        },
        "discursive_conflict": {
            "argue", "argued", "argues", "debate", "debated", "claim",
            "claimed", "assert", "asserted", "deny", "denied",
            "refute", "refuted", "rebut", "rebutted", "contend",
            "contended", "dispute", "disputed",
        },
        "institutional_conflict": {
            "sue", "sued", "indict", "indicted", "prosecute",
            "prosecuted", "investigate", "investigated", "sanction",
            "sanctioned", "regulate", "regulated", "ban", "banned",
            "block", "blocked", "veto", "vetoed", "impeach",
            "impeached", "litigate", "subpoena",
        },
        "coercion_conflict": {
            "force", "forced", "coerce", "coerced", "threaten",
            "threatened", "pressure", "pressured", "intimidate",
            "intimidated", "demand", "demanded", "warn", "warned",
            "punish", "punished", "retaliate", "retaliated",
        },
    }

    OPPOSITION_MARKERS: Set[str] = {
        "but", "however", "yet", "although", "though", "whereas",
        "while", "despite", "in spite of", "on the contrary",
        "on the other hand", "in contrast", "nevertheless",
        "nonetheless", "conversely", "instead", "rather",
    }

    POLARIZATION_TERMS: Set[str] = {
        "us", "them", "we", "they", "ours", "theirs",
        "patriots", "traitors", "elites", "people", "real",
        "fake", "true", "false", "good", "evil", "right", "wrong",
        "always", "never", "all", "none", "every", "everyone",
        "nobody", "extreme", "radical", "extremist",
    }

    CONFLICT_PHRASES: Set[str] = {
        "us versus them", "us vs them", "good versus evil",
        "war on", "war against", "fight against", "stand against",
        "rise up", "take back", "take over", "shut down",
        "double down", "lash out", "crack down", "fight back",
    }

    # =========================================================

    def __init__(self):

        self.conflict_verbs = {
            k: normalize_lexicon_terms(v)
            for k, v in self.CONFLICT_VERBS.items()
        }

        self.opposition = normalize_lexicon_terms(self.OPPOSITION_MARKERS)
        self.polarization = normalize_lexicon_terms(self.POLARIZATION_TERMS)
        self.conflict_phrases = normalize_lexicon_terms(self.CONFLICT_PHRASES)

        logger.info("NarrativePropagationAnalyzer initialized (final)")

    # =========================================================

    def analyze(
        self,
        ctx: FeatureContext,
        hero_entities: Optional[List[str]] = None,
        villain_entities: Optional[List[str]] = None,
        victim_entities: Optional[List[str]] = None,
    ) -> Dict[str, float]:

        # 🔥 lazy-safe
        ctx.ensure_tokens()

        if ctx.safe_n_tokens() == 0:
            return self._empty()

        n_tokens = ctx.safe_n_tokens()

        # -----------------------------------------------------
        # CONFLICT DISTRIBUTION
        # -----------------------------------------------------
        # `raw`  — per-category token densities in [0, 1]
        # `dist` — same densities renormalized to a probability
        #          distribution; used only for entropy/diversity, not
        #          surfaced as `_ratio` outputs.

        raw = self._conflict_distribution(ctx)
        dist = self._normalize(raw)

        # -----------------------------------------------------
        # OTHER SIGNALS
        # -----------------------------------------------------

        opposition = self._density(ctx, self.opposition)
        polarization = self._density(ctx, self.polarization)
        phrase = self._density(ctx, self.conflict_phrases)

        # -----------------------------------------------------
        # ACTOR STRUCTURE
        # -----------------------------------------------------

        actor = self._actor_roles(
            ctx,
            hero_entities,
            villain_entities,
            victim_entities,
        )

        # -----------------------------------------------------
        # PROPAGATION INTENSITY
        # -----------------------------------------------------

        propagation = (
            0.4 * sum(raw.values()) +
            0.2 * opposition +
            0.2 * polarization +
            0.2 * phrase
        )

        # -----------------------------------------------------
        # DIVERSITY
        # -----------------------------------------------------

        diversity = self._entropy(dist)

        # -----------------------------------------------------
        # PUNCTUATION (FIXED)
        # -----------------------------------------------------

        exclaim = self._punctuation(ctx, "!")
        question = self._punctuation(ctx, "?")

        # -----------------------------------------------------
        # OUTPUT
        # -----------------------------------------------------

        return {
            **{f"{k}_ratio": self._safe(v) for k, v in raw.items()},
            "opposition_marker_ratio": self._safe(opposition),
            "polarization_ratio": self._safe(polarization),
            "conflict_phrase_ratio": self._safe(phrase),
            **actor,
            "conflict_propagation_intensity": self._safe(propagation),
            "conflict_diversity": self._safe(diversity),
            "conflict_exclamation_ratio": self._safe(exclaim),
            "conflict_question_ratio": self._safe(question),
        }

    # =========================================================

    def _conflict_distribution(self, ctx: FeatureContext) -> Dict[str, float]:

        n_tokens = ctx.safe_n_tokens()

        return {
            k: sum(ctx.safe_counts().get(t, 0) for t in lexicon) / (n_tokens + EPS)
            for k, lexicon in self.conflict_verbs.items()
        }

    # =========================================================

    def _density(self, ctx: FeatureContext, lexicon: Set[str]) -> float:

        n_tokens = ctx.safe_n_tokens()

        token_hits = sum(
            ctx.safe_counts().get(t, 0)
            for t in lexicon
            if " " not in t
        )

        # PERF-A2: shared per-ctx phrase-hit cache.
        phrase_hits = cached_phrase_match_count(ctx, lexicon)

        # 🔥 weighted fusion
        combined = 0.7 * token_hits + 0.3 * phrase_hits

        return combined / (n_tokens + EPS)

    # =========================================================

    def _actor_roles(
        self,
        ctx: FeatureContext,
        heroes: Optional[List[str]],
        villains: Optional[List[str]],
        victims: Optional[List[str]],
    ) -> Dict[str, float]:

        text = ctx.text_lower or ""

        heroes = heroes or []
        villains = villains or []
        victims = victims or []

        hero_mentions = sum(text.count(h.lower()) for h in heroes)
        villain_mentions = sum(text.count(v.lower()) for v in villains)
        victim_mentions = sum(text.count(v.lower()) for v in victims)

        total = hero_mentions + villain_mentions + victim_mentions

        if total == 0:
            return {
                "hero_villain_conflict_score": 0.0,
                "villain_victim_harm_score": 0.0,
                "hero_victim_protection_score": 0.0,
            }

        return {
            "hero_villain_conflict_score":
                self._safe(min(hero_mentions, villain_mentions) / total),
            "villain_victim_harm_score":
                self._safe(min(villain_mentions, victim_mentions) / total),
            "hero_victim_protection_score":
                self._safe(min(hero_mentions, victim_mentions) / total),
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

    def _entropy(self, dist: Dict[str, float]) -> float:

        # NUM-A1: delegate to the shared, well-guarded helper so single-
        # category distributions and zero-mass distributions return 0.0
        # cleanly instead of dividing by ~EPS.
        return safe_normalized_entropy(dist.values())

    # =========================================================

    def _punctuation(self, ctx: FeatureContext, symbol: str) -> float:

        # PERF-A1: shared punctuation-count cache.
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
            "violent_conflict_ratio": 0.0,
            "political_conflict_ratio": 0.0,
            "discursive_conflict_ratio": 0.0,
            "institutional_conflict_ratio": 0.0,
            "coercion_conflict_ratio": 0.0,
            "opposition_marker_ratio": 0.0,
            "polarization_ratio": 0.0,
            "conflict_phrase_ratio": 0.0,
            "hero_villain_conflict_score": 0.0,
            "villain_victim_harm_score": 0.0,
            "hero_victim_protection_score": 0.0,
            "conflict_propagation_intensity": 0.0,
            "conflict_diversity": 0.0,
            "conflict_exclamation_ratio": 0.0,
            "conflict_question_ratio": 0.0,
        }


# =========================================================
# VECTOR
# =========================================================

def narrative_propagation_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, NARRATIVE_PROPAGATION_KEYS)