from __future__ import annotations

import logging
from typing import Dict, List, Set, Optional

import numpy as np

from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.feature_context import FeatureContext
from src.analysis.feature_schema import NARRATIVE_ROLE_KEYS, make_vector
from src.analysis._text_features import normalize_lexicon_terms
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

class NarrativeRoleExtractor(BaseAnalyzer):

    name = "narrative_roles"
    expected_keys = set(NARRATIVE_ROLE_KEYS)

    HERO_TERMS: Set[str] = {...}
    VILLAIN_TERMS: Set[str] = {...}
    VICTIM_TERMS: Set[str] = {...}

    # =========================================================

    def __init__(self):

        self.hero_terms = normalize_lexicon_terms(self.HERO_TERMS)
        self.villain_terms = normalize_lexicon_terms(self.VILLAIN_TERMS)
        self.victim_terms = normalize_lexicon_terms(self.VICTIM_TERMS)

        logger.info("NarrativeRoleExtractor initialized (final)")

    # =========================================================

    def analyze(self, ctx: FeatureContext) -> Dict[str, float]:

        # 🔥 shared spaCy
        doc = get_doc(ctx, task="syntax")

        hero_entities: Dict[str, float] = {}
        villain_entities: Dict[str, float] = {}
        victim_entities: Dict[str, float] = {}

        for token in doc:

            lemma = token.lemma_.lower()

            if lemma in self.hero_terms:
                self._assign(token, hero_entities, victim_entities)

            elif lemma in self.villain_terms:
                self._assign(token, villain_entities, victim_entities)

            elif lemma in self.victim_terms:
                obj = self._get_object(token)
                if obj:
                    victim_entities[obj] = victim_entities.get(obj, 0) + 1

            # passive victim detection
            if token.dep_ == "nsubjpass":
                entity = self._resolve_entity(token)
                if entity:
                    victim_entities[entity] = victim_entities.get(entity, 0) + 1

        # -----------------------------------------------------
        # CONVERT TO SCORES (IMPORTANT FIX)
        # -----------------------------------------------------

        return self._role_scores(
            hero_entities,
            villain_entities,
            victim_entities,
        )

    # =========================================================

    def _assign(self, token, actor_dict, victim_dict):

        subject = self._get_subject(token)
        obj = self._get_object(token)

        if subject:
            actor_dict[subject] = actor_dict.get(subject, 0) + 1

        if obj:
            victim_dict[obj] = victim_dict.get(obj, 0) + 1

    # =========================================================

    def _get_subject(self, token) -> Optional[str]:
        for child in token.children:
            if child.dep_ in {"nsubj", "nsubjpass"}:
                return self._resolve_entity(child)
        return None

    # =========================================================

    def _get_object(self, token) -> Optional[str]:
        for child in token.children:
            if child.dep_ in {"dobj", "pobj", "obj"}:
                return self._resolve_entity(child)
        return None

    # =========================================================

    def _resolve_entity(self, token) -> Optional[str]:

        # named entity span — token.ent_start/ent_end only exist in
        # spaCy 4+; in spaCy 3 we scan doc.ents instead.
        if token.ent_iob_ in {"B", "I"} and token.ent_type_:
            for ent in token.doc.ents:
                if ent.start <= token.i < ent.end:
                    text = ent.text.lower().strip()
                    if len(text) > 2:
                        return text

        # fallback: noun/proper noun
        if token.pos_ in {"NOUN", "PROPN"}:
            return token.lemma_.lower()

        return None

    # =========================================================

    def _role_scores(
        self,
        hero_entities: Dict[str, float],
        villain_entities: Dict[str, float],
        victim_entities: Dict[str, float],
    ) -> Dict[str, float]:

        heroes = sum(hero_entities.values())
        villains = sum(villain_entities.values())
        victims = sum(victim_entities.values())

        total = heroes + villains + victims

        if total == 0:
            return self._empty()

        hero_r = heroes / total
        villain_r = villains / total
        victim_r = victims / total

        # balance mapping [-1,1] → [0,1]
        hv_balance = (hero_r - villain_r + 1.0) / 2.0

        return {
            "hero_ratio": self._safe(hero_r),
            "villain_ratio": self._safe(villain_r),
            "victim_ratio": self._safe(victim_r),
            "hero_vs_villain_balance": self._safe(hv_balance),
            "hero_entities": float(len(hero_entities)),
            "villain_entities": float(len(villain_entities)),
            "victim_entities": float(len(victim_entities)),
        }

    # =========================================================

    def _safe(self, value: float) -> float:

        if not np.isfinite(value):
            return 0.0

        return float(np.clip(value, 0.0, MAX_CLIP))

    # =========================================================

    def _empty(self) -> Dict[str, float]:

        return {
            "hero_ratio": 0.0,
            "villain_ratio": 0.0,
            "victim_ratio": 0.0,
            "hero_vs_villain_balance": 0.5,
            "hero_entities": 0.0,
            "villain_entities": 0.0,
            "victim_entities": 0.0,
        }


# =========================================================
# VECTOR
# =========================================================

def narrative_role_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, NARRATIVE_ROLE_KEYS)