# src/features/narrative_features.py

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Set

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature
from src.features.base.numerics import normalized_entropy
from src.features.base.tokenization import ensure_tokens_word

logger = logging.getLogger(__name__)

EPS = 1e-8
MAX_CLIP = 1.0


# ---------------------------------------------------------
# Lexicons
# ---------------------------------------------------------

HERO_TERMS = {...}
VILLAIN_TERMS = {...}
VICTIM_TERMS = {...}

CONFLICT_TERMS = {...}
RESOLUTION_TERMS = {...}
CRISIS_TERMS = {...}

POLARIZATION_TERMS = {...}


# ---------------------------------------------------------
# Feature
# ---------------------------------------------------------

@dataclass
@register_feature
class NarrativeFeatures(BaseFeature):

    name: str = "narrative_features"
    group: str = "narrative"
    description: str = "Normalized narrative structure features"

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return {}

        tokens = ensure_tokens_word(context, text)
        n = len(tokens)

        if n == 0:
            return {}

        counter = Counter(tokens)

        def ratio(lexicon: Set[str]) -> float:
            return sum(counter.get(w, 0) for w in lexicon) / (n + EPS)

        raw_roles = {
            "hero": ratio(HERO_TERMS),
            "villain": ratio(VILLAIN_TERMS),
            "victim": ratio(VICTIM_TERMS),
        }

        raw_context = {
            "conflict": ratio(CONFLICT_TERMS),
            "resolution": ratio(RESOLUTION_TERMS),
            "crisis": ratio(CRISIS_TERMS),
            "polarization": ratio(POLARIZATION_TERMS),
        }

        # -------------------------
        # ROLE DISTRIBUTION
        # -------------------------

        role_vals = np.array(list(raw_roles.values()), dtype=np.float32)
        role_total = role_vals.sum()

        if role_total > 0:
            role_dist = role_vals / (role_total + EPS)
        else:
            role_dist = np.zeros_like(role_vals)

        # -------------------------
        # CONTEXT DISTRIBUTION
        # -------------------------

        ctx_vals = np.array(list(raw_context.values()), dtype=np.float32)
        ctx_total = ctx_vals.sum()

        if ctx_total > 0:
            ctx_dist = ctx_vals / (ctx_total + EPS)
        else:
            ctx_dist = np.zeros_like(ctx_vals)

        # -------------------------
        # INTENSITY
        # -------------------------

        intensity = float(np.linalg.norm(ctx_vals))

        # -------------------------
        # ENTROPY
        # -------------------------

        role_entropy = normalized_entropy(role_dist)
        context_entropy = normalized_entropy(ctx_dist)

        # -------------------------
        # PROGRESSION (FIXED)
        # -------------------------

        progression = raw_context["resolution"] / (
            raw_context["resolution"] + raw_context["conflict"] + EPS
        )

        # -------------------------
        # RHETORIC (FIXED)
        # -------------------------

        rhetoric = (text.count("!") + text.count("?")) / (n + EPS)

        # -------------------------
        # OUTPUT
        # -------------------------

        return {
            "narrative_hero": self._safe(role_dist[0]),
            "narrative_villain": self._safe(role_dist[1]),
            "narrative_victim": self._safe(role_dist[2]),

            "narrative_conflict": self._safe(ctx_dist[0]),
            "narrative_resolution": self._safe(ctx_dist[1]),
            "narrative_crisis": self._safe(ctx_dist[2]),
            "narrative_polarization": self._safe(ctx_dist[3]),

            "narrative_intensity": self._safe(intensity),
            "narrative_role_entropy": self._safe(role_entropy),
            "narrative_context_entropy": self._safe(context_entropy),

            "narrative_progression": self._safe(progression),
            "narrative_rhetoric": self._safe(rhetoric),
        }

    # -----------------------------------------------------

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))