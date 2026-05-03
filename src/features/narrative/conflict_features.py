from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature
from src.features.base.lexicon_loader import load_lexicon_set
from src.features.base.lexicon_matcher import LexiconMatcher, to_token_array
from src.features.base.numerics import EPS, MAX_CLIP
from src.features.base.text_signals import get_text_signals
from src.features.base.tokenization import ensure_tokens_word

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Lexicons — audit fix §1.1, see src/config/lexicons/narrative_conflict.json.
# ---------------------------------------------------------

CONFRONTATION_TERMS = load_lexicon_set("narrative_conflict", "confrontation")
DISPUTE_TERMS = load_lexicon_set("narrative_conflict", "dispute")
ACCUSATION_TERMS = load_lexicon_set("narrative_conflict", "accusation")
AGGRESSIVE_LANGUAGE = load_lexicon_set("narrative_conflict", "aggressive")
POLARIZATION_TERMS = load_lexicon_set("narrative_conflict", "polarization")
ESCALATION_TERMS = load_lexicon_set("narrative_conflict", "escalation")


# ---------------------------------------------------------
# Vectorized matchers — audit fix §2.2.
# ---------------------------------------------------------

_CONFLICT_MATCHERS: Dict[str, LexiconMatcher] = {
    "confrontation": LexiconMatcher(CONFRONTATION_TERMS, "conflict_confrontation"),
    "dispute":       LexiconMatcher(DISPUTE_TERMS,       "conflict_dispute"),
    "accusation":    LexiconMatcher(ACCUSATION_TERMS,    "conflict_accusation"),
    "aggression":    LexiconMatcher(AGGRESSIVE_LANGUAGE, "conflict_aggression"),
    "polarization":  LexiconMatcher(POLARIZATION_TERMS,  "conflict_polarization"),
    "escalation":    LexiconMatcher(ESCALATION_TERMS,    "conflict_escalation"),
}


# ---------------------------------------------------------
# Feature
# ---------------------------------------------------------

@dataclass
@register_feature
class ConflictFeatures(BaseFeature):

    name: str = "conflict_features"
    group: str = "conflict"
    description: str = "Normalized conflict discourse features"

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return {}

        tokens = ensure_tokens_word(context, text)
        n = len(tokens)

        if n == 0:
            return {}

        # Audit fix §2.2 — vectorised lexicon counts. Single
        # ``to_token_array`` materialises the contiguous numpy view
        # once for all six categories.
        tokens_arr = to_token_array(tokens)
        denom = n + EPS
        raw = {
            key: matcher.count_in_tokens(tokens_arr) / denom
            for key, matcher in _CONFLICT_MATCHERS.items()
        }

        # -------------------------
        # NORMALIZED DISTRIBUTION
        # -------------------------

        values = np.array(list(raw.values()), dtype=np.float32)
        total = float(values.sum())

        if total < EPS:
            dist = {k: 0.0 for k in raw}
        else:
            norm = values / (total + EPS)
            dist = dict(zip(raw.keys(), norm.astype(float)))

        # -------------------------
        # INTENSITY (STRONGER)
        # -------------------------

        intensity = float(np.linalg.norm(values))

        # -------------------------
        # DIVERSITY (weighted)
        # -------------------------

        diversity = float(np.count_nonzero(values) / len(values))

        # -------------------------
        # RHETORIC — audit fix §3.4 / §4.3 read from the shared,
        # NER-aware text-signal cache instead of recomputing
        # ``text.count("!")`` / ``text.count("?")`` inline.
        # -------------------------

        signals = get_text_signals(context, n)
        rhetoric = signals["exclamation_density"] + signals["question_density"]

        # -------------------------
        # OUTPUT
        # -------------------------

        return {
            "conflict_confrontation_ratio": self._safe(dist["confrontation"]),
            "conflict_dispute_ratio": self._safe(dist["dispute"]),
            "conflict_accusation_ratio": self._safe(dist["accusation"]),
            "conflict_aggression_ratio": self._safe(dist["aggression"]),
            "conflict_polarization_ratio": self._safe(dist["polarization"]),
            "conflict_escalation_ratio": self._safe(dist["escalation"]),

            "conflict_intensity": self._safe(intensity),
            "conflict_diversity": self._safe(diversity),

            "conflict_rhetoric_score": self._safe(rhetoric),
        }

    # -----------------------------------------------------

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))
