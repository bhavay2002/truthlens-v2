from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature
from src.features.base.lexicon_loader import load_lexicon_set
from src.features.base.lexicon_matcher import LexiconMatcher, to_token_array
from src.features.base.numerics import EPS, MAX_CLIP, normalized_entropy
from src.features.base.tokenization import ensure_tokens_word

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Lexicons — audit fix §1.1, see src/config/lexicons/ideology.json.
# ---------------------------------------------------------

LEFT_LEXICON = load_lexicon_set("ideology", "left")
RIGHT_LEXICON = load_lexicon_set("ideology", "right")
POLARIZING_TERMS = load_lexicon_set("ideology", "polarizing")
GROUP_REFERENCES = load_lexicon_set("ideology", "group_references")

# Reserved for compiled multi-word patterns (currently none).
COMPILED_IDEOLOGY_PHRASES: list = []


# ---------------------------------------------------------
# Vectorized matchers — audit fix §2.2.
# ---------------------------------------------------------

_IDEOLOGY_MATCHERS: Dict[str, LexiconMatcher] = {
    "left":         LexiconMatcher(LEFT_LEXICON,     "ideology_left"),
    "right":        LexiconMatcher(RIGHT_LEXICON,    "ideology_right"),
    "polarization": LexiconMatcher(POLARIZING_TERMS, "ideology_polarization"),
    "group_ref":    LexiconMatcher(GROUP_REFERENCES, "ideology_group_reference"),
}


# ---------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------

@dataclass
@register_feature
class IdeologicalFeatures(BaseFeature):

    name: str = "ideological_features"
    group: str = "ideology"

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return {}

        text_lower = text.lower()
        tokens = ensure_tokens_word(context, text)

        if not tokens:
            return {}

        n = len(tokens)
        denom = n + EPS

        # Audit fix §2.2 — single ``np.isin`` per category replaces the
        # per-token Python loop over four ``ratio(counter, lexicon, n)``
        # calls. The four lexicons share one ``tokens_arr`` view.
        tokens_arr = to_token_array(tokens)
        counts = {
            key: matcher.count_in_tokens(tokens_arr)
            for key, matcher in _IDEOLOGY_MATCHERS.items()
        }

        raw = {
            "left":  counts["left"] / denom,
            "right": counts["right"] / denom,
        }
        polarization = counts["polarization"] / denom
        group_ref = counts["group_ref"] / denom

        # -------------------------
        # NORMALIZED IDEOLOGY (CRITICAL)
        # -------------------------

        values = np.array(list(raw.values()), dtype=np.float32)
        total = values.sum()

        if total < EPS:
            dist = {k: 0.0 for k in raw}
        else:
            norm = values / (total + EPS)
            dist = dict(zip(raw.keys(), norm.astype(float)))

        # -------------------------
        # BALANCE (FIXED)
        # -------------------------

        balance = 1.0 - abs(dist["left"] - dist["right"])

        # -------------------------
        # ENTROPY (FIXED)
        # -------------------------

        probs = np.array(list(dist.values()), dtype=np.float32)

        entropy = normalized_entropy(probs)

        # -------------------------
        # PHRASE INTENSITY
        # -------------------------

        phrase_hits = sum(
            len(p.findall(text_lower)) for p in COMPILED_IDEOLOGY_PHRASES
        )
        phrase_score = phrase_hits / (n + EPS)

        # -------------------------
        # GLOBAL SIGNALS
        # -------------------------

        signal_strength = (
            (raw["left"] + raw["right"]) * 0.6 +
            polarization * 0.4
        )

        # -------------------------
        # OUTPUT
        # -------------------------

        return {
            "ideology_left_ratio": self._safe(dist["left"]),
            "ideology_right_ratio": self._safe(dist["right"]),

            "ideology_balance": self._safe(balance),
            "ideology_entropy": self._safe(entropy),

            "ideology_polarization_ratio": self._safe(polarization),
            "ideology_group_reference_ratio": self._safe(group_ref),

            "ideology_phrase_count": self._safe(phrase_score),

            "ideology_signal_strength": self._safe(signal_strength),
        }

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))
