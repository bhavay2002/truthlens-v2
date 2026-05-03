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
from src.features.base.text_signals import get_text_signals
from src.features.base.tokenization import ensure_tokens_word

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Lexicons — audit fix §1.1, see src/config/lexicons/narrative_frame.json.
# Distinct vocabulary from src/features/bias/framing_features.py;
# narrative-side adds ``responsibility`` and tunes the rest for story
# framing rather than policy framing.
# ---------------------------------------------------------

CONFLICT_FRAME = load_lexicon_set("narrative_frame", "conflict")
ECONOMIC_FRAME = load_lexicon_set("narrative_frame", "economic")
HUMAN_INTEREST_FRAME = load_lexicon_set("narrative_frame", "human_interest")
MORAL_FRAME = load_lexicon_set("narrative_frame", "moral")
RESPONSIBILITY_FRAME = load_lexicon_set("narrative_frame", "responsibility")


# ---------------------------------------------------------
# Vectorized matchers — audit fix §2.2.
# ---------------------------------------------------------

_NARR_FRAME_MATCHERS: Dict[str, LexiconMatcher] = {
    "conflict":       LexiconMatcher(CONFLICT_FRAME,       "frame_conflict"),
    "economic":       LexiconMatcher(ECONOMIC_FRAME,       "frame_economic"),
    "human":          LexiconMatcher(HUMAN_INTEREST_FRAME, "frame_human_interest"),
    "moral":          LexiconMatcher(MORAL_FRAME,          "frame_moral"),
    "responsibility": LexiconMatcher(RESPONSIBILITY_FRAME, "frame_responsibility"),
}


# ---------------------------------------------------------
# Feature
# ---------------------------------------------------------

@dataclass
@register_feature
class NarrativeFrameFeatures(BaseFeature):

    name: str = "narrative_frame_features"
    group: str = "framing"
    description: str = "Normalized narrative frame features"

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return {}

        tokens = ensure_tokens_word(context, text)
        n = len(tokens)

        if n == 0:
            return {}

        # Audit fix §2.2 — vectorised lexicon counts.
        tokens_arr = to_token_array(tokens)
        denom = n + EPS

        raw = {
            key: matcher.count_in_tokens(tokens_arr) / denom
            for key, matcher in _NARR_FRAME_MATCHERS.items()
        }

        # -------------------------
        # NORMALIZED DISTRIBUTION
        # -------------------------

        values = np.array(list(raw.values()), dtype=np.float32)
        total = values.sum()

        if total < EPS:
            dist = {k: 0.0 for k in raw}
        else:
            norm = values / (total + EPS)
            dist = dict(zip(raw.keys(), norm.astype(float)))

        # -------------------------
        # INTENSITY
        # -------------------------

        intensity = float(np.linalg.norm(values))

        # -------------------------
        # ENTROPY
        # -------------------------

        probs = np.array(list(dist.values()), dtype=np.float32)

        entropy = normalized_entropy(probs)

        # -------------------------
        # DOMINANCE (FIXED)
        # -------------------------

        dominance = float(np.max(probs))

        # -------------------------
        # BALANCE (FIXED)
        # -------------------------

        balance = 1.0 - float(np.std(probs))

        # -------------------------
        # RHETORIC — audit fix §3.4 / §4.3, shared text-signal cache.
        # -------------------------

        signals = get_text_signals(context, n)
        rhetoric = signals["exclamation_density"] + signals["question_density"]

        # -------------------------
        # OUTPUT
        # -------------------------

        return {
            "frame_conflict": self._safe(dist["conflict"]),
            "frame_economic": self._safe(dist["economic"]),
            "frame_human_interest": self._safe(dist["human"]),
            "frame_moral": self._safe(dist["moral"]),
            "frame_responsibility": self._safe(dist["responsibility"]),

            "frame_intensity": self._safe(intensity),
            "frame_entropy": self._safe(entropy),

            "frame_dominance": self._safe(dominance),
            "frame_balance": self._safe(balance),

            "frame_rhetoric": self._safe(rhetoric),
        }

    # -----------------------------------------------------

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))
