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
# Lexicons — audit fix §1.1, see src/config/lexicons/framing.json.
# ---------------------------------------------------------

ECONOMIC_FRAME = load_lexicon_set("framing", "economic")
MORAL_FRAME = load_lexicon_set("framing", "moral")
SECURITY_FRAME = load_lexicon_set("framing", "security")
HUMAN_INTEREST_FRAME = load_lexicon_set("framing", "human_interest")
CONFLICT_FRAME = load_lexicon_set("framing", "conflict")

# Reserved for compiled multi-word patterns (currently none).
COMPILED_FRAME_PHRASES: list = []


# ---------------------------------------------------------
# Vectorized matchers — audit fix §2.2.
#
# Built once at import time so every per-document call is a single
# ``np.isin`` per category instead of the previous ``Counter(tokens) +
# sum(counter.get(w, 0) for w in lexicon)`` Python loop. On 20-document
# batches this is the difference between ~120 ms and ~8 ms per
# extractor in synthetic profiling on the existing ``LexiconMatcher``.
# ---------------------------------------------------------

_FRAME_MATCHERS: Dict[str, LexiconMatcher] = {
    "economic": LexiconMatcher(ECONOMIC_FRAME,       "frame_economic"),
    "moral":    LexiconMatcher(MORAL_FRAME,          "frame_moral"),
    "security": LexiconMatcher(SECURITY_FRAME,       "frame_security"),
    "human":    LexiconMatcher(HUMAN_INTEREST_FRAME, "frame_human"),
    "conflict": LexiconMatcher(CONFLICT_FRAME,       "frame_conflict"),
}


# ---------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------

@dataclass
@register_feature
class FramingFeatures(BaseFeature):

    name: str = "framing_features"
    group: str = "framing"

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

        # Audit fix §2.2 — vectorised lexicon counts replace the
        # per-token Python loop. Single ``to_token_array`` materialises
        # the contiguous numpy view once for all five categories.
        tokens_arr = to_token_array(tokens)
        raw = {
            key: matcher.count_in_tokens(tokens_arr) / denom
            for key, matcher in _FRAME_MATCHERS.items()
        }

        # -------------------------
        # NORMALIZED DISTRIBUTION (CRITICAL)
        # -------------------------

        values = np.array(list(raw.values()), dtype=np.float32)
        total = float(values.sum())

        if total < EPS:
            dist = {k: 0.0 for k in raw}
        else:
            norm = values / (total + EPS)
            dist = dict(zip(raw.keys(), norm.astype(float)))

        # -------------------------
        # PHRASE INTENSITY (FIXED)
        # -------------------------

        phrase_hits = sum(len(p.findall(text_lower)) for p in COMPILED_FRAME_PHRASES)
        phrase_score = phrase_hits / (n + EPS)

        # -------------------------
        # STRUCTURAL SIGNALS
        # -------------------------

        quote_count = text.count('"')
        quote_density = quote_count / (n + EPS)

        # -------------------------
        # GLOBAL METRICS
        # -------------------------

        intensity = float(np.mean(list(raw.values())))

        probs = np.array(list(dist.values()), dtype=np.float32)

        entropy = normalized_entropy(probs)

        diversity = sum(v > 0 for v in raw.values()) / len(raw)

        dominance = max(dist.values()) if dist else 0.0

        # -------------------------
        # OUTPUT
        # -------------------------

        return {
            "frame_economic": self._safe(dist["economic"]),
            "frame_moral": self._safe(dist["moral"]),
            "frame_security": self._safe(dist["security"]),
            "frame_human": self._safe(dist["human"]),
            "frame_conflict": self._safe(dist["conflict"]),

            "frame_phrase_score": self._safe(phrase_score),
            "frame_quote_density": self._safe(quote_density),

            "frame_intensity": self._safe(intensity),
            "frame_diversity": self._safe(diversity),
            "frame_entropy": self._safe(entropy),
            "frame_dominance": self._safe(dominance),
        }

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))
