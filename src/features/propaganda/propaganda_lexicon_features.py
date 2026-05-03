# src/features/propaganda_lexicon_features.py

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature
from src.features.base.lexicon_matcher import LexiconMatcher, to_token_array
from src.features.base.numerics import normalized_entropy
from src.features.base.tokenization import ensure_tokens_word

logger = logging.getLogger(__name__)

EPS = 1e-8
MAX_CLIP = 1.0


# ---------------------------------------------------------
# Lexicons (reuse yours)
# ---------------------------------------------------------

NAME_CALLING = {...}
FEAR_APPEAL = {...}
EXAGGERATION = {...}
BANDWAGON = {...}
SLOGANS = {...}

BANDWAGON_PHRASES = [...]
SLOGAN_PHRASES = [...]


# ---------------------------------------------------------
# Vectorized matchers (built once at import)
# ---------------------------------------------------------

_PROP_LEX_MATCHERS: Dict[str, LexiconMatcher] = {
    "name_calling": LexiconMatcher(NAME_CALLING, "name_calling"),
    "fear":         LexiconMatcher(FEAR_APPEAL,  "fear"),
    "exaggeration": LexiconMatcher(EXAGGERATION, "exaggeration"),
    "bandwagon":    LexiconMatcher(BANDWAGON,    "bandwagon"),
    "slogan":       LexiconMatcher(SLOGANS,      "slogan"),
}


# ---------------------------------------------------------
# Helpers (legacy — retained for any external callers)
# ---------------------------------------------------------

def _count(counter: Counter, lexicon: Set[str]) -> int:
    return sum(counter.get(w, 0) for w in lexicon)


def _ratio(counter: Counter, lexicon: Set[str], total: int) -> float:
    return _count(counter, lexicon) / (total + EPS)


def _phrase_hits(text: str, patterns: List[str]) -> int:
    return sum(bool(re.search(p, text)) for p in patterns)


# ---------------------------------------------------------
# Feature
# ---------------------------------------------------------

@dataclass
@register_feature
class PropagandaLexiconFeatures(BaseFeature):

    name: str = "propaganda_lexicon_features"
    group: str = "propaganda"
    description: str = "Normalized propaganda lexicon + phrase features"

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip().lower()
        if not text:
            return {}

        tokens = ensure_tokens_word(context, text)
        n = len(tokens)

        if n == 0:
            return {}

        # Vectorized counts via shared matchers.
        tokens_arr = to_token_array(tokens)
        denom = n + EPS
        raw = {
            key: matcher.count_in_tokens(tokens_arr) / denom
            for key, matcher in _PROP_LEX_MATCHERS.items()
        }

        # -------------------------
        # PHRASE SIGNALS (INTEGRATED)
        # -------------------------

        phrase_bandwagon = _phrase_hits(text, BANDWAGON_PHRASES)
        phrase_slogan = _phrase_hits(text, SLOGAN_PHRASES)

        raw["bandwagon"] += phrase_bandwagon * 0.1
        raw["slogan"] += phrase_slogan * 0.1

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

        probs = np.array(list(dist.values()), dtype=np.float32)

        # -------------------------
        # INTENSITY
        # -------------------------

        intensity = float(np.linalg.norm(values))

        # -------------------------
        # ENTROPY (CRITICAL)
        # -------------------------

        entropy = normalized_entropy(probs)

        # -------------------------
        # DIVERSITY
        # -------------------------

        diversity = float(np.count_nonzero(values) / len(values))

        # -------------------------
        # RHETORIC + CAPS (shared, NER-masked)
        # -------------------------
        # Audit fix §2.3 + §3.2 — read the shared signals computed once
        # per context; both ! and caps emphasis come from the
        # headline-weighted, NER-aware ``text_signals`` helper.

        from src.features.base.text_signals import get_text_signals
        signals = get_text_signals(context, n)
        caps_ratio = signals["caps_ratio"]
        # Audit fix §4.3 — question_density also reads from the shared
        # cache; was previously a duplicate ``text.count('?') / n``
        # computation in this file.
        rhetoric = signals["exclamation_density"] + signals["question_density"]

        # -------------------------
        # OUTPUT
        # -------------------------

        return {
            "prop_lex_name_calling": self._safe(dist["name_calling"]),
            "prop_lex_fear": self._safe(dist["fear"]),
            "prop_lex_exaggeration": self._safe(dist["exaggeration"]),
            "prop_lex_bandwagon": self._safe(dist["bandwagon"]),
            "prop_lex_slogan": self._safe(dist["slogan"]),

            "prop_lex_phrase_bandwagon": float(phrase_bandwagon),
            "prop_lex_phrase_slogan": float(phrase_slogan),

            "prop_lex_intensity": self._safe(intensity),
            "prop_lex_entropy": self._safe(entropy),
            "prop_lex_diversity": self._safe(diversity),

            "prop_lex_rhetoric": self._safe(rhetoric),
            "prop_lex_caps_ratio": self._safe(caps_ratio),
        }

    # -----------------------------------------------------

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))

    # -----------------------------------------------------

    def extract_batch(self, contexts):
        return [self.extract(ctx) for ctx in contexts]