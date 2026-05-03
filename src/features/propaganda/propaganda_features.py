# src/features/propaganda_features.py

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Set

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
# Lexicons — audit fix §1.1, see src/config/lexicons/propaganda.json.
# ---------------------------------------------------------

NAME_CALLING = load_lexicon_set("propaganda", "name_calling")
FEAR_APPEAL = load_lexicon_set("propaganda", "fear_appeal")
EXAGGERATION = load_lexicon_set("propaganda", "exaggeration")
GLITTERING_GENERALITIES = load_lexicon_set("propaganda", "glittering_generalities")
US_VS_THEM = load_lexicon_set("propaganda", "us_vs_them")
AUTHORITY_APPEAL = load_lexicon_set("propaganda", "authority_appeal")
INTENSIFIERS = load_lexicon_set("propaganda", "intensifiers")


# ---------------------------------------------------------
# Vectorized matchers (built once at import)
# ---------------------------------------------------------

_PROP_MATCHERS: Dict[str, LexiconMatcher] = {
    "name_calling": LexiconMatcher(NAME_CALLING,             "name_calling"),
    "fear":         LexiconMatcher(FEAR_APPEAL,              "fear"),
    "exaggeration": LexiconMatcher(EXAGGERATION,             "exaggeration"),
    "glitter":      LexiconMatcher(GLITTERING_GENERALITIES,  "glitter"),
    "us_vs_them":   LexiconMatcher(US_VS_THEM,               "us_vs_them"),
    "authority":    LexiconMatcher(AUTHORITY_APPEAL,         "authority"),
    "intensifier":  LexiconMatcher(INTENSIFIERS,             "intensifier"),
}


# ---------------------------------------------------------
# Helper (legacy — retained for any external callers)
# ---------------------------------------------------------

def _ratio(counter: Counter, lexicon: Set[str], total: int) -> float:
    return sum(counter.get(w, 0) for w in lexicon) / (total + EPS)


# ---------------------------------------------------------
# Feature
# ---------------------------------------------------------

@dataclass
@register_feature
class PropagandaFeatures(BaseFeature):

    name: str = "propaganda_features"
    group: str = "propaganda"
    description: str = "Normalized propaganda feature signals"

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return {}

        tokens = ensure_tokens_word(context, text)
        n = len(tokens)

        if n == 0:
            return {}

        # Vectorized count per category — single np.isin per category,
        # no Python token loop, no Counter materialization.
        tokens_arr = to_token_array(tokens)
        denom = n + EPS
        raw = {
            key: matcher.count_in_tokens(tokens_arr) / denom
            for key, matcher in _PROP_MATCHERS.items()
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
        # DIVERSITY
        # -------------------------

        diversity = float(np.count_nonzero(values) / len(values))

        # -------------------------
        # RHETORIC + CAPS (shared, NER-masked)
        # -------------------------
        # Audit fix §2.3 + §3.2 + §4.3 — share with bias / manipulation
        # extractors via ``get_text_signals``. ``question_density`` now
        # also comes from the shared cache (was duplicated across three
        # propaganda files prior to the §4.3 fix).

        signals = get_text_signals(context, n)
        caps_ratio = signals["caps_ratio"]

        # -------------------------
        # OUTPUT
        # -------------------------

        return {
            "propaganda_name_calling_ratio": self._safe(dist["name_calling"]),
            "propaganda_fear_ratio": self._safe(dist["fear"]),
            "propaganda_exaggeration_ratio": self._safe(dist["exaggeration"]),
            "propaganda_glitter_ratio": self._safe(dist["glitter"]),
            "propaganda_us_vs_them_ratio": self._safe(dist["us_vs_them"]),
            "propaganda_authority_ratio": self._safe(dist["authority"]),
            "propaganda_intensifier_ratio": self._safe(dist["intensifier"]),

            "propaganda_intensity": self._safe(intensity),
            "propaganda_diversity": self._safe(diversity),

            "propaganda_exclamation_density": self._safe(signals["exclamation_density"]),
            "propaganda_caps_ratio": self._safe(caps_ratio),
        }

    # -----------------------------------------------------

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))

    # -----------------------------------------------------

    def extract_batch(self, contexts):
        return [self.extract(ctx) for ctx in contexts]