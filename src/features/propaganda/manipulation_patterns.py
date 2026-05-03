# src/features/manipulation_patterns.py

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Set

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature
from src.features.base.lexicon_loader import load_lexicon_set
from src.features.base.numerics import EPS, MAX_CLIP, normalized_entropy
from src.features.base.text_signals import get_text_signals
from src.features.base.tokenization import ensure_tokens_word, ensure_tokens_word_counter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Lexicons — audit fix §1.1, see src/config/lexicons/manipulation.json.
# ---------------------------------------------------------

URGENCY_TERMS = load_lexicon_set("manipulation", "urgency")
FEAR_TERMS = load_lexicon_set("manipulation", "fear")
BLAME_TERMS = load_lexicon_set("manipulation", "blame")
SCAPEGOAT_TERMS = load_lexicon_set("manipulation", "scapegoat")
ABSOLUTE_TERMS = load_lexicon_set("manipulation", "absolute")
CONSPIRACY_TERMS = load_lexicon_set("manipulation", "conspiracy")
FALSE_DILEMMA_TERMS = load_lexicon_set("manipulation", "false_dilemma")
EXAGGERATION_TERMS = load_lexicon_set("manipulation", "exaggeration")
INTENSIFIERS = load_lexicon_set("manipulation", "intensifiers")


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def _ratio(counter: Counter, lexicon: Set[str], total: int) -> float:
    return sum(counter.get(w, 0) for w in lexicon) / (total + EPS)


# ---------------------------------------------------------
# Feature
# ---------------------------------------------------------

@dataclass
@register_feature
class ManipulationPatterns(BaseFeature):

    name: str = "manipulation_patterns"
    group: str = "propaganda"
    description: str = "Normalized manipulation pattern features"

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return self._empty()

        tokens = ensure_tokens_word(context, text)
        n = len(tokens)

        if n == 0:
            return self._empty()

        # Audit fix §2.5 — share the per-context Counter cache.
        counter = ensure_tokens_word_counter(context)

        raw = {
            "urgency": _ratio(counter, URGENCY_TERMS, n),
            "fear": _ratio(counter, FEAR_TERMS, n),
            "blame": _ratio(counter, BLAME_TERMS, n),
            "scapegoat": _ratio(counter, SCAPEGOAT_TERMS, n),
            "absolute": _ratio(counter, ABSOLUTE_TERMS, n),
            "conspiracy": _ratio(counter, CONSPIRACY_TERMS, n),
            "dilemma": _ratio(counter, FALSE_DILEMMA_TERMS, n),
            "exaggeration": _ratio(counter, EXAGGERATION_TERMS, n),
            "intensifier": _ratio(counter, INTENSIFIERS, n),
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

        probs = np.array(list(dist.values()), dtype=np.float32)

        # -------------------------
        # INTENSITY (STRONGER)
        # -------------------------

        intensity = float(np.linalg.norm(values))

        # -------------------------
        # ENTROPY (CRITICAL)
        # -------------------------

        entropy = normalized_entropy(probs)

        # -------------------------
        # DIVERSITY (WEIGHTED)
        # -------------------------

        diversity = float(np.count_nonzero(values) / len(values))

        # -------------------------
        # RHETORIC + CAPS (shared, NER-masked)
        # -------------------------
        # Audit fix §2.3 + §3.2 — single canonical computation in
        # ``text_signals``; this extractor reads from the cache.

        signals = get_text_signals(context, n)
        caps_ratio = signals["caps_ratio"]
        # Audit fix §4.3 — question_density now sourced from the shared
        # text_signals cache; was a duplicate ``text.count('?') / n`` here.
        rhetoric = signals["exclamation_density"] + signals["question_density"]

        # -------------------------
        # OUTPUT
        # -------------------------

        return {
            "manipulation_urgency": self._safe(dist["urgency"]),
            "manipulation_fear": self._safe(dist["fear"]),
            "manipulation_blame": self._safe(dist["blame"]),
            "manipulation_scapegoat": self._safe(dist["scapegoat"]),
            "manipulation_absolute": self._safe(dist["absolute"]),
            "manipulation_conspiracy": self._safe(dist["conspiracy"]),
            "manipulation_false_dilemma": self._safe(dist["dilemma"]),
            "manipulation_exaggeration": self._safe(dist["exaggeration"]),
            "manipulation_intensifier": self._safe(dist["intensifier"]),

            "manipulation_intensity": self._safe(intensity),
            "manipulation_entropy": self._safe(entropy),
            "manipulation_diversity": self._safe(diversity),

            "manipulation_rhetoric": self._safe(rhetoric),
            "manipulation_caps_emphasis": self._safe(caps_ratio),
        }

    # -----------------------------------------------------

    def _empty(self) -> Dict[str, float]:
        # §11.1 — consistent fixed-key zero dict for empty / zero-token inputs.
        return {
            "manipulation_urgency":      0.0,
            "manipulation_fear":         0.0,
            "manipulation_blame":        0.0,
            "manipulation_scapegoat":    0.0,
            "manipulation_absolute":     0.0,
            "manipulation_conspiracy":   0.0,
            "manipulation_false_dilemma": 0.0,
            "manipulation_exaggeration": 0.0,
            "manipulation_intensifier":  0.0,
            "manipulation_intensity":    0.0,
            "manipulation_entropy":      0.0,
            "manipulation_diversity":    0.0,
            "manipulation_rhetoric":     0.0,
            "manipulation_caps_emphasis": 0.0,
        }

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))