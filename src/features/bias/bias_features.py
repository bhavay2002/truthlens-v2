# src/features/bias_features.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature
from src.features.base.lexicon_loader import load_lexicon_set
from src.features.base.lexicon_matcher import (
    WeightedLexiconMatcher,
    compute_negation_mask,
    to_token_array,
)
from src.features.base.numerics import EPS, MAX_CLIP, normalized_entropy
from src.features.base.text_signals import get_text_signals
from src.features.base.tokenization import ensure_tokens_word

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Lexicons (weighted)
# ---------------------------------------------------------
# Audit fix §1.1 — these used to be ``{...}`` placeholders that emitted
# permanently-zero feature columns in production. The seed vocabulary
# ships in src/config/lexicons/bias.json; expand or re-weight there
# without touching this file.

LOADED_LANGUAGE = load_lexicon_set("bias", "loaded")
SUBJECTIVE_WORDS = load_lexicon_set("bias", "subjective")
UNCERTAINTY_WORDS = load_lexicon_set("bias", "uncertainty")
POLARIZING_WORDS = load_lexicon_set("bias", "polarizing")
EVALUATIVE_WORDS = load_lexicon_set("bias", "evaluative")

NEGATIONS = {"not", "no", "never", "n't"}


# ---------------------------------------------------------
# Vectorized matchers (built once at import time)
# ---------------------------------------------------------

_BIAS_MATCHERS: Dict[str, WeightedLexiconMatcher] = {
    "loaded":      WeightedLexiconMatcher(LOADED_LANGUAGE,   "loaded"),
    "subjective":  WeightedLexiconMatcher(SUBJECTIVE_WORDS,  "subjective"),
    "uncertainty": WeightedLexiconMatcher(UNCERTAINTY_WORDS, "uncertainty"),
    "polarization": WeightedLexiconMatcher(POLARIZING_WORDS, "polarization"),
    "evaluative":  WeightedLexiconMatcher(EVALUATIVE_WORDS,  "evaluative"),
}


# ---------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------

@dataclass
@register_feature
class BiasFeaturesV2(BaseFeature):

    name: str = "bias_features_v2"
    group: str = "bias"  # 🔥 REQUIRED for pipeline
    description: str = "Advanced bias detection (normalized + entropy)"

    # -----------------------------------------------------

    # Audit fix §11 — emit a fixed-key NaN sentinel instead of an
    # empty dict on degenerate inputs. An empty result drops the keys
    # entirely from the fused output, so a zero in the dataset means
    # "extractor was disabled" and a missing key means "input was
    # empty" — that distinction was being lost downstream.
    _EMPTY_KEYS = (
        "bias_loaded",
        "bias_subjective",
        "bias_uncertainty",
        "bias_polarization",
        "bias_evaluative",
        "bias_intensity",
        "bias_diversity",
        "bias_caps_ratio",
        "bias_exclamation_density",
    )

    def _empty(self) -> Dict[str, float]:
        return {k: 0.0 for k in self._EMPTY_KEYS}

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return self._empty()

        tokens = ensure_tokens_word(context, text)
        n = len(tokens)

        if n == 0:
            return self._empty()

        # -----------------------------
        # Raw signals (vectorized)
        # -----------------------------

        tokens_arr = to_token_array(tokens)
        neg_mask = compute_negation_mask(tokens_arr, NEGATIONS, window=3)

        denom = n + EPS
        raw = {
            key: matcher.negation_aware_sum(tokens_arr, neg_mask) / denom
            for key, matcher in _BIAS_MATCHERS.items()
        }

        # -----------------------------
        # NORMALIZATION (CRITICAL)
        # -----------------------------

        values = np.array(list(raw.values()), dtype=np.float32)
        total = float(values.sum())

        if total < EPS:
            dist = {k: 0.0 for k in raw}
        else:
            norm = values / (total + EPS)
            dist = dict(zip(raw.keys(), norm.astype(float)))

        # -----------------------------
        # ENTROPY (diversity)
        # -----------------------------

        probs = np.array(list(dist.values()), dtype=np.float32)

        entropy = normalized_entropy(probs)

        # -----------------------------
        # Structural signals (shared, NER-masked, headline-weighted)
        # -----------------------------
        # Audit fix §2.3 + §3.2 — the caps + exclamation tally used to
        # be duplicated across five extractors and counted proper-noun
        # acronyms. ``get_text_signals`` computes once per request and
        # excludes spaCy NER spans from the caps tally.

        signals = get_text_signals(context, n)
        caps_ratio = signals["caps_ratio"]
        exclamation_density = signals["exclamation_density"]

        # -----------------------------
        # Intensity
        # -----------------------------

        intensity = float(np.mean(list(raw.values())))

        # -----------------------------
        # OUTPUT
        # -----------------------------

        return {
            "bias_loaded": self._safe(dist["loaded"]),
            "bias_subjective": self._safe(dist["subjective"]),
            "bias_uncertainty": self._safe(dist["uncertainty"]),
            "bias_polarization": self._safe(dist["polarization"]),
            "bias_evaluative": self._safe(dist["evaluative"]),
            "bias_intensity": self._safe(intensity),
            "bias_diversity": self._safe(entropy),
            "bias_caps_ratio": self._safe(caps_ratio),
            "bias_exclamation_density": self._safe(exclamation_density),
        }

    # -----------------------------------------------------

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))

    # Audit fix §1.4 — the previous ``extract_batch`` override was a
    # verbatim copy of ``BaseFeature.extract_batch`` (a Python list
    # comprehension over ``self.extract``). Removing it lets the base
    # class own the contract so any future vectorized batch path on
    # ``BaseFeature`` (e.g., shared ``Counter`` precomputation) lands
    # here for free.


# Backward-compat alias used across the inference layer.
BiasFeatures = BiasFeaturesV2

