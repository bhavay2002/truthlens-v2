# src/features/token_features.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature
from src.features.base.numerics import EPS, MAX_CLIP, normalized_entropy
from src.features.base.tokenization import ensure_tokens_word

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Feature
# ---------------------------------------------------------

@dataclass
@register_feature
class TokenFeatures(BaseFeature):

    name: str = "token_features"
    group: str = "token"
    description: str = "Advanced token distribution features"

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return {}

        tokens = ensure_tokens_word(context, text)
        n = len(tokens)

        if n == 0:
            return self._empty()

        tokens_arr = np.array(tokens, dtype=str)
        unique, counts = np.unique(tokens_arr, return_counts=True)

        vocab = len(unique)

        # -------------------------
        # BASIC SIZE (RAW log magnitudes)
        # -------------------------
        # Audit fix §1.1 — emit raw log1p; the FeatureScalingPipeline
        # handles cross-corpus normalisation. The previous /10.0 magic
        # constant assumed n ~ exp(10) ≈ 22k tokens, which silently
        # saturated for short news headlines and never reached 1.0 for
        # long-form articles.

        length_log = float(np.log1p(n))
        vocab_log = float(np.log1p(vocab))

        # -------------------------
        # FREQUENCY DISTRIBUTION
        # -------------------------

        probs = counts / (n + EPS)

        # entropy
        entropy = normalized_entropy(probs)

        # -------------------------
        # CONCENTRATION (TOP-K)
        # -------------------------

        topk = np.sort(probs)[-5:] if len(probs) >= 5 else probs
        topk_mass = float(np.sum(topk))

        # -------------------------
        # REPETITION STRENGTH
        # -------------------------

        repetition = float(np.sum(probs ** 2))

        # -------------------------
        # INEQUALITY (GINI-LIKE)
        # -------------------------

        sorted_probs = np.sort(probs)
        gini = float(1.0 - 2.0 * np.sum((len(probs) - np.arange(len(probs))) * sorted_probs) / (len(probs) + EPS))

        # -------------------------
        # TOKEN LENGTH STATS
        # -------------------------

        lengths = np.char.str_len(tokens_arr)

        # Audit fix §1.1 — character length is a raw magnitude, not a
        # ratio. Hand-tuned divisors (/20.0, /10.0) were an ad-hoc
        # squashing into [0, 1]; let the scaling stage do that.
        avg_len = float(np.mean(lengths))
        std_len = float(np.std(lengths))

        # -------------------------
        # OUTPUT
        # -------------------------

        return {
            "tok_length_log": self._safe_unbounded(length_log),
            "tok_vocab_log": self._safe_unbounded(vocab_log),

            "tok_entropy": self._safe(entropy),
            "tok_topk_mass": self._safe(topk_mass),

            "tok_repetition_strength": self._safe(repetition),
            "tok_gini": self._safe(gini),

            "tok_avg_length": self._safe_unbounded(avg_len),
            "tok_std_length": self._safe_unbounded(std_len),
        }

    # -----------------------------------------------------

    def _empty(self) -> Dict[str, float]:
        return {
            "tok_length_log": 0.0,
            "tok_vocab_log": 0.0,
            "tok_entropy": 0.0,
            "tok_topk_mass": 0.0,
            "tok_repetition_strength": 0.0,
            "tok_gini": 0.0,
            "tok_avg_length": 0.0,
            "tok_std_length": 0.0,
        }

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))

    def _safe_unbounded(self, v: float) -> float:
        """Drop NaN / negative values but do NOT clip the upper bound.

        Audit fix §1.1 — see :class:`SyntacticFeatures` for the full
        rationale. Raw magnitudes flow through to the scaling stage.
        """
        if not np.isfinite(v) or v < 0:
            return 0.0
        return float(v)