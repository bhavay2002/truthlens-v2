# src/features/emotion/emotion_features.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature
from src.features.base.lexicon_matcher import LexiconMatcher, to_token_array
from src.features.base.numerics import normalized_entropy
from src.features.base.tokenization import ensure_tokens_word

from src.features.emotion.emotion_schema import (
    EMOTION_LABELS,
    EMOTION_TERMS,
)

logger = logging.getLogger(__name__)

EPS = 1e-8
MAX_CLIP = 1.0


# -------------------------------------------------------
# Reverse lookup
# -------------------------------------------------------

WORD_TO_EMOTION = {
    word: emotion
    for emotion, words in EMOTION_TERMS.items()
    for word in words
}


# -------------------------------------------------------
# Emotion groups (OPTIONAL BUT IMPORTANT)
# -------------------------------------------------------

POSITIVE_EMOTIONS = {
    "joy", "trust", "love", "optimism"
}

NEGATIVE_EMOTIONS = {
    "anger", "fear", "sadness", "disgust"
}


# -------------------------------------------------------
# Vectorized matchers (one per emotion label, built once)
# -------------------------------------------------------

_EMOTION_MATCHERS = {
    emotion: LexiconMatcher(EMOTION_TERMS.get(emotion, ()), name=emotion)
    for emotion in EMOTION_LABELS
}


# -------------------------------------------------------
# Lexicon detector (vectorized)
# -------------------------------------------------------

def _lexicon_emotions(tokens):

    tokens_arr = to_token_array(tokens)

    counts = {
        emotion: _EMOTION_MATCHERS[emotion].count_in_tokens(tokens_arr)
        for emotion in EMOTION_LABELS
    }

    total_hits = sum(counts.values())
    total_tokens = len(tokens)

    return counts, total_hits, total_tokens


# -------------------------------------------------------
# Feature extractor
# -------------------------------------------------------

@dataclass
@register_feature
class EmotionFeatures(BaseFeature):

    name: str = "emotion_features"
    group: str = "emotion"

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return {}

        tokens = ensure_tokens_word(context, text)
        counts, total_hits, total_tokens = _lexicon_emotions(tokens)

        if total_tokens == 0:
            return {}

        # -------------------------
        # DISTRIBUTION (normalized)
        # -------------------------

        values = np.array([counts[e] for e in EMOTION_LABELS], dtype=np.float32)

        if total_hits == 0:
            dist = np.zeros_like(values)
        else:
            dist = values / (total_hits + EPS)

        # -------------------------
        # COVERAGE (CRITICAL)
        # -------------------------

        coverage = total_hits / (total_tokens + EPS)

        # -------------------------
        # ENTROPY
        # -------------------------

        entropy = normalized_entropy(dist)

        # -------------------------
        # INTENSITY (FIXED)
        # -------------------------

        intensity = float(np.linalg.norm(dist))  # stable

        # -------------------------
        # POLARITY
        # -------------------------

        pos = sum(dist[EMOTION_LABELS.index(e)] for e in POSITIVE_EMOTIONS if e in EMOTION_LABELS)
        neg = sum(dist[EMOTION_LABELS.index(e)] for e in NEGATIVE_EMOTIONS if e in EMOTION_LABELS)

        # Defensive clamp: ``dist`` is normalised by ``total_hits`` which is
        # the sum of *all* emotion counts including labels in neither
        # ``POSITIVE_EMOTIONS`` nor ``NEGATIVE_EMOTIONS`` (e.g. ``surprise``,
        # ``anticipation``). That keeps ``pos - neg`` in ``[-1, 1]`` in
        # theory, but float accumulation can push the result a few ULPs
        # outside that range, which then escapes the ``[0, 1]`` invariant
        # of the ``(polarity + 1) / 2`` remap below. Clip first.
        polarity = float(np.clip(pos - neg, -1.0, 1.0))

        # -------------------------
        # OUTPUT
        # -------------------------
        # Emotion is multi-label by design: each `emotion_<label>` is a
        # per-label scalar (the share of total emotion-token hits that
        # match that label). The previous one-hot
        # ``emotion_dominant_<label>`` column was removed (audit task 3):
        # it discarded all but one label per article and was redundant
        # with argmax over the per-label columns at inference time.

        features: Dict[str, float] = {}

        for i, emotion in enumerate(EMOTION_LABELS):
            features[f"emotion_{emotion}"] = self._safe(dist[i])

        features.update({
            "emotion_coverage": self._safe(coverage),
            "emotion_intensity": self._safe(intensity),
            "emotion_entropy": self._safe(entropy),
            "emotion_polarity": self._safe((polarity + 1.0) / 2.0),  # normalize to [0,1]
        })

        return features

    # -----------------------------------------------------

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))

    # -----------------------------------------------------

    def extract_batch(self, contexts):
        return [self.extract(ctx) for ctx in contexts]