# src/features/emotion/emotion_trajectory_features.py

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature
from src.features.base.lexicon_matcher import LexiconMatcher, to_token_array
from src.features.base.tokenization import ensure_tokens_word, tokenize_words

from src.features.base.numerics import EPS, MAX_CLIP

from src.features.emotion.emotion_schema import (
    EMOTION_LABELS,
    WORD_TO_EMOTION,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Sentence splitter — audit fix §4
# ------------------------------------------------------------
# The local ``_SENT_SPLIT_RE`` / ``_split_sentences`` pair was duplicated
# across syntactic / graph / trajectory extractors and they had subtly
# different boundary behaviour. The canonical splitter now lives in
# ``src.features.base.segmentation`` and is shared by all four sites.
from src.features.base.segmentation import split_sentences as _split_sentences  # noqa: E402


# ------------------------------------------------------------
# Per-emotion vectorized matchers (built once at import)
# ------------------------------------------------------------
# Audit fix §2.1 + §2.4 — replace the per-sentence ``for token in ...:
# WORD_TO_EMOTION.get(t)`` Python loop with a precompiled
# :class:`LexiconMatcher` per emotion label. ``negation_aware_sum`` is
# overkill here (no negation modelling for trajectories), so we use the
# unweighted ``count_in_tokens`` path which is a single ``np.isin`` per
# emotion bucket.

def _build_emotion_matchers() -> List[LexiconMatcher]:
    by_label: Dict[str, List[str]] = {label: [] for label in EMOTION_LABELS}
    for word, label in WORD_TO_EMOTION.items():
        if label in by_label:
            by_label[label].append(word)
    return [
        LexiconMatcher(by_label[label], name=label)
        for label in EMOTION_LABELS
    ]


_EMOTION_MATCHERS: List[LexiconMatcher] = _build_emotion_matchers()


# ------------------------------------------------------------
# Per-sentence emotion vector
# ------------------------------------------------------------

def _sentence_vector(sentence_tokens: List[str]) -> np.ndarray:
    """Return the L1-normalised emotion distribution for a sentence."""
    vec = np.zeros(len(EMOTION_LABELS), dtype=np.float32)
    if not sentence_tokens:
        return vec

    arr = to_token_array(sentence_tokens)
    for j, matcher in enumerate(_EMOTION_MATCHERS):
        vec[j] = matcher.count_in_tokens(arr)

    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


# ------------------------------------------------------------
# Feature extractor
# ------------------------------------------------------------

@dataclass
@register_feature
class EmotionTrajectoryFeatures(BaseFeature):

    name: str = "emotion_trajectory_features"
    group: str = "emotion"
    description: str = "Emotion trajectory modeling (vector-based)"

    # -----------------------------------------------------

    def _segment_vectors(self, context: FeatureContext) -> List[np.ndarray]:
        """Split the document into sentences and return one vector per sentence.

        Audit fix §2.4 — eliminate per-sentence re-tokenization with a
        bare ``\\w+`` regex. We tokenize each sentence through the
        canonical :func:`tokenize_words` helper (Unicode-aware,
        contractions handled) so the matchers see the same tokens as
        the rest of the pipeline. The whole-document tokenization on
        ``context.tokens_word`` is still warmed by upstream extractors;
        we cannot reuse it directly here because trajectory analysis
        needs per-sentence boundaries.
        """
        text = context.text or ""
        sentences = _split_sentences(text)
        if not sentences:
            # Single zero vector keeps the rest of the pipeline well-defined
            # (np.stack on a single vector still works).
            return [np.zeros(len(EMOTION_LABELS), dtype=np.float32)]

        # Make sure the document-level token cache exists for any
        # downstream extractor that runs after us in the same context.
        ensure_tokens_word(context, text)

        return [_sentence_vector(tokenize_words(s)) for s in sentences]

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return self._empty()

        vectors = self._segment_vectors(context)

        # §11.6 — track whether we had only one sentence BEFORE duplicating.
        # When the document is a single sentence the slope / volatility /
        # shift_mean are all trivially zero (identical vector duplicated).
        # Emitting ``emotion_traj_single_sentence=1.0`` lets the downstream
        # model attenuate the trajectory signal on those rows instead of
        # learning a spurious "single-sentence document" pattern from the
        # bimodal cliff in the other columns.
        single_sentence = len(vectors) == 1
        if single_sentence:
            vectors.append(vectors[0])

        mat = np.stack(vectors)  # shape: (T, E)

        # -------------------------
        # Intensity trajectory
        # -------------------------

        intensities = np.linalg.norm(mat, axis=1)

        # -------------------------
        # Core stats
        # -------------------------

        mean_val = float(np.mean(intensities))
        std_val = float(np.std(intensities))

        # normalized slope
        x = np.linspace(0, 1, len(intensities))
        slope = float(np.polyfit(x, intensities, 1)[0])

        # peak (smoothed)
        peak_idx = int(np.argmax(intensities))
        peak_position = peak_idx / max(len(intensities) - 1, 1)

        volatility = float(np.mean(np.abs(np.diff(intensities))))

        range_val = float(np.max(intensities) - np.min(intensities))

        # -------------------------
        # NEW: Distribution shift
        # -------------------------

        shifts = [
            np.linalg.norm(mat[i] - mat[i - 1])
            for i in range(1, len(mat))
        ]

        shift_mean = float(np.mean(shifts)) if shifts else 0.0

        # -------------------------
        # NEW: Entropy over time
        # -------------------------

        entropies = []
        for v in mat:
            if v.sum() > 0:
                e = -np.sum(v * np.log(v + EPS))
                e /= np.log(len(v))
                entropies.append(e)
            else:
                entropies.append(0.0)

        entropy_mean = float(np.mean(entropies))

        # -------------------------
        # Output (bounded)
        # -------------------------

        return {
            "emotion_traj_mean": self._safe(mean_val),
            "emotion_traj_std": self._safe(std_val),
            "emotion_traj_slope": self._safe((slope + 1) / 2),  # normalize
            "emotion_traj_peak_position": self._safe(peak_position),
            "emotion_traj_volatility": self._safe(volatility),
            "emotion_traj_range": self._safe(range_val),

            # advanced signals
            "emotion_traj_shift_mean": self._safe(shift_mean),
            "emotion_traj_entropy_mean": self._safe(entropy_mean),

            # §11.6 — availability indicator: 1.0 when the document was a
            # single sentence (trajectory stats are degenerate in that case).
            "emotion_traj_single_sentence": 1.0 if single_sentence else 0.0,
        }

    # -----------------------------------------------------

    def _empty(self) -> Dict[str, float]:
        return {
            "emotion_traj_mean": 0.0,
            "emotion_traj_std": 0.0,
            "emotion_traj_slope": 0.0,
            "emotion_traj_peak_position": 0.0,
            "emotion_traj_volatility": 0.0,
            "emotion_traj_range": 0.0,
            "emotion_traj_shift_mean": 0.0,
            "emotion_traj_entropy_mean": 0.0,
            # §11.6 — availability indicator (1.0 = degenerate / unavailable).
            "emotion_traj_single_sentence": 1.0,
        }

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))
