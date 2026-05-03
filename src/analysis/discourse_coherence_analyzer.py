from __future__ import annotations

import logging
from typing import Dict, Set, List

import numpy as np

from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.feature_context import FeatureContext
from src.analysis._text_features import (
    phrase_match_count,
    cached_phrase_match_count,
    normalize_lexicon_terms,
)
from src.analysis.feature_schema import DISCOURSE_COHERENCE_KEYS, make_vector
from src.analysis.spacy_loader import get_doc

logger = logging.getLogger(__name__)


# =========================================================
# CONSTANTS
# =========================================================

EPS = 1e-8
MAX_CLIP = 1.0


# =========================================================
# ANALYZER
# =========================================================

class DiscourseCoherenceAnalyzer(BaseAnalyzer):

    name = "discourse_coherence"
    expected_keys = set(DISCOURSE_COHERENCE_KEYS)

    TRANSITION_MARKERS = {
        "however","nevertheless","nonetheless",
        "yet","still","though","although",
        "in contrast","on the other hand",
        "therefore","thus","hence",
        "consequently","as a result",
        "because","since","due to",
        "furthermore","moreover","additionally",
        "also","besides","similarly",
        "first","second","next","then",
        "finally","meanwhile",
        "in conclusion","overall",
        "in summary","ultimately",
        "for example","for instance"
    }

    def __init__(self):
        self.transition_phrases = normalize_lexicon_terms(self.TRANSITION_MARKERS)

    # =========================================================

    def analyze(self, ctx: FeatureContext) -> Dict[str, float]:

        # Section 4: defensive safe accessor (BaseAnalyzer pre-warms,
        # but we never want to compare None == 0).
        if ctx.safe_n_tokens() == 0:
            return self._empty_features()

        # 🔥 shared spaCy doc
        doc = get_doc(ctx, task="syntax")
        sentences = list(doc.sents)

        if len(sentences) < 2:
            return self._empty_features()

        # -----------------------------------------------------
        # SENTENCE TOKEN SETS  (PERF-A7)
        # -----------------------------------------------------
        # Cache the per-sentence lemma sets on the shared dict. They're
        # O(sentences * tokens) to allocate; reusing them across re-runs
        # of this analyzer (and other future consumers) saves the work.
        sent_tokens = ctx.shared.get("disc_sent_lemmas")
        if sent_tokens is None:
            sent_tokens = [self._sentence_token_set(sent) for sent in sentences]
            ctx.shared["disc_sent_lemmas"] = sent_tokens

        # -----------------------------------------------------
        # LOCAL COHERENCE (adjacent similarity)
        # -----------------------------------------------------

        local_sim = [
            self._safe_jaccard(sent_tokens[i], sent_tokens[i + 1])
            for i in range(len(sent_tokens) - 1)
        ]

        mean_local = float(np.mean(local_sim)) if local_sim else 0.0

        # -----------------------------------------------------
        # GLOBAL COHERENCE (start vs end)
        # -----------------------------------------------------

        global_score = self._safe_jaccard(
            sent_tokens[0], sent_tokens[-1]
        )

        # -----------------------------------------------------
        # SMOOTHED COHERENCE
        # -----------------------------------------------------

        coherence = 0.7 * mean_local + 0.3 * global_score

        # nonlinear smoothing (important)
        coherence = float(np.sqrt(max(coherence, 0.0)))

        features = {
            "sentence_coherence": self._safe(coherence),
            "topic_drift": self._safe(1.0 - coherence),
        }

        # -----------------------------------------------------
        # NARRATIVE CONTINUITY (IMPROVED)
        # -----------------------------------------------------

        features.update(self._narrative_continuity(doc))

        # -----------------------------------------------------
        # TRANSITION SIGNAL
        # -----------------------------------------------------

        features["discourse_transition_ratio"] = self._transition_ratio(
            ctx,
            ctx.safe_n_tokens(),
        )

        return features

    # =========================================================

    def _sentence_token_set(self, sent) -> Set[str]:
        return {
            token.lemma_.lower()
            for token in sent
            if token.is_alpha and not token.is_stop
        }

    # =========================================================

    def _safe_jaccard(self, a: Set[str], b: Set[str]) -> float:

        if not a and not b:
            return 0.0

        denom = len(a | b)
        if denom == 0:
            return 0.0

        return self._safe(len(a & b) / denom)

    # =========================================================

    def _narrative_continuity(self, doc) -> Dict[str, float]:

        entities = [ent.text.lower() for ent in doc.ents]

        # F14: emit BOTH the new canonical key
        # ``entity_repetition_ratio`` (which describes what the metric
        # actually measures: 1 − unique/total over named entities) and
        # the legacy ``narrative_continuity`` alias so existing
        # consumers keep working. Both carry the same value.
        if not entities:
            return {
                "narrative_continuity": 0.0,
                "entity_repetition_ratio": 0.0,
            }

        total = len(entities)
        unique = len(set(entities))

        repetition = 1.0 - (unique / (total + EPS))

        continuity = float(np.sqrt(max(repetition, 0.0)))
        value = self._safe(continuity)

        return {
            "narrative_continuity": value,
            "entity_repetition_ratio": value,
        }

    # =========================================================

    def _transition_ratio(self, ctx: FeatureContext, n_tokens: int) -> float:

        if n_tokens <= 0:
            return 0.0

        # PERF-A2: shared per-ctx phrase-hit cache.
        hits = cached_phrase_match_count(ctx, self.transition_phrases)

        return self._safe(hits / (n_tokens + EPS))

    # =========================================================

    def _safe(self, value: float) -> float:

        if not np.isfinite(value):
            return 0.0

        return float(np.clip(value, 0.0, MAX_CLIP))

    # =========================================================

    def _empty_features(self) -> Dict[str, float]:

        return {k: 0.0 for k in DISCOURSE_COHERENCE_KEYS}


# =========================================================
# VECTOR CONVERSION
# =========================================================

def discourse_coherence_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, DISCOURSE_COHERENCE_KEYS)