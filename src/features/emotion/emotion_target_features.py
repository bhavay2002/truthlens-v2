# src/features/emotion/emotion_target_features.py

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Any

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature
from src.features.base.numerics import EPS, MAX_CLIP, normalized_entropy
from src.features.base.spacy_loader import get_shared_nlp
from src.features.base.tokenization import ensure_tokens_word
from src.features.emotion.emotion_schema import WORD_TO_EMOTION

logger = logging.getLogger(__name__)


# -----------------------------------------------------
# Target categories
# -----------------------------------------------------

FIRST_PERSON = {"i", "me", "my", "mine", "we", "our", "us"}
SECOND_PERSON = {"you", "your", "yours"}
THIRD_PERSON = {"he", "she", "they", "them", "their", "his", "her", "its"}


# -----------------------------------------------------
# Feature extractor
# -----------------------------------------------------

@dataclass
@register_feature
class EmotionTargetFeatures(BaseFeature):

    name: str = "emotion_target_features"
    group: str = "emotion"
    description: str = "Emotion target attribution (self/other/entity/group)"

    _nlp: Any = field(default=None, init=False, repr=False)
    _spacy_available: bool = field(default=False, init=False, repr=False)

    # -------------------------------------------------

    def initialize(self) -> None:
        if self._nlp is not None or self._spacy_available:
            return
        self._nlp = get_shared_nlp("en_core_web_sm")
        self._spacy_available = self._nlp is not None

    # -------------------------------------------------
    # spaCy-based (IMPROVED)
    # -------------------------------------------------

    def _extract_spacy(self, text: str) -> Dict[str, float]:

        doc = self._nlp(text)

        counts = {
            "self": 0.0,
            "other": 0.0,
            "entity": 0.0,
            "group": 0.0,
        }

        total_emotions = 0

        for token in doc:
            tok = token.text.lower()

            if tok not in WORD_TO_EMOTION:
                continue

            total_emotions += 1

            # 🔥 dependency-aware neighbors
            related = list(token.children)
            if token.head is not None:
                related.append(token.head)

            for r in related:
                t = r.text.lower()

                if t in FIRST_PERSON:
                    counts["self"] += 1.0

                elif t in SECOND_PERSON or t in THIRD_PERSON:
                    counts["other"] += 1.0

                # entity (PERSON/ORG)
                if r.ent_type_ in {"PERSON", "ORG"}:
                    counts["entity"] += 1.0

                # group (plural + noun chunk)
                if r.pos_ == "NOUN" and r.tag_ == "NNS":
                    counts["group"] += 0.5  # softer weight

        total_emotions = max(total_emotions, 1)

        # -------------------------
        # Normalize
        # -------------------------

        ratios = {
            k: v / total_emotions for k, v in counts.items()
        }

        values = np.array(list(ratios.values()), dtype=np.float32)

        # -------------------------
        # Density (aligned)
        # -------------------------

        density = total_emotions / max(len(doc), 1)

        # -------------------------
        # Entropy (NEW)
        # -------------------------

        entropy = normalized_entropy(values)

        return {
            "emotion_target_self_ratio": self._safe(ratios["self"]),
            "emotion_target_other_ratio": self._safe(ratios["other"]),
            "emotion_target_entity_ratio": self._safe(ratios["entity"]),
            "emotion_target_group_ratio": self._safe(ratios["group"]),

            "emotion_target_density": self._safe(density),
            "emotion_target_entropy": self._safe(entropy),
        }

    # -------------------------------------------------
    # Fallback (IMPROVED)
    # -------------------------------------------------

    def _extract_fallback(self, context: FeatureContext) -> Dict[str, float]:

        tokens = ensure_tokens_word(context)

        if not tokens:
            return self._empty()

        counts = {"self": 0, "other": 0}
        emotion_positions = [
            i for i, t in enumerate(tokens) if t in WORD_TO_EMOTION
        ]

        for pos in emotion_positions:
            window = tokens[max(0, pos - 3): pos + 4]

            for w in window:
                if w in FIRST_PERSON:
                    counts["self"] += 1
                elif w in SECOND_PERSON or w in THIRD_PERSON:
                    counts["other"] += 1

        total_emotions = max(len(emotion_positions), 1)

        ratios = {
            "self": counts["self"] / total_emotions,
            "other": counts["other"] / total_emotions,
        }

        density = len(emotion_positions) / max(len(tokens), 1)

        return {
            "emotion_target_self_ratio": self._safe(ratios["self"]),
            "emotion_target_other_ratio": self._safe(ratios["other"]),
            "emotion_target_entity_ratio": 0.0,
            "emotion_target_group_ratio": 0.0,
            "emotion_target_density": self._safe(density),
            "emotion_target_entropy": 0.0,
        }

    # -------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        if not isinstance(context.text, str):
            raise TypeError("FeatureContext.text must be a string")

        if not context.text.strip():
            return self._empty()

        self.initialize()

        if self._spacy_available:
            return self._extract_spacy(context.text)

        return self._extract_fallback(context)

    # -------------------------------------------------

    def _empty(self) -> Dict[str, float]:
        return {
            "emotion_target_self_ratio": 0.0,
            "emotion_target_other_ratio": 0.0,
            "emotion_target_entity_ratio": 0.0,
            "emotion_target_group_ratio": 0.0,
            "emotion_target_density": 0.0,
            "emotion_target_entropy": 0.0,
        }

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))