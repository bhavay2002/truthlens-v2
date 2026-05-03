
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.features.base.base_feature import FeatureContext
from src.features.emotion.emotion_lexicon_features import EmotionLexiconFeatures
from src.features.emotion.emotion_schema import EMOTION_LABELS


# =========================================================
# RESULT
# =========================================================

@dataclass
class EmotionResult:
    dominant_emotion: str
    confidence: float
    emotion_scores: Dict[str, float]

    # advanced signals
    emotion_entropy: float
    emotion_intensity: float
    emotion_coverage: float


# =========================================================
# ANALYZER
# =========================================================

class EmotionLexiconAnalyzer:

    def __init__(self) -> None:
        self._extractor = EmotionLexiconFeatures()

    # -----------------------------------------------------

    def analyze(self, text: str) -> EmotionResult:

        if not isinstance(text, str) or not text.strip():
            return self._empty_result()

        context = FeatureContext(text=text)
        features = self._extractor.extract(context)

        # -------------------------
        # Distribution
        # -------------------------

        emotion_scores: Dict[str, float] = {
            emotion: float(features.get(f"lexicon_emotion_{emotion}", 0.0))
            for emotion in EMOTION_LABELS
        }

        values = np.array(list(emotion_scores.values()), dtype=np.float32)

        # -------------------------
        # Dominant emotion
        # -------------------------

        if values.sum() == 0:
            dominant = "neutral"
            confidence = 0.0
        else:
            idx = int(np.argmax(values))
            dominant = EMOTION_LABELS[idx]
            confidence = float(values[idx])

        # -------------------------
        # Additional signals
        # -------------------------

        entropy = float(features.get("lexicon_emotion_entropy", 0.0))
        intensity = float(features.get("lexicon_emotion_intensity_l2", 0.0))
        coverage = float(features.get("lexicon_emotion_coverage", 0.0))

        return EmotionResult(
            dominant_emotion=dominant,
            confidence=round(confidence, 4),
            emotion_scores={
                k: round(v, 4) for k, v in emotion_scores.items()
            },
            emotion_entropy=round(entropy, 4),
            emotion_intensity=round(intensity, 4),
            emotion_coverage=round(coverage, 4),
        )

    # -----------------------------------------------------

    def _empty_result(self) -> EmotionResult:
        return EmotionResult(
            dominant_emotion="neutral",
            confidence=0.0,
            emotion_scores={e: 0.0 for e in EMOTION_LABELS},
            emotion_entropy=0.0,
            emotion_intensity=0.0,
            emotion_coverage=0.0,
        )