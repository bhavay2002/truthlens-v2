from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Set

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature
from src.features.base.numerics import EPS, MAX_CLIP, normalized_entropy
from src.features.base.tokenization import ensure_tokens_word, ensure_tokens_word_counter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Lexicons
# ---------------------------------------------------------

CAUSAL = {"because","since","therefore","thus","hence","consequently"}
CONTRAST = {"however","but","although","though","nevertheless","yet"}
ADDITIVE = {"also","furthermore","moreover","additionally","besides"}
SEQUENTIAL = {"first","second","then","next","finally"}
EVIDENTIAL = {"according","reported","evidence","study","data","research"}


# ---------------------------------------------------------
# Feature
# ---------------------------------------------------------

@dataclass
@register_feature
class DiscourseFeatures(BaseFeature):

    name: str = "discourse_features"
    group: str = "discourse"
    description: str = "Normalized discourse structure features"

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return self._empty()

        tokens = ensure_tokens_word(context, text)
        n = len(tokens)

        if n == 0:
            return self._empty()

        # Audit fix §2.5 — read the per-context cached Counter instead
        # of materialising a fresh one. Eight extractors used to call
        # ``Counter(tokens)`` on the same token list per request.
        counter = ensure_tokens_word_counter(context)

        def ratio(lexicon: Set[str]) -> float:
            return sum(counter.get(w, 0) for w in lexicon) / (n + EPS)

        raw = {
            "causal": ratio(CAUSAL),
            "contrast": ratio(CONTRAST),
            "additive": ratio(ADDITIVE),
            "sequential": ratio(SEQUENTIAL),
            "evidential": ratio(EVIDENTIAL),
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
        # INTENSITY
        # -------------------------

        intensity = float(np.linalg.norm(values))

        # -------------------------
        # ENTROPY (CRITICAL)
        # -------------------------

        entropy = normalized_entropy(probs)

        # -------------------------
        # BALANCE
        # -------------------------

        balance = 1.0 - float(np.std(probs))

        # -------------------------
        # OUTPUT
        # -------------------------

        # §10.1 — keys renamed to match feature_schema.DISCOURSE_FEATURES so
        # partition_feature_sections() routes them to the "discourse" head
        # instead of dropping them into "other".
        return {
            "discourse_causal_ratio":     self._safe(dist["causal"]),
            "discourse_contrast_ratio":   self._safe(dist["contrast"]),
            "discourse_additive_ratio":   self._safe(dist["additive"]),
            "discourse_sequential_ratio": self._safe(dist["sequential"]),
            "discourse_evidential_ratio": self._safe(dist["evidential"]),
            # intensity → marker_density; entropy → diversity; balance dropped.
            "discourse_marker_density":   self._safe(intensity),
            "discourse_diversity":        self._safe(entropy),
        }

    # -----------------------------------------------------

    def _empty(self) -> Dict[str, float]:
        # §11.1 — return a fixed-key zero dict so the schema validator
        # sees a consistent shape on empty-text inputs.
        return {
            "discourse_causal_ratio":     0.0,
            "discourse_contrast_ratio":   0.0,
            "discourse_additive_ratio":   0.0,
            "discourse_sequential_ratio": 0.0,
            "discourse_evidential_ratio": 0.0,
            "discourse_marker_density":   0.0,
            "discourse_diversity":        0.0,
        }

    # -----------------------------------------------------

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))