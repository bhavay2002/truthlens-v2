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

CLAIM_MARKERS = {"therefore","thus","clearly","obviously","conclude","shows"}
PREMISE_MARKERS = {"because","since","given","as","assuming"}
EVIDENCE_MARKERS = {"evidence","study","data","report","research","analysis"}
COUNTERARGUMENT_MARKERS = {"however","although","but","nevertheless","yet"}

INTERROGATIVES = {"why","how","what","who"}


# ---------------------------------------------------------
# Feature
# ---------------------------------------------------------

@dataclass
@register_feature
class ArgumentStructureFeatures(BaseFeature):

    name: str = "argument_structure_features"
    group: str = "argument"
    description: str = "Normalized argument structure features"

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

        def ratio(lexicon: Set[str]) -> float:
            return sum(counter.get(w, 0) for w in lexicon) / (n + EPS)

        raw = {
            "claim": ratio(CLAIM_MARKERS),
            "premise": ratio(PREMISE_MARKERS),
            "evidence": ratio(EVIDENCE_MARKERS),
            "counter": ratio(COUNTERARGUMENT_MARKERS),
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
        # INTENSITY (ARGUMENT STRENGTH)
        # -------------------------

        intensity = float(np.linalg.norm(values))

        # -------------------------
        # ENTROPY (CRITICAL)
        # -------------------------

        entropy = normalized_entropy(probs)

        # -------------------------
        # RHETORICAL QUESTIONS (FIXED)
        # -------------------------

        question_marks = text.count("?")
        interrogative_hits = sum(counter.get(w, 0) for w in INTERROGATIVES)

        rhetorical = (question_marks + interrogative_hits) / (n + EPS)

        # -------------------------
        # ARGUMENT BALANCE
        # -------------------------

        balance = 1.0 - float(np.std(probs))

        # -------------------------
        # OUTPUT
        # -------------------------

        # §10.1 — keys renamed to match feature_schema.ARGUMENT_FEATURES so
        # partition_feature_sections() routes them to the "discourse" head.
        return {
            "argument_claim_ratio":              self._safe(dist["claim"]),
            "argument_premise_ratio":            self._safe(dist["premise"]),
            "argument_evidence_ratio":           self._safe(dist["evidence"]),
            "argument_counterargument_ratio":    self._safe(dist["counter"]),
            # intensity → structure_density; balance → structure_diversity;
            # entropy dropped (not in schema).
            "argument_structure_density":        self._safe(intensity),
            "argument_structure_diversity":      self._safe(balance),
            "argument_rhetorical_question_ratio": self._safe(rhetorical),
        }

    # -----------------------------------------------------

    def _empty(self) -> Dict[str, float]:
        # §11.1 — consistent fixed-key zero dict for empty / zero-token inputs.
        return {
            "argument_claim_ratio":              0.0,
            "argument_premise_ratio":            0.0,
            "argument_evidence_ratio":           0.0,
            "argument_counterargument_ratio":    0.0,
            "argument_structure_density":        0.0,
            "argument_structure_diversity":      0.0,
            "argument_rhetorical_question_ratio": 0.0,
        }

    # -----------------------------------------------------

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))