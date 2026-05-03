from __future__ import annotations

import logging
from typing import Dict, Set

import numpy as np

from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.feature_context import FeatureContext
from src.analysis._text_features import (
    term_ratio,
    phrase_match_count,
    normalize_lexicon_terms,
)
from src.analysis.feature_schema import INFORMATION_OMISSION_KEYS, make_vector

logger = logging.getLogger(__name__)


# =========================================================
# CONSTANTS
# =========================================================

EPS = 1e-8
MAX_CLIP = 1.0


# =========================================================
# ANALYZER
# =========================================================

class InformationOmissionDetector(BaseAnalyzer):

    name = "information_omission"
    expected_keys = set(INFORMATION_OMISSION_KEYS)

    # -----------------------------------------------------
    # LEXICONS (KEEP FULL SETS)
    # -----------------------------------------------------

    COUNTERARGUMENT_MARKERS: Set[str] = {...}
    EVIDENCE_MARKERS: Set[str] = {...}
    CLAIM_MARKERS: Set[str] = {...}
    FRAMING_MARKERS: Set[str] = {...}

    # =========================================================

    def __init__(self):

        self.counter = normalize_lexicon_terms(self.COUNTERARGUMENT_MARKERS)
        self.evidence = normalize_lexicon_terms(self.EVIDENCE_MARKERS)
        self.claim = normalize_lexicon_terms(self.CLAIM_MARKERS)
        self.framing = normalize_lexicon_terms(self.FRAMING_MARKERS)

        logger.info("InformationOmissionDetector initialized (final)")

    # =========================================================

    def analyze(self, ctx: FeatureContext) -> Dict[str, float]:

        # 🔥 ensure lazy context
        ctx.ensure_tokens()

        if ctx.safe_n_tokens() == 0:
            return self._empty()

        # -----------------------------------------------------
        # DENSITIES
        # -----------------------------------------------------

        counter = self._density(ctx, self.counter)
        evidence = self._density(ctx, self.evidence)
        claim = self._density(ctx, self.claim)
        framing = self._density(ctx, self.framing)

        # -----------------------------------------------------
        # RELATIVE NORMALIZATION
        # -----------------------------------------------------

        total = counter + evidence + claim + framing + EPS

        counter_n = counter / total
        evidence_n = evidence / total
        claim_n = claim / total
        framing_n = framing / total

        # -----------------------------------------------------
        # FEATURES
        # -----------------------------------------------------

        # absence of counterarguments
        missing_counter = 1.0 - counter_n

        # one-sided narrative (bounded logistic form)
        one_sided_raw = (claim_n + framing_n) / (counter_n + EPS)
        one_sided = one_sided_raw / (1.0 + one_sided_raw)

        # lack of evidence
        incomplete_evidence = 1.0 - evidence_n

        # claim vs evidence imbalance
        claim_vs_evidence = claim_n / (claim_n + evidence_n + EPS)

        # -----------------------------------------------------
        # OUTPUT
        # -----------------------------------------------------

        return {
            "missing_counterargument_score": self._safe(missing_counter),
            "one_sided_framing_score": self._safe(one_sided),
            "incomplete_evidence_score": self._safe(incomplete_evidence),
            "claim_evidence_imbalance": self._safe(claim_vs_evidence),
        }

    # =========================================================

    def _density(self, ctx: FeatureContext, lexicon: Set[str]) -> float:

        n_tokens = ctx.safe_n_tokens()

        # token signal
        token_score = term_ratio(
            ctx.safe_counts(),
            n_tokens,
            lexicon,
        )

        # phrase signal
        phrase_hits = phrase_match_count(
            ctx.text_lower or "",
            lexicon,
        )

        phrase_score = phrase_hits / (n_tokens + EPS)

        # 🔥 weighted fusion (prevents double counting)
        combined = 0.7 * token_score + 0.3 * phrase_score

        return self._safe(combined)

    # =========================================================

    def _safe(self, value: float) -> float:

        if not np.isfinite(value):
            return 0.0

        return float(np.clip(value, 0.0, MAX_CLIP))

    # =========================================================

    def _empty(self) -> Dict[str, float]:

        return {
            "missing_counterargument_score": 0.0,
            "one_sided_framing_score": 0.0,
            "incomplete_evidence_score": 0.0,
            "claim_evidence_imbalance": 0.0,
        }


# =========================================================
# VECTOR CONVERSION
# =========================================================

def information_omission_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, INFORMATION_OMISSION_KEYS)