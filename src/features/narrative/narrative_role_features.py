from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Any

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature
from src.features.base.lexicon_loader import load_lexicon_set
from src.features.base.lexicon_matcher import LexiconMatcher, to_token_array
from src.features.base.numerics import EPS, MAX_CLIP, normalized_entropy
from src.features.base.spacy_doc import ensure_spacy_doc
from src.features.base.spacy_loader import get_shared_nlp
from src.features.base.tokenization import ensure_tokens_word

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Lexicons — audit fix §1.1, see src/config/lexicons/narrative.json.
#
# Previously declared inline as ``{...}`` (a single-element set whose
# only member is the ``Ellipsis`` sentinel), which made every ratio
# permanently zero. They now share the same JSON source as
# ``narrative_features.py`` so the two extractors agree on the
# lexicon and a calibration sweep updates both at once.
# ---------------------------------------------------------

HERO_TERMS = load_lexicon_set("narrative", "hero")
VILLAIN_TERMS = load_lexicon_set("narrative", "villain")
VICTIM_TERMS = load_lexicon_set("narrative", "victim")
POLARIZATION_TERMS = load_lexicon_set("narrative", "polarization")


# ---------------------------------------------------------
# Vectorized matchers — audit fix §2.2.
# ---------------------------------------------------------

_ROLE_MATCHERS: Dict[str, LexiconMatcher] = {
    "hero":    LexiconMatcher(HERO_TERMS,    "narrative_role_hero"),
    "villain": LexiconMatcher(VILLAIN_TERMS, "narrative_role_villain"),
    "victim":  LexiconMatcher(VICTIM_TERMS,  "narrative_role_victim"),
}

_POLARIZATION_MATCHER = LexiconMatcher(
    POLARIZATION_TERMS, "narrative_role_polarization"
)


# ---------------------------------------------------------
# Feature
# ---------------------------------------------------------

@dataclass
@register_feature
class NarrativeRoleFeatures(BaseFeature):

    name: str = "narrative_role_features"
    group: str = "narrative"
    description: str = "Normalized narrative role modeling"

    _nlp: Any = field(default=None, init=False, repr=False)
    _spacy_available: bool = field(default=False, init=False, repr=False)

    # -----------------------------------------------------

    def initialize(self) -> None:
        if self._nlp is not None or self._spacy_available:
            return
        self._nlp = get_shared_nlp("en_core_web_sm")
        self._spacy_available = self._nlp is not None

    # -----------------------------------------------------

    def _entity_density(self, context: FeatureContext) -> float:
        """Return ``len(doc.ents) / len(doc)`` using the shared cache.

        Audit fix §2.7 — ``ensure_spacy_doc`` returns the per-context
        cached ``Doc`` if any other extractor in the same request has
        already parsed the text (typically the syntactic or graph
        extractor). Falls back to ``self._nlp(text)`` only when no
        cached doc exists, and seeds the cache for the next consumer.
        """
        self.initialize()

        if not self._spacy_available or self._nlp is None:
            return 0.0

        doc = ensure_spacy_doc(context)
        if doc is None:
            return 0.0
        # §11.3 — len(doc) counts ALL tokens including whitespace-only tokens,
        # inflating the denominator.  Filter to content tokens (non-space) so
        # the density reflects actual linguistic content in the document.
        content_token_count = sum(1 for t in doc if not t.is_space)
        return len(doc.ents) / max(content_token_count, 1)

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return self._empty()

        tokens = ensure_tokens_word(context, text)
        n = len(tokens)

        if n == 0:
            return self._empty()

        # Audit fix §2.2 — vectorised lexicon counts.
        tokens_arr = to_token_array(tokens)
        denom = n + EPS

        raw = {
            key: matcher.count_in_tokens(tokens_arr) / denom
            for key, matcher in _ROLE_MATCHERS.items()
        }

        polarization = _POLARIZATION_MATCHER.count_in_tokens(tokens_arr) / denom

        # -------------------------
        # ROLE DISTRIBUTION (CRITICAL)
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
        # ENTROPY
        # -------------------------

        entropy = normalized_entropy(probs)

        # -------------------------
        # BALANCE (FIXED)
        # -------------------------

        balance = 1.0 - float(np.std(probs))

        # -------------------------
        # DIVERSITY (WEIGHTED)
        # -------------------------

        diversity = float(np.count_nonzero(values) / len(values))

        # -------------------------
        # ENTITY SIGNAL
        # -------------------------

        entity_density = self._entity_density(context)

        # -------------------------
        # OUTPUT
        # -------------------------

        # Names MUST match src/features/feature_schema.py:NARRATIVE_FEATURES.
        # These are FEATURE names (model inputs) and are intentionally
        # distinct from the LABEL columns ("hero", "villain", "victim")
        # declared in data_contracts.CONTRACTS["narrative"].
        return {
            "narrative_role_hero_ratio": self._safe(dist["hero"]),
            "narrative_role_villain_ratio": self._safe(dist["villain"]),
            "narrative_role_victim_ratio": self._safe(dist["victim"]),

            "narrative_role_polarization_ratio": self._safe(polarization),

            "narrative_role_intensity": self._safe(intensity),
            "narrative_role_entropy": self._safe(entropy),

            "narrative_role_balance": self._safe(balance),
            "narrative_role_diversity": self._safe(diversity),

            "narrative_entity_density": self._safe(entity_density),
        }

    # -----------------------------------------------------

    def _empty(self) -> Dict[str, float]:
        # §11.1 — consistent fixed-key zero dict for empty / zero-token inputs.
        return {
            "narrative_role_hero_ratio":        0.0,
            "narrative_role_villain_ratio":     0.0,
            "narrative_role_victim_ratio":      0.0,
            "narrative_role_polarization_ratio": 0.0,
            "narrative_role_intensity":         0.0,
            "narrative_role_entropy":           0.0,
            "narrative_role_balance":           0.0,
            "narrative_role_diversity":         0.0,
            "narrative_entity_density":         0.0,
        }

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))
