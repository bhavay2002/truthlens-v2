from __future__ import annotations

import logging
import numbers
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np
from src.features.base.numerics import EPS

logger = logging.getLogger(__name__)

FeatureVector = Dict[str, float]

@dataclass
class FeatureSchemaValidator:
    """
    Production-grade schema validator for feature vectors.
    """

    expected_features: List[str]

    strict: bool = True
    allow_missing: bool = False
    allow_extra: bool = False

    #  NEW
    fill_value: float = 0.0
    return_numpy: bool = False

    _expected_set: Set[str] = field(init=False)
    _feature_index: Dict[str, int] = field(init=False)

    # -----------------------------------------------------

    def __post_init__(self) -> None:

        if not self.expected_features:
            raise ValueError("Expected feature schema cannot be empty")

        self._expected_set = set(self.expected_features)

        #  FAST INDEX
        self._feature_index = {
            f: i for i, f in enumerate(self.expected_features)
        }

        # Audit fix §5.1 — reject schemas that mix word-token and BPE
        # token denominators in the same vector. Mixing the two silently
        # produces statistically incomparable rates (one is per-Unicode
        # word, the other per subword piece) which corrupts every
        # downstream rate / density feature. We require the entire
        # schema to commit to a single token source so the contract is
        # checkable at extractor-output time, not after training.
        self._check_token_source_consistency(self.expected_features)

        logger.info(
            "FeatureSchemaValidator initialized | features=%d strict=%s",
            len(self.expected_features),
            self.strict,
        )

    @staticmethod
    def _check_token_source_consistency(features: List[str]) -> None:
        """Audit fix §5.1 — flag mixed ``tokens_word`` vs ``tokens_bpe``
        derived feature names in the same schema.

        Convention: feature keys denominated against the canonical
        Unicode word tokenization carry ``_word_`` (or end in
        ``_per_word``); features denominated against the HF subword
        tokens carry ``_bpe_`` (or ``_per_bpe``). Any extractor that
        emits both kinds in the same schema is almost certainly a bug
        (e.g. a copy-paste of two different rate formulas) and we fail
        loudly at construction time.
        """
        word_keys = [
            f for f in features
            if "_word_" in f or f.endswith("_per_word")
        ]
        bpe_keys = [
            f for f in features
            if "_bpe_" in f or f.endswith("_per_bpe")
        ]
        if word_keys and bpe_keys:
            raise ValueError(
                "Schema mixes word-token and BPE-token derived features; "
                f"word keys={word_keys[:5]!r} bpe keys={bpe_keys[:5]!r}. "
                "Pick a single token source per extractor."
            )

    # =====================================================
    # SINGLE VALIDATION
    # =====================================================

    def validate(self, features: FeatureVector) -> FeatureVector:

        if not isinstance(features, dict):
            raise TypeError("Features must be a dictionary")

        keys = set(features.keys())

        missing = self._expected_set - keys
        extra = keys - self._expected_set

        if missing and not self.allow_missing:
            raise ValueError(f"Missing required features: {sorted(missing)}")

        if extra and not self.allow_extra:
            raise ValueError(f"Unexpected extra features: {sorted(extra)}")

        validated: FeatureVector = {}

        for key in self.expected_features:

            value = features.get(key, self.fill_value)

            if not isinstance(value, numbers.Number):
                raise TypeError(f"Feature '{key}' must be numeric")

            value = float(value)

            #  FIX: NaN / Inf handling
            if not np.isfinite(value):
                value = self.fill_value

            validated[key] = value

        return validated

    # =====================================================
    # FAST VECTOR CONVERSION
    # =====================================================

    def enforce_order(self, features: FeatureVector):

        vec = np.full(len(self.expected_features), self.fill_value, dtype=np.float32)

        for key, value in features.items():

            idx = self._feature_index.get(key)
            if idx is None:
                continue

            if isinstance(value, numbers.Number):
                v = float(value)
                if np.isfinite(v):
                    vec[idx] = v

        return vec if self.return_numpy else vec.tolist()

    # =====================================================
    # BATCH VALIDATION
    # =====================================================

    def validate_batch(
        self,
        feature_list: List[FeatureVector],
    ) -> List[FeatureVector]:

        if not feature_list:
            raise ValueError("Feature list cannot be empty")

        validated = []

        for idx, fv in enumerate(feature_list):

            try:
                validated.append(self.validate(fv))

            except Exception as e:
                logger.error("Validation failed at index %d: %s", idx, e)

                if self.strict:
                    raise

                validated.append({})

        return validated

    # =====================================================
    # BATCH VECTOR (FAST)
    # =====================================================

    def enforce_order_batch(
        self,
        feature_list: List[FeatureVector],
    ):

        n = len(feature_list)
        d = len(self.expected_features)

        matrix = np.full((n, d), self.fill_value, dtype=np.float32)

        for i, features in enumerate(feature_list):

            for key, value in features.items():

                idx = self._feature_index.get(key)
                if idx is None:
                    continue

                if isinstance(value, numbers.Number):
                    v = float(value)
                    if np.isfinite(v):
                        matrix[i, idx] = v

        return matrix if self.return_numpy else matrix.tolist()

    # =====================================================
    # DIAGNOSTICS
    # =====================================================

    def diff(self, features: FeatureVector) -> Tuple[List[str], List[str]]:
        """
        Return (missing, extra)
        """

        keys = set(features.keys())

        missing = list(self._expected_set - keys)
        extra = list(keys - self._expected_set)

        return missing, extra

    # =====================================================
    # SCHEMA METADATA
    # =====================================================

    def schema_summary(self) -> Dict[str, int]:
        return {
            "num_features": len(self.expected_features),
        }

    #  NEW: schema hash (CRITICAL)
    def schema_hash(self) -> str:
        """
        Stable hash for schema consistency (train vs inference)
        """

        joined = "|".join(self.expected_features)
        return hashlib.md5(joined.encode()).hexdigest()