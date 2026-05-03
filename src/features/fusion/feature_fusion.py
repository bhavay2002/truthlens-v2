from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Union

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.tokenization import ensure_tokens_word
from src.features.fusion.feature_merger import merge_features

logger = logging.getLogger(__name__)


# =========================================================
# FEATURE FUSION
# =========================================================

@dataclass
class FeatureFusion:
    """
    Aggregates outputs from every registered extractor.

    Scaling is intentionally NOT done here. Per-row, per-extractor numeric
    scaling is statistically invalid (mixes incompatible units within a
    single row). All scaling lives in `FeatureScalingPipeline`, which is
    fitted on the training set and applied as a separate stage in
    `FeatureEngineeringPipeline`.
    """

    features: List[BaseFeature] = field(default_factory=list)
    enforce_unique_names: bool = True
    return_vector: bool = False

    _feature_order: List[str] = field(default_factory=list, init=False)

    # -----------------------------------------------------

    def _validate_feature_names(self) -> None:
        names = [f.name for f in self.features]

        if len(names) != len(set(names)):
            counts = Counter(names)
            duplicates = {name for name, cnt in counts.items() if cnt > 1}
            raise ValueError(f"Duplicate feature extractors detected: {duplicates}")

    # -----------------------------------------------------

    def _ensure_initialized(self) -> None:
        if not hasattr(self, "_initialized"):
            for feature in self.features:
                feature.initialize()
            self._initialized = True

    # -----------------------------------------------------

    def _ensure_validated(self) -> None:
        if self.enforce_unique_names and not hasattr(self, "_validated"):
            self._validate_feature_names()
            self._validated = True

    # -----------------------------------------------------

    def _finalize(self, fused: Dict[str, float]):
        if not self._feature_order:
            self._feature_order = sorted(fused.keys())

        if self.return_vector:
            return np.array(
                [fused.get(k, 0.0) for k in self._feature_order],
                dtype=np.float32,
            )
        return fused

    # =====================================================
    # SINGLE
    # =====================================================

    def extract(self, context: FeatureContext):

        self._ensure_validated()
        self._ensure_initialized()

        # Tokenize once at the top so each extractor reads
        # `context.tokens_word` instead of re-running its own regex.
        ensure_tokens_word(context)

        outputs: List[Dict[str, float]] = []

        for feature in self.features:
            try:
                output = feature.safe_extract(context)
            except Exception:
                logger.exception("Feature failed: %s", feature.name)
                continue

            if isinstance(output, dict) and output:
                outputs.append(output)

        return self._finalize(merge_features(outputs))

    # =====================================================
    # BATCH (dispatches through each feature's extract_batch)
    # =====================================================

    def extract_batch(self, contexts: List[FeatureContext]):

        self._ensure_validated()
        self._ensure_initialized()

        if not contexts:
            return []

        # Tokenize each context once before per-feature dispatch.
        for ctx in contexts:
            ensure_tokens_word(ctx)

        n = len(contexts)
        per_context: List[List[Dict[str, float]]] = [[] for _ in range(n)]

        for feature in self.features:
            try:
                batch_outputs = feature.safe_extract_batch(contexts)
            except Exception:
                logger.exception("Feature batch failed: %s", feature.name)
                batch_outputs = [{} for _ in range(n)]

            # Pad / truncate defensively
            if len(batch_outputs) != n:
                logger.warning(
                    "Feature '%s' returned %d outputs for %d inputs; padding",
                    feature.name, len(batch_outputs), n,
                )
                batch_outputs = list(batch_outputs[:n]) + [{}] * max(0, n - len(batch_outputs))

            # Audit fix §1.5 — emit a per-extractor ``<name>_extracted``
            # indicator so the model can mask zero-fills caused by a
            # silent extractor failure (or by an empty input). The
            # codebase already uses this pattern for ``sem_available``
            # and ``syn_spacy_available``; this generalises it to every
            # extractor so a downstream feature drop is observable.
            indicator_key = f"{feature.name}_extracted"
            for i, output in enumerate(batch_outputs):
                if isinstance(output, dict) and output:
                    per_context[i].append(output)
                    per_context[i].append({indicator_key: 1.0})
                else:
                    per_context[i].append({indicator_key: 0.0})

        results = []
        for outputs in per_context:
            results.append(self._finalize(merge_features(outputs)))
        return results

    # -----------------------------------------------------

    def get_feature_order(self) -> List[str]:
        return self._feature_order

    # -----------------------------------------------------

    def get_feature_dim(self) -> int:
        return len(self._feature_order)
