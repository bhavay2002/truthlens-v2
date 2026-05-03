from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False

from src.features.base.base_feature import FeatureContext
from src.features.base.matrix_build import collect_feature_names, dict_rows_to_matrix
from src.features.pipelines.batch_feature_pipeline import BatchFeaturePipeline
from src.features.pipelines.feature_pipeline import (
    partition_feature_sections,
    BIAS_FEATURE_NAMES,
    FRAMING_FEATURE_NAMES,
    IDEOLOGICAL_FEATURE_NAMES,
)
from src.features.cache.cache_manager import CacheManager
from src.features.fusion.feature_scaling import FeatureScalingPipeline
from src.features.fusion.feature_selection import FeatureSelectionPipeline

logger = logging.getLogger(__name__)

FeatureVector = Dict[str, float]


@dataclass
class DatasetFeatureGenerator:

    pipeline: BatchFeaturePipeline
    cache_manager: Optional[CacheManager] = None
    cache_namespace: str = "dataset_features"

    # Audit fix §1.6 — scaler / selector are now first-class fields on
    # the generator instead of being fished out of ``pipeline.pipeline``
    # (where they never lived in the first place: ``FeaturePipeline``
    # has no ``scaler``/``selector`` attribute, so the previous cache
    # branch's ``if pipeline.scaler is not None`` would have raised
    # ``AttributeError`` the moment the cache was actually populated).
    # Both branches of ``generate()`` now share the exact same tail, so
    # the no-cache path (used by the test suite) and the cache path
    # (used in production) validate the same pipeline.
    scaler: Optional[FeatureScalingPipeline] = None
    selector: Optional[FeatureSelectionPipeline] = None

    _feature_order: List[str] = field(default_factory=list, init=False)

    # =====================================================
    # CONTEXT BUILD
    # =====================================================

    def _build_contexts(self, texts: List[str]) -> List[FeatureContext]:

        contexts = []

        for text in texts:
            if not text or not isinstance(text, str):
                raise ValueError("Input texts must be non-empty strings")

            contexts.append(FeatureContext(text=text))

        return contexts

    # =====================================================
    # CACHE-AWARE EXTRACTION
    # =====================================================

    def _cached_extract(self, contexts: List[FeatureContext]) -> List[FeatureVector]:

        if self.cache_manager is None:
            return self.pipeline.extract(contexts)

        cache = self.cache_manager.get_cache(self.cache_namespace)

        results: List[Optional[FeatureVector]] = [None] * len(contexts)

        uncached_contexts = []
        uncached_indices = []

        for i, ctx in enumerate(contexts):

            # Audit fix §1.9 — use the public ``context_key`` API instead
            # of reaching into the leading-underscore helper.
            key = self.cache_manager.context_key(ctx)
            cached = cache.load(key)

            if cached is not None:
                results[i] = cached
            else:
                uncached_contexts.append(ctx)
                uncached_indices.append(i)

        if uncached_contexts:

            new_features = self.pipeline.extract(uncached_contexts)

            for idx, feat, ctx in zip(uncached_indices, new_features, uncached_contexts):
                key = self.cache_manager.context_key(ctx)
                cache.save(key, feat)
                results[idx] = feat

        if any(r is None for r in results):
            raise RuntimeError("Incomplete feature extraction (cache mismatch)")

        return [r for r in results if r is not None]

    # =====================================================
    # MATRIX GENERATION
    # =====================================================

    def generate(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        fit: bool = False,
    ) -> Tuple[np.ndarray, List[str]]:

        if not texts:
            raise ValueError("Input text list cannot be empty")

        contexts = self._build_contexts(texts)

        logger.info(
            "Generating dataset features | samples=%d cache=%s scaler=%s selector=%s",
            len(contexts),
            self.cache_manager is not None,
            self.scaler is not None,
            self.selector is not None,
        )

        # ---------------------------
        # 1. Extraction (cache-aware OR pure passthrough)
        # ---------------------------
        # Audit fix §1.6 — the no-cache branch used to call
        # ``pipeline.extract_with_labels`` (which silently drops both
        # ``labels`` and ``fit`` because ``FeaturePipeline.process``
        # ignores them) and never ran scaler/selector at all. The cache
        # branch in turn tried to reach into ``self.pipeline.pipeline``
        # for a scaler/selector that does not exist there. Both bugs
        # are eliminated by funnelling every code path through
        # ``_cached_extract`` (which short-circuits to the raw pipeline
        # when ``cache_manager is None``) and applying scaler+selector
        # in the common tail below.

        features = self._cached_extract(contexts)

        if not features:
            raise RuntimeError("Empty feature output")

        # ---------------------------
        # 2. Scaling (common tail — runs whether cache is on or off)
        # ---------------------------
        if self.scaler is not None:
            if fit:
                self.scaler.fit(features)
                logger.info("Scaler fitted on %d samples", len(features))

            features = self.scaler.transform(features, return_array=False)

        # ---------------------------
        # 3. Selection (common tail — runs whether cache is on or off)
        # ---------------------------
        if self.selector is not None:
            if fit:
                self.selector.fit(features, labels)
                logger.info("Selector fitted on %d samples", len(features))

            features = self.selector.transform(features, return_array=False)

        # ---------------------------
        # Matrix build — audit fix §2.8
        # ---------------------------
        # The schema (column union) and the dense ``float32`` matrix
        # are built by the centralised :mod:`matrix_build` helper so
        # both this method and ``generate_by_section`` share one
        # pre-allocation routine. Behaviour is unchanged: the column
        # order is the deterministic ``sorted`` union of dict keys
        # across every row, and missing keys are written as ``0.0``.

        feature_names = collect_feature_names(features)
        matrix = dict_rows_to_matrix(features, feature_names, dtype=np.float32)

        self._feature_order = feature_names

        logger.info(
            "Feature matrix ready | shape=%s",
            matrix.shape,
        )

        return matrix, feature_names

    # =====================================================
    # SECTION SPLIT
    # =====================================================

    def generate_by_section(
        self,
        texts: List[str],
        sections: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[np.ndarray, List[str]]]:

        if not texts:
            raise ValueError("Input text list cannot be empty")

        contexts = self._build_contexts(texts)

        if not self.pipeline._initialized:
            self.pipeline.initialize()

        raw_features = self.pipeline.extract(contexts)

        partitioned: Dict[str, List[FeatureVector]] = {}

        for sample in raw_features:
            sec_map = partition_feature_sections(sample)

            for sec_name, sec_features in sec_map.items():
                partitioned.setdefault(sec_name, []).append(sec_features)

        result: Dict[str, Tuple[np.ndarray, List[str]]] = {}

        for sec_name, samples in partitioned.items():

            if sections and sec_name not in sections:
                continue

            if not samples or not samples[0]:
                continue

            # Audit fix §2.8 — same centralised pre-allocated build as
            # :meth:`generate`. Keeping both call sites on a single
            # helper is the whole point of the §2.8 refactor.
            feature_names = collect_feature_names(samples)
            matrix = dict_rows_to_matrix(samples, feature_names, dtype=np.float32)

            result[sec_name] = (matrix, feature_names)

            logger.info("Section %s → shape=%s", sec_name, matrix.shape)

        return result

    # =====================================================
    # DATAFRAME
    # =====================================================

    def generate_dataframe(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        fit: bool = False,
    ):

        if not PANDAS_AVAILABLE:
            raise RuntimeError("pandas not installed")

        matrix, names = self.generate(texts, labels, fit)

        df = pd.DataFrame(matrix, columns=names)

        if labels is not None:
            df["label"] = labels

        return df

    # =====================================================
    # UTILITIES
    # =====================================================

    def get_feature_order(self) -> List[str]:

        if not self._feature_order:
            raise RuntimeError("Call generate() first")

        return self._feature_order

    def get_bias_module_feature_names(self) -> Dict[str, List[str]]:

        return {
            "bias": BIAS_FEATURE_NAMES,
            "framing": FRAMING_FEATURE_NAMES,
            "ideology": IDEOLOGICAL_FEATURE_NAMES,
        }
