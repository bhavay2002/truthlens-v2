from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np
import torch

from src.graph.graph_pipeline import GraphPipeline, get_default_pipeline
from src.features.base.base_feature import FeatureContext
from src.features.bias.bias_features import BiasFeatures
from src.features.bias.framing_features import FramingFeatures
from src.features.bias.ideological_features import IdeologicalFeatures
from src.features.feature_schema_validator import FeatureSchemaValidator
from src.features.feature_statistics import FeatureStatistics
from src.features.pipelines.feature_pipeline import ALL_BIAS_MODULE_FEATURE_NAMES

logger = logging.getLogger(__name__)

# Module-level singleton extractors — instantiated once at import, reused on
# every call to prepare_from_text instead of being rebuilt per request.
_BIAS_EXTRACTOR: Optional["BiasFeatures"] = None
_FRAMING_EXTRACTOR: Optional["FramingFeatures"] = None
_IDEOLOGICAL_EXTRACTOR: Optional["IdeologicalFeatures"] = None


def _get_text_extractors():
    global _BIAS_EXTRACTOR, _FRAMING_EXTRACTOR, _IDEOLOGICAL_EXTRACTOR
    if _BIAS_EXTRACTOR is None:
        _BIAS_EXTRACTOR = BiasFeatures()
        _FRAMING_EXTRACTOR = FramingFeatures()
        _IDEOLOGICAL_EXTRACTOR = IdeologicalFeatures()
    return _BIAS_EXTRACTOR, _FRAMING_EXTRACTOR, _IDEOLOGICAL_EXTRACTOR


# =========================================================
# FLATTEN
# =========================================================
#
# LAT-2: this used to live in two places — ``_flatten`` (in-process) and
# ``_prepare_flat_features_worker`` (multiprocessing fallback). The two
# disagreed: the worker emitted ``{key}_count`` for list/tuple/set values
# while ``_flatten`` silently dropped them, so the schema picked up
# different numbers depending on batch size. They are now a single
# helper at module scope, used by both code paths.

def _flatten_features(features: Dict[str, Any]) -> Dict[str, float]:
    flat: Dict[str, float] = {}

    for key, value in features.items():
        if key == "text":
            continue

        if isinstance(value, bool):
            flat[key] = float(value)

        elif isinstance(value, (int, float)):
            flat[key] = float(value)

        elif isinstance(value, (list, tuple, set)):
            flat[f"{key}_count"] = float(len(value))

        elif isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, (int, float)):
                    flat[f"{key}_{k}"] = float(v)

    return flat


# =========================================================
# CONFIG
# =========================================================

@dataclass
class FeaturePreparationConfig:
    feature_schema: List[str]
    apply_scaling: bool = True
    apply_feature_selection: bool = True
    return_tensor: bool = True
    dtype: str = "float32"
    derive_graph_features: bool = False  # 🔥 disabled by default for speed


# =========================================================
# MAIN CLASS
# =========================================================

class FeaturePreparer:

    def __init__(
        self,
        config: FeaturePreparationConfig,
        scaler: Optional[Any] = None,
        selector: Optional[Any] = None,
    ):

        self.config = config
        self.scaler = scaler
        self.selector = selector

        self.feature_dim = len(config.feature_schema)
        self.feature_index = {f: i for i, f in enumerate(config.feature_schema)}

        self.schema_validator = FeatureSchemaValidator(
            expected_features=config.feature_schema,
            strict=False,
            allow_missing=True,
            allow_extra=True,
        )

        # G-R1: reuse the process-wide singleton when graph features
        # are enabled — avoids duplicating spaCy + 15 analyzers.
        self.graph_pipeline = get_default_pipeline() if config.derive_graph_features else None

        logger.info(f"FeaturePreparer initialized | dim={self.feature_dim}")

    # =====================================================
    # CORE FLATTEN
    # =====================================================
    #
    # LAT-2: there used to be a multiprocessing pool path here. Spawning
    # workers per batch for a CPU-bound dict flatten was strictly slower
    # than the in-process loop (process startup + pickling cost dwarf the
    # arithmetic), and the worker emitted a slightly different schema than
    # the in-process flattener — so the active code path silently changed
    # under load. Both paths now go through ``_flatten_features``.

    def _flatten(self, features: Dict[str, Any]) -> Dict[str, float]:
        return _flatten_features(features)

    # =====================================================
    # VECTORIZE
    # =====================================================

    def _to_vector(self, flat: Dict[str, float]):

        vec = np.zeros(self.feature_dim, dtype=np.float32)

        for k, v in flat.items():
            idx = self.feature_index.get(k)
            if idx is not None:
                vec[idx] = v

        return vec

    # =====================================================
    # TRANSFORMS
    # =====================================================

    def _transform(self, X):

        if self.scaler:
            X = self.scaler.transform(X)

        if self.selector:
            X = self.selector.transform(X)

        return X

    # =====================================================
    # SINGLE
    # =====================================================

    def prepare_single(self, features: Dict[str, Any]):

        flat = self._flatten(features)
        vec = self._to_vector(flat)[None, :]

        vec = self._transform(vec)

        if self.config.return_tensor:
            return torch.tensor(vec, dtype=torch.float32)

        return vec

    # =====================================================
    # BATCH
    # =====================================================

    def prepare_batch(self, feature_dicts: List[Dict[str, Any]]):

        # LAT-2: always flatten in-process. The previous code spawned a
        # multiprocessing pool for batches of >=32 dicts, which was both
        # slower (process startup + pickling for a CPU-trivial flatten)
        # and emitted a different schema than the in-process branch.
        flats = [self._flatten(f) for f in feature_dicts]

        X = np.zeros((len(flats), self.feature_dim), dtype=np.float32)

        for i, flat in enumerate(flats):
            for k, v in flat.items():
                idx = self.feature_index.get(k)
                if idx is not None:
                    X[i, idx] = v

        X = self._transform(X)

        if self.config.return_tensor:
            return torch.tensor(X, dtype=torch.float32)

        return X

    # =====================================================
    # DIRECT TEXT → FEATURES
    # =====================================================

    def prepare_from_text(self, text: str):

        ctx = FeatureContext(text=text)

        bias_ext, framing_ext, ideological_ext = _get_text_extractors()

        features = {"text": text}
        features.update(bias_ext.extract(ctx))
        features.update(framing_ext.extract(ctx))
        features.update(ideological_ext.extract(ctx))

        return self.prepare_single(features)

    # =====================================================
    # STATS
    # =====================================================

    def compute_feature_statistics(self, feature_dicts):

        flats = [self._flatten(f) for f in feature_dicts]

        stats = FeatureStatistics()

        return {
            "summary": stats.dataset_summary(flats),
            "constant_features": stats.detect_constant_features(flats),
        }

    # =====================================================
    # UTIL
    # =====================================================

    def get_feature_schema(self):
        return self.config.feature_schema

    def feature_dimension(self):
        return self.feature_dim