"""
Backward-compat shim — implementation lives in
``src.features.fusion.feature_reduction`` (audit task 8 merge).

Importing ``FeaturePruner`` from this path continues to work for
existing call sites, but new code should import from
``src.features.fusion.feature_reduction`` directly.
"""

from __future__ import annotations

from src.features.fusion.feature_reduction import (  # noqa: F401
    DEFAULT_CORRELATION_THRESHOLD,
    DEFAULT_VARIANCE_THRESHOLD,
    FeaturePruner,
    FeatureReductionPipeline,
    FeatureVector,
    _dict_to_matrix,
    _matrix_to_dicts,
)

__all__ = [
    "DEFAULT_CORRELATION_THRESHOLD",
    "DEFAULT_VARIANCE_THRESHOLD",
    "FeaturePruner",
    "FeatureReductionPipeline",
    "FeatureVector",
]
