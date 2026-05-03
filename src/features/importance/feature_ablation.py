"""Backward-compat shim — see :mod:`src.evaluation.importance.feature_ablation`."""

from __future__ import annotations

from src.evaluation.importance.feature_ablation import *  # noqa: F401,F403
from src.evaluation.importance.feature_ablation import (  # noqa: F401
    FeatureAblation,
    accuracy_metric,
    mse_metric,
)
