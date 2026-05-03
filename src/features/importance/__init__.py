"""Backward-compat shim — implementation lives in
:mod:`src.evaluation.importance` (audit fix §8).

The importance helpers (FeatureAblation / PermutationImportance /
ShapImportance) are offline-only and were never imported by
``src.inference``, ``api.app``, or any model forward path. They
existed under ``src/features/importance/`` for historical reasons
only.  Their sole live caller is ``src.evaluation.advanced_analysis``,
so the package now lives under :mod:`src.evaluation.importance`.

This shim keeps any external pickle / notebook / experiment script
that still imports the old path working without a churn-only diff.
New code MUST import from :mod:`src.evaluation.importance`.
"""

from __future__ import annotations

from src.evaluation.importance.feature_ablation import (  # noqa: F401
    FeatureAblation,
)
from src.evaluation.importance.permutation_importance import (  # noqa: F401
    PermutationImportance,
)
from src.evaluation.importance.shap_importance import (  # noqa: F401
    ShapImportance,
)

__all__ = [
    "FeatureAblation",
    "PermutationImportance",
    "ShapImportance",
]
