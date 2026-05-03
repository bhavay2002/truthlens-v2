"""Backward-compat shim — see :mod:`src.evaluation.importance.shap_importance`."""

from __future__ import annotations

from src.evaluation.importance.shap_importance import *  # noqa: F401,F403
from src.evaluation.importance.shap_importance import (  # noqa: F401
    ShapImportance,
)
