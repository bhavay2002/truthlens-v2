"""
Package: src.models.ensemble
Description:
    Ensemble architectures for combining predictions from multiple models.

    Exposes:
      • EnsembleModel / EnsembleConfig          — average / weighted-average / majority-vote
      • WeightedEnsembleModel / WeightedEnsembleConfig — explicit per-model weight assignment
      • StackingEnsembleModel / StackingEnsembleConfig — meta-learner stacking
"""

from src.models.ensemble.ensemble_model import EnsembleConfig, EnsembleModel
from src.models.ensemble.stacking_ensemble import (
    StackingEnsembleConfig,
    StackingEnsembleModel,
)
from src.models.ensemble.weighted_ensemble import (
    WeightedEnsembleConfig,
    WeightedEnsembleModel,
)

__all__ = [
    "EnsembleConfig",
    "EnsembleModel",
    "StackingEnsembleConfig",
    "StackingEnsembleModel",
    "WeightedEnsembleConfig",
    "WeightedEnsembleModel",
]
