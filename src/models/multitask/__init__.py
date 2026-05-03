"""Multi-task model components."""

from ..heads.multitask_head import MultiTaskHead as TaskHead
from .multitask_truthlens_model import MultiTaskTruthLensModel
from .interacting_model import (
    InteractingMultiTaskModel,
    InteractingMultiTaskConfig,
    MultiViewPooling,
    CrossTaskInteractionLayer,
    LatentFusionHead,
)

__all__ = [
    "MultiTaskTruthLensModel",
    "TaskHead",
    "InteractingMultiTaskModel",
    "InteractingMultiTaskConfig",
    "MultiViewPooling",
    "CrossTaskInteractionLayer",
    "LatentFusionHead",
]

