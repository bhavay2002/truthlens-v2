"""Multi-task model components."""

from ..heads.multitask_head import MultiTaskHead as TaskHead
from .multitask_truthlens_model import MultiTaskTruthLensModel

__all__ = ["MultiTaskTruthLensModel", "TaskHead"]

