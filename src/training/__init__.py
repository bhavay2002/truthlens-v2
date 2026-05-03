"""
Public exports for the ``src.training`` package.

These are the stable entry points used by the rest of TruthLens (api,
inference, evaluation, hyperparameter tuning).  Internal helpers must be
imported from their submodule paths directly.

Note: ``hyperparameter_tuning`` (Optuna) is an *optional* dependency.
Its symbols (``tune_task``, ``tune_all_tasks``, ``create_study``,
``build_objective``) are exposed via ``__getattr__`` so importing this
package does not pull Optuna into memory unless the caller actually asks
for it.
"""

from .trainer import Trainer, TrainerConfig
from .training_step import TrainingStep, TrainingStepConfig, TrainAction
from .training_setup import (
    TrainingSetupConfig,
    setup_runtime,
    optimize_model,
    run_sanity_check,
    get_autocast,
    create_grad_scaler,
)
from .training_utils import (
    TrainingMetrics,
    get_device,
    move_batch_to_device,
    compute_grad_norm,
    get_current_lr,
    compute_throughput,
)
from .loss_engine import LossEngine, LossEngineConfig
from .evaluation_engine import (
    EvaluationEngine,
    EvaluationConfig,
    StreamingAccuracy,
    StreamingF1,
    StreamingMSE,
)
from .monitor_engine import MonitoringEngine, MonitoringConfig, MonitorAction
from .task_scheduler import TaskScheduler, TaskSchedulerConfig
from .experiment_tracker import ExperimentTracker, ExperimentTrackerConfig
from .distributed_engine import DistributedEngine, DistributedConfig
from .instrumentation import (
    AutoDebugEngine,
    LossTracker,
    SpikeDetector,
    GradTracker,
    FailureMemory,
    AnomalyClassifier,
)
from .cross_validation import (
    cross_validate_task,
    cross_validate_all_tasks,
    build_splits,
    resolve_metric,
    build_dashboard,
)
from .create_trainer_fn import create_trainer_fn
from .curriculum import CurriculumScheduler, CurriculumConfig, CurriculumState
from .hard_sample_miner import HardSampleMiner, HardSampleMinerConfig
from .confidence_filter import ConfidenceFilter, ConfidenceFilterConfig
from .dynamic_task_balancer import DynamicTaskWeightBalancer, DynamicTaskBalancerConfig
from .pcgrad import PCGradOptimizer

# Optuna-backed tuning is optional — import lazily.
_LAZY = {
    "tune_task",
    "tune_all_tasks",
    "create_study",
    "build_objective",
}


def __getattr__(name):
    if name in _LAZY:
        from . import hyperparameter_tuning as _ht
        return getattr(_ht, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Trainer
    "Trainer",
    "TrainerConfig",
    # Step
    "TrainingStep",
    "TrainingStepConfig",
    "TrainAction",
    # Setup
    "TrainingSetupConfig",
    "setup_runtime",
    "optimize_model",
    "run_sanity_check",
    "get_autocast",
    "create_grad_scaler",
    # Utils
    "TrainingMetrics",
    "get_device",
    "move_batch_to_device",
    "compute_grad_norm",
    "get_current_lr",
    "compute_throughput",
    # Loss / Eval
    "LossEngine",
    "LossEngineConfig",
    "EvaluationEngine",
    "EvaluationConfig",
    "StreamingAccuracy",
    "StreamingF1",
    "StreamingMSE",
    # Engines
    "MonitoringEngine",
    "MonitoringConfig",
    "MonitorAction",
    "TaskScheduler",
    "TaskSchedulerConfig",
    "ExperimentTracker",
    "ExperimentTrackerConfig",
    "DistributedEngine",
    "DistributedConfig",
    # Instrumentation
    "AutoDebugEngine",
    "LossTracker",
    "SpikeDetector",
    "GradTracker",
    "FailureMemory",
    "AnomalyClassifier",
    # CV
    "cross_validate_task",
    "cross_validate_all_tasks",
    "build_splits",
    "resolve_metric",
    "build_dashboard",
    "create_trainer_fn",
    # Curriculum + hard mining
    "CurriculumScheduler",
    "CurriculumConfig",
    "CurriculumState",
    "HardSampleMiner",
    "HardSampleMinerConfig",
    # Training pipeline fixes
    "ConfidenceFilter",
    "ConfidenceFilterConfig",
    "DynamicTaskWeightBalancer",
    "DynamicTaskBalancerConfig",
    # v2 — PCGrad optimizer
    "PCGradOptimizer",
    # Tuning (lazy)
    "tune_task",
    "tune_all_tasks",
    "create_study",
    "build_objective",
]
