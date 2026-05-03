"""
File: task_config.py
Location: src/config/

Production-grade task registry for multi-task system.

This module:
- Builds task registry from YAML config
- Validates task definitions
- Provides fast lookup helpers
- Acts as SINGLE runtime source of truth

NOTE:
YAML config is the only source of truth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

from src.utils.config_loader import load_app_config

logger = logging.getLogger(__name__)


# =========================================================
# TASK DATACLASS (STRICT CONTRACT)
# =========================================================

@dataclass(slots=True, frozen=True)
class TaskDefinition:
    name: str
    task_type: str
    num_labels: int
    loss: str
    loss_weight: float
    threshold: float
    auto_threshold: bool


# =========================================================
# INTERNAL REGISTRY
# =========================================================

_TASK_REGISTRY: Dict[str, TaskDefinition] = {}


# =========================================================
# VALIDATION
# =========================================================

VALID_TASK_TYPES = {"binary", "multiclass", "multilabel"}
VALID_LOSSES = {"bce", "cross_entropy"}


def _validate_task(task: TaskDefinition):

    if task.task_type not in VALID_TASK_TYPES:
        raise ValueError(f"{task.name}: invalid task_type")

    if task.num_labels <= 0:
        raise ValueError(f"{task.name}: num_labels must be > 0")

    if task.loss not in VALID_LOSSES:
        raise ValueError(f"{task.name}: invalid loss")

    if task.loss_weight <= 0:
        raise ValueError(f"{task.name}: loss_weight must be > 0")

    if task.task_type == "multilabel" and task.threshold <= 0:
        raise ValueError(f"{task.name}: invalid threshold")


# =========================================================
# REGISTRY BUILDER
# =========================================================

def _build_registry():

    config = load_app_config()

    for task_name, task_cfg in config.tasks.items():

        # ---- derive defaults intelligently ----
        if task_cfg.task_type == "multilabel":
            loss = "bce"
        elif task_cfg.task_type == "binary":
            loss = "bce"
        else:
            loss = "cross_entropy"

        definition = TaskDefinition(
            name=task_name,
            task_type=task_cfg.task_type,
            num_labels=task_cfg.num_labels,
            loss=loss,
            loss_weight=1.0,
            threshold=0.5,
            auto_threshold=(task_cfg.task_type == "multilabel"),
        )

        _validate_task(definition)

        _TASK_REGISTRY[task_name] = definition

    if not _TASK_REGISTRY:
        raise RuntimeError("Task registry is empty")

    logger.info("Task registry initialized | %d tasks", len(_TASK_REGISTRY))


# Build on import (safe due to caching)
_build_registry()


# =========================================================
# PUBLIC API (USED ACROSS SYSTEM)
# =========================================================

def get_task(task: str) -> TaskDefinition:
    return _TASK_REGISTRY[task]


def get_all_tasks():
    return list(_TASK_REGISTRY.keys())


def get_task_type(task: str) -> str:
    return _TASK_REGISTRY[task].task_type


def get_output_dim(task: str) -> int:
    return _TASK_REGISTRY[task].num_labels


def get_loss_name(task: str) -> str:
    return _TASK_REGISTRY[task].loss


def get_loss_weight(task: str) -> float:
    return _TASK_REGISTRY[task].loss_weight


def get_threshold(task: str) -> float:
    return _TASK_REGISTRY[task].threshold


def use_auto_threshold(task: str) -> bool:
    return _TASK_REGISTRY[task].auto_threshold


def is_multilabel(task: str) -> bool:
    return _TASK_REGISTRY[task].task_type == "multilabel"


def is_binary(task: str) -> bool:
    return _TASK_REGISTRY[task].task_type == "binary"


def is_multiclass(task: str) -> bool:
    return _TASK_REGISTRY[task].task_type == "multiclass"


# =========================================================
# BACKWARD COMPATIBILITY: dict-style TASK_CONFIG proxy
# =========================================================
# Older modules import ``TASK_CONFIG`` and access it as
# ``TASK_CONFIG[task]["type"]`` / ``["num_labels"]`` and iterate via
# ``TASK_CONFIG.keys()`` / ``TASK_CONFIG.items()``. The registry is the new
# source of truth; this proxy preserves the legacy contract without forcing
# the rest of the codebase to change.

class _TaskConfigProxy:
    """Read-only mapping wrapper around the task registry.

    Exposes each registered task as ``{"type": str, "num_labels": int,
    "loss": str, "loss_weight": float, "threshold": float,
    "auto_threshold": bool}`` so historical lookups continue to work.
    """

    __slots__ = ()

    @staticmethod
    def _as_dict(td: TaskDefinition) -> Dict[str, Any]:
        return {
            "type": td.task_type,
            "task_type": td.task_type,
            "num_labels": td.num_labels,
            "loss": td.loss,
            "loss_weight": td.loss_weight,
            "threshold": td.threshold,
            "auto_threshold": td.auto_threshold,
        }

    def __getitem__(self, task: str) -> Dict[str, Any]:
        if task not in _TASK_REGISTRY:
            raise KeyError(task)
        return self._as_dict(_TASK_REGISTRY[task])

    def __contains__(self, task: object) -> bool:
        return task in _TASK_REGISTRY

    def __iter__(self):
        return iter(_TASK_REGISTRY)

    def __len__(self) -> int:
        return len(_TASK_REGISTRY)

    def keys(self):
        return _TASK_REGISTRY.keys()

    def values(self):
        return [self._as_dict(td) for td in _TASK_REGISTRY.values()]

    def items(self):
        return [(name, self._as_dict(td)) for name, td in _TASK_REGISTRY.items()]

    def get(self, task: str, default: Any = None) -> Any:
        td = _TASK_REGISTRY.get(task)
        return self._as_dict(td) if td is not None else default

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"TaskConfigProxy({list(_TASK_REGISTRY)!r})"


# Public dict-like API expected by legacy callers.
TASK_CONFIG: _TaskConfigProxy = _TaskConfigProxy()
