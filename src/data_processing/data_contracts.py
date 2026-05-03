"""
Data Contracts for TruthLens

Defines strict schemas for each task.
Used by:
- data_validator
- dataset_factory
- training pipeline
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional

# EMOTION-11: pull the live label count from the canonical schema so
# the multilabel column list below auto-resizes if EMOTION_LABELS is
# ever changed again. Aliased with a leading underscore because this
# module's public API doesn't re-export it.
from src.features.emotion.emotion_schema import (
    NUM_EMOTION_LABELS as _NUM_EMOTION_LABELS,
)


# =========================================================
# GLOBAL DEFAULTS  (CFG-D6 — single source of truth)
#
# ``DEFAULT_MAX_LENGTH`` was previously duplicated across
# ``DataPipelineConfig.max_length``, ``DatasetBuildConfig.max_length``,
# ``build_dataset(max_length=…)`` and ``build_all_datasets(max_length=…)``
# — four literals that had to be kept in sync by hand. Centralising the
# constant here means every downstream default reads from one place; an
# experiment that flips it (e.g. to 256) only has to set it once.
# =========================================================

DEFAULT_MAX_LENGTH: int = 512


# =========================================================
# BASE CONTRACT
# =========================================================

@dataclass(frozen=True)
class DataContract:
    task: str
    task_type: str  # classification | multilabel

    text_column: str

    # labels
    label_columns: List[str]

    # classification only
    num_classes: Optional[int] = None

    # optional metadata columns
    optional_columns: Optional[List[str]] = None


# =========================================================
# TASK CONTRACTS
# =========================================================

CONTRACTS: Dict[str, DataContract] = {

    # -----------------------------------------------------
    # SINGLE-LABEL TASKS
    # -----------------------------------------------------

    "bias": DataContract(
        task="bias",
        task_type="classification",
        text_column="text",
        label_columns=["bias_label"],
        num_classes=2,
    ),

    "ideology": DataContract(
        task="ideology",
        task_type="classification",
        text_column="text",
        label_columns=["ideology_label"],
        num_classes=3,
    ),

    "propaganda": DataContract(
        task="propaganda",
        task_type="classification",
        text_column="text",
        label_columns=["propaganda_label"],
        num_classes=2,
    ),

    # -----------------------------------------------------
    # MULTI-LABEL TASKS
    # -----------------------------------------------------

    "narrative_frame": DataContract(
        task="narrative_frame",
        task_type="multilabel",
        text_column="text",
        label_columns=["CO", "EC", "HI", "MO", "RE"],
    ),

    "narrative": DataContract(
        task="narrative",
        task_type="multilabel",
        text_column="text",
        label_columns=["hero", "villain", "victim"],
        optional_columns=[
            "hero_entities",
            "villain_entities",
            "victim_entities",
        ],
    ),

    # EMOTION-11: positional column count derived from the canonical
    # ``EMOTION_LABELS`` list (src/features/emotion/emotion_schema.py).
    # Edit there to change the live label count; everything downstream
    # reads from this single source of truth.
    "emotion": DataContract(
        task="emotion",
        task_type="multilabel",
        text_column="text",
        label_columns=[
            f"emotion_{i}" for i in range(_NUM_EMOTION_LABELS)
        ],
    ),
}


# =========================================================
# ACCESS HELPERS
# =========================================================

def get_contract(task: str) -> DataContract:
    if task not in CONTRACTS:
        raise ValueError(f"Unknown task: {task}")
    return CONTRACTS[task]


def list_tasks() -> List[str]:
    return list(CONTRACTS.keys())


# =========================================================
# VALIDATION HELPERS
# =========================================================

def get_required_columns(task: str) -> List[str]:
    contract = get_contract(task)
    return [contract.text_column] + contract.label_columns


def get_optional_columns(task: str) -> List[str]:
    contract = get_contract(task)
    return contract.optional_columns or []


def is_multilabel(task: str) -> bool:
    return get_contract(task).task_type == "multilabel"


def is_classification(task: str) -> bool:
    return get_contract(task).task_type == "classification"


def get_num_classes(task: str) -> Optional[int]:
    return get_contract(task).num_classes


# =========================================================
# DEBUG / INSPECTION
# =========================================================

def describe_contract(task: str) -> Dict:
    c = get_contract(task)

    return {
        "task": c.task,
        "type": c.task_type,
        "text_column": c.text_column,
        "labels": c.label_columns,
        "num_classes": c.num_classes,
        "optional": c.optional_columns,
    }