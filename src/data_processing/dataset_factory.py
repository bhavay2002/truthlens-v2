"""
Contract-driven dataset factory.

Single entry point: ``build_dataset(task=…, df=…, tokenizer=…, …)``.
All label-column names are pulled from ``data_contracts.CONTRACTS``,
guaranteeing schema consistency across cleaning, validation, sampling
and dataset construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from src.data_processing.data_contracts import get_contract, DEFAULT_MAX_LENGTH
from src.data_processing.dataset import (
    ClassificationDataset,
    MultiLabelDataset,
)
from src.data_processing.dataset import TASK_ORDER

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass(frozen=True)
class DatasetBuildConfig:
    """Tunables that affect tokenization / cache key.

    UNUSED-D1 (resolved): ``DatasetBuildConfig`` is now wired into
    ``build_dataset`` / ``build_all_datasets`` via the optional
    ``config=`` keyword. Loose ``max_length=…`` / ``log_truncation=…``
    kwargs are kept for back-compat (they're still honoured when no
    ``config`` is passed) but new callers should pass a single
    ``DatasetBuildConfig`` instance so the cache-key extra reflects every
    tokenization-relevant knob in one place.
    """

    max_length: int = DEFAULT_MAX_LENGTH
    return_offsets_mapping: bool = False
    log_truncation: bool = True


# =========================================================
# FACTORY
# =========================================================

def build_dataset(
    *,
    task: str,
    df: pd.DataFrame,
    tokenizer: Any,
    max_length: int = DEFAULT_MAX_LENGTH,
    return_offsets_mapping: bool = False,
    log_truncation: bool = True,
    config: "DatasetBuildConfig | None" = None,
    valid_label_indices: "list[int] | None" = None,
):
    """
    Build a dataset for ``task`` from ``df`` using the canonical task contract.

    When ``config`` is provided its fields take precedence over the loose
    ``max_length`` / ``return_offsets_mapping`` / ``log_truncation`` kwargs.
    The loose kwargs are retained for back-compat with existing callers.
    """
    contract = get_contract(task)

    if config is not None:
        max_length = config.max_length
        return_offsets_mapping = config.return_offsets_mapping
        log_truncation = config.log_truncation

    logger.info(
        "Building dataset | task=%s | type=%s | rows=%d | max_length=%d",
        task,
        contract.task_type,
        len(df),
        max_length,
    )

    common_kwargs = dict(
        text_col=contract.text_column,
        max_length=max_length,
        return_offsets_mapping=return_offsets_mapping,
        log_truncation=log_truncation,
    )

    if contract.task_type == "classification":
        return ClassificationDataset(
            df=df,
            tokenizer=tokenizer,
            label_col=contract.label_columns[0],
            num_classes=contract.num_classes,
            task_name=task,
            **common_kwargs,
        )

    if contract.task_type == "multilabel":
        # ``valid_label_indices`` is computed ONCE on the train split by
        # the trainer (via ``training.loss_balancer.plan_for_dataframe``)
        # and passed into both train AND val/test ``build_dataset``
        # calls. Computing it per split here would let the val split
        # decide its own valid columns and silently misalign with the
        # head's output width.
        if valid_label_indices is not None:
            kept_cols = [contract.label_columns[i] for i in valid_label_indices]
            dropped = len(contract.label_columns) - len(kept_cols)
            if dropped:
                logger.warning(
                    "Dropped %d single-class multilabel column(s) from "
                    "task=%s (kept %d/%d): %s",
                    dropped, task, len(kept_cols),
                    len(contract.label_columns), kept_cols,
                )
        return MultiLabelDataset(
            df=df,
            tokenizer=tokenizer,
            label_cols=contract.label_columns,
            task_name=task,
            valid_label_indices=valid_label_indices,
            **common_kwargs,
        )

    raise ValueError(f"Unsupported task type: {contract.task_type}")


# =========================================================
# BULK FACTORY (MULTI-TASK)
# =========================================================

def build_all_datasets(
    *,
    datasets: Dict[str, Dict[str, pd.DataFrame]],
    tokenizer: Any,
    max_length: int = DEFAULT_MAX_LENGTH,
    return_offsets_mapping: bool = False,
    log_truncation: bool = True,
    config: "DatasetBuildConfig | None" = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Build datasets for every task / split.

    ``datasets`` shape:
        {"bias": {"train": df, "val": df, "test": df}, ...}

    See ``build_dataset`` for the precedence rule between ``config`` and
    the loose kwargs.
    """
    result: Dict[str, Dict[str, Any]] = {}

    for task, splits in datasets.items():
        result[task] = {}
        for split, df in splits.items():
            result[task][split] = build_dataset(
                task=task,
                df=df,
                tokenizer=tokenizer,
                max_length=max_length,
                return_offsets_mapping=return_offsets_mapping,
                log_truncation=log_truncation,
                config=config,
            )

    return result


# =========================================================
# COMPATIBILITY CHECK
# =========================================================

def validate_dataset_compatibility(task: str, df: pd.DataFrame) -> None:
    """Raise if ``df`` is missing any column required by the task contract."""
    contract = get_contract(task)

    missing = []
    if contract.text_column not in df.columns:
        missing.append(contract.text_column)
    for col in contract.label_columns:
        if col not in df.columns:
            missing.append(col)

    if missing:
        raise ValueError(
            f"Dataset mismatch for task={task}. Missing columns: {missing}"
        )


def build_task_masks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["mask_bias"] = (out.get("bias_label", pd.Series([0] * len(out))).fillna(0).astype(float) >= 0).astype(int)
    out["mask_emotion"] = 1 if any(c.startswith("emotion_") for c in out.columns) else 0
    out["mask_propaganda"] = (out.get("propaganda_label", pd.Series([0] * len(out))).fillna(0).notna()).astype(int)
    out["mask_ideology"] = (out.get("ideology_label", pd.Series([0] * len(out))).fillna(0).notna()).astype(int)
    out["mask_narrative"] = 1 if any(c in out.columns for c in ["hero", "villain", "victim"]) else 0
    return out
