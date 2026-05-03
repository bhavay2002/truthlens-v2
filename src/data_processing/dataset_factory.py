"""
Contract-driven dataset factory.

Single entry point: ``build_dataset(task=…, df=…, tokenizer=…, …)``.
All label-column names are pulled from ``data_contracts.CONTRACTS``,
guaranteeing schema consistency across cleaning, validation, sampling
and dataset construction.

Fixes applied (audit v3):
  BUG-D1: build_task_masks produced incorrect per-row masks.
    - mask_emotion and mask_narrative were scalar integers (1 or 0),
      not per-row boolean Series, so the resulting DataFrame column was
      broadcast to a single scalar rather than a N-length array.
    - mask_propaganda and mask_ideology always returned True because
      .fillna(0).notna() is trivially True on a fill result.
    Rewritten to produce correct per-row binary Series for every task.
  UNUSED-D1 note: build_task_masks is not called anywhere in the core
    pipeline; it is exposed here as a utility for external callers.
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
    """Return a copy of ``df`` with per-row binary mask columns for each task.

    BUG-D1 fix: the previous implementation produced scalar integers (1 or 0)
    for emotion and narrative (broadcasting a scalar to the entire column) and
    always True for propaganda / ideology (``fillna(0).notna()`` is trivially
    True). All five masks are now correct per-row boolean Series.

    Mask semantics: 1 if the row has a non-null, non-negative value for the
    task's primary label column(s); 0 otherwise.
    """
    out = df.copy()

    # bias — single integer label column
    if "bias_label" in out.columns:
        out["mask_bias"] = out["bias_label"].notna().astype(int)
    else:
        out["mask_bias"] = 0

    # emotion — any emotion_* column present and non-null
    emotion_cols = [c for c in out.columns if c.startswith("emotion_")]
    if emotion_cols:
        out["mask_emotion"] = out[emotion_cols].notna().any(axis=1).astype(int)
    else:
        out["mask_emotion"] = 0

    # propaganda — single integer label column
    if "propaganda_label" in out.columns:
        out["mask_propaganda"] = out["propaganda_label"].notna().astype(int)
    else:
        out["mask_propaganda"] = 0

    # ideology — single integer label column
    if "ideology_label" in out.columns:
        out["mask_ideology"] = out["ideology_label"].notna().astype(int)
    else:
        out["mask_ideology"] = 0

    # narrative — any of hero/villain/victim present and non-null
    narrative_cols = [c for c in ("hero", "villain", "victim") if c in out.columns]
    if narrative_cols:
        out["mask_narrative"] = out[narrative_cols].notna().any(axis=1).astype(int)
    else:
        out["mask_narrative"] = 0

    return out
