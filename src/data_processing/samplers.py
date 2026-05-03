"""
Per-task samplers.

Label column names come from ``data_contracts`` so this module stays in
sync with cleaning, validation and the dataset factory.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

from src.data_processing.data_contracts import get_contract

logger = logging.getLogger(__name__)


# =========================================================
# CLASSIFICATION
# =========================================================

def build_classification_sampler(
    labels: List[int],
    *,
    normalize: bool = True,
) -> WeightedRandomSampler:
    """Inverse-frequency WeightedRandomSampler for classification tasks."""
    labels_arr = np.asarray(labels, dtype=np.int64)

    class_counts = np.bincount(labels_arr)
    total = class_counts.sum()
    class_weights = total / np.maximum(class_counts, 1)

    if normalize:
        class_weights = class_weights / class_weights.sum()

    sample_weights = class_weights[labels_arr]

    logger.info("Sampler | classification | classes=%d", len(class_counts))

    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


# =========================================================
# MULTILABEL
# =========================================================

def build_multilabel_sampler(
    label_matrix: np.ndarray,
    *,
    epsilon: float = 1.0,
) -> WeightedRandomSampler:
    """
    Inverse-frequency WeightedRandomSampler for multilabel tasks.

    A modest ``epsilon`` (default 1.0 — additive smoothing) avoids the
    1e6 weight blowup that ``epsilon=1e-6`` produced for zero-positive
    columns.
    """
    label_matrix = np.asarray(label_matrix, dtype=np.float32)

    # Laplace-smoothed positive counts
    pos_counts = label_matrix.sum(axis=0) + epsilon
    label_weights = 1.0 / pos_counts

    sample_weights = (label_matrix * label_weights).sum(axis=1)
    # Rows with no positive labels get a baseline weight (mean of label_weights)
    fallback = float(label_weights.mean())
    sample_weights = np.where(sample_weights == 0, fallback, sample_weights)

    logger.info("Sampler | multilabel | labels=%d", label_matrix.shape[1])

    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


# =========================================================
# CONTRACT-DRIVEN FACTORY
# =========================================================

def build_sampler(
    *,
    task: str,
    df: pd.DataFrame,
    use_weighted: bool = True,
) -> Optional[WeightedRandomSampler]:
    """Build the right sampler for ``task`` using the task contract."""
    if not use_weighted:
        return None

    contract = get_contract(task)

    if contract.task_type == "classification":
        col = contract.label_columns[0]
        if col not in df.columns:
            raise KeyError(
                f"Sampler: column '{col}' missing for task '{task}'."
            )
        return build_classification_sampler(df[col].values)

    if contract.task_type == "multilabel":
        cols = contract.label_columns
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"Sampler: columns {missing} missing for task '{task}'."
            )
        return build_multilabel_sampler(df[cols].values)

    raise ValueError(f"Unknown task type: {contract.task_type}")
