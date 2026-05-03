"""
Per-task samplers.

Label column names come from ``data_contracts`` so this module stays in
sync with cleaning, validation and the dataset factory.

New in this version:
  TaskPresenceMaskSampler — weights samples by the number of tasks for
  which they carry valid labels, ensuring batches have balanced cross-task
  label coverage rather than being dominated by single-task rows.
"""

from __future__ import annotations

import logging
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler, Sampler

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
# TASK-PRESENCE MASK SAMPLER  (cross-task balanced sampling)
# =========================================================

class TaskPresenceMaskSampler(Sampler):
    """Weighted sampler that prioritises rows with richer cross-task labels.

    In a partially-supervised multi-task dataset many rows carry labels for
    only one task. Naïve random sampling then produces batches dominated by
    single-task rows and starves the shared encoder of cross-task gradient
    signal. This sampler weights each row by:

        w_i = sum_t( mask_{i,t} ) * diversity_boost

    where ``mask_{i,t} = 1`` iff row *i* has a valid label for task *t*.
    Rows with labels for *k* tasks receive *k* times the base weight, so
    multi-task rows are sampled proportionally more often.

    An optional ``class_weights`` dict (task → per-class weight vector) can
    be folded in to further up-weight rare-class rows within each task.

    Parameters
    ----------
    task_mask_matrix:
        Boolean / int array of shape (N, T) where N = dataset size, T = number
        of tasks. Entry [i, t] is 1 if sample i has a valid label for task t.
    num_samples:
        Total samples to draw. Defaults to N (one full epoch).
    replacement:
        Whether to sample with replacement. Defaults to True.
    diversity_boost:
        Multiplier applied to the per-row task count before normalising.
        Values > 1 increase the relative weight of multi-task rows.
    min_weight:
        Floor weight so that single-task rows are never completely ignored.
    """

    def __init__(
        self,
        task_mask_matrix: np.ndarray,
        *,
        num_samples: Optional[int] = None,
        replacement: bool = True,
        diversity_boost: float = 1.5,
        min_weight: float = 0.1,
    ) -> None:
        task_mask_matrix = np.asarray(task_mask_matrix, dtype=np.float32)
        if task_mask_matrix.ndim != 2:
            raise ValueError(
                "task_mask_matrix must be 2-D (N, T); "
                f"got shape {task_mask_matrix.shape}"
            )

        n = task_mask_matrix.shape[0]
        # per-row task count (number of tasks with a valid label)
        task_counts = task_mask_matrix.sum(axis=1)
        # boost multi-task rows and apply floor
        weights = np.maximum(task_counts * diversity_boost, min_weight)
        # normalise to sum = N so the effective epoch length is preserved
        weights = weights / weights.sum() * n

        self._weights = torch.tensor(weights, dtype=torch.double)
        self._num_samples = int(num_samples) if num_samples is not None else n
        self._replacement = replacement

        n_multitask = int((task_counts > 1).sum())
        logger.info(
            "TaskPresenceMaskSampler | rows=%d | tasks=%d | multi-task rows=%d "
            "(%.1f%%) | diversity_boost=%.2f",
            n,
            task_mask_matrix.shape[1],
            n_multitask,
            100.0 * n_multitask / max(n, 1),
            diversity_boost,
        )

    def __iter__(self) -> Iterator[int]:
        indices = torch.multinomial(
            self._weights,
            num_samples=self._num_samples,
            replacement=self._replacement,
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self._num_samples


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
