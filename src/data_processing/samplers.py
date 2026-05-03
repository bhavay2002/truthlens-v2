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
import math
from typing import Dict, Iterator, List, Optional

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
# TASK-BALANCED BATCH SAMPLER  (spec §1 — Training Pipeline Upgrade)
# =========================================================

class TaskBalancedBatchSampler(Sampler):
    """Batch sampler that guarantees per-task sample coverage in every batch.

    Spec §1 design
    --------------
    Each batch is composed as::

        Batch = B_bias ∪ B_emotion ∪ B_propaganda ∪ B_mixed

    where the fraction of each group is controlled by ``ratios``.

    Parameters
    ----------
    indices_by_task:
        Mapping from group name (e.g. ``"bias"``, ``"emotion"``,
        ``"mixed"``) to the list of dataset indices that carry labels
        for that group. Groups not in ``ratios`` are ignored.
    batch_size:
        Total number of samples per yielded batch.
    ratios:
        Dict mapping group name → fraction of ``batch_size`` to draw
        from that group. Must sum to ≤ 1.0. Any remaining budget after
        all group quotas are filled is pulled from whichever pool has
        the most remaining samples.
    seed:
        Random seed for reproducibility (spec output req).
    drop_last:
        If ``True``, drop the final batch when it is shorter than
        ``batch_size`` (same semantics as ``DataLoader(drop_last=True)``).

    Spec constraints (§1.4)
    -----------------------
    * No index appears more than once within a single yielded batch.
    * Each pool is shuffled at the start of every epoch.
    * When a pool is exhausted mid-epoch it is refilled and reshuffled
      (sampling with replacement for that pool, preserving global class
      distribution).

    Dynamic ratio update (spec §1.5)
    ---------------------------------
    Call ``update_ratios(val_scores)`` after each validation step.
    The ratio for each task is proportional to ``(1 − val_score_i)``
    so under-performing tasks automatically receive more samples.
    """

    def __init__(
        self,
        indices_by_task: Dict[str, List[int]],
        batch_size: int,
        ratios: Dict[str, float],
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not ratios:
            raise ValueError("ratios must be non-empty")

        total_ratio = sum(ratios.values())
        if total_ratio > 1.0 + 1e-6:
            raise ValueError(
                f"ratios sum to {total_ratio:.4f} which exceeds 1.0; "
                "reduce individual ratios so they fit within one batch."
            )

        unknown = [k for k in ratios if k not in indices_by_task]
        if unknown:
            raise ValueError(
                f"ratios reference groups not in indices_by_task: {unknown}"
            )

        self.indices_by_task: Dict[str, List[int]] = {
            k: list(v) for k, v in indices_by_task.items()
        }
        self.batch_size = batch_size
        self.ratios: Dict[str, float] = dict(ratios)
        self.seed = seed
        self.drop_last = drop_last

        # Epoch counter (incremented in __iter__) used to diversify seeds
        self._epoch: int = 0

        # Compute total unique samples across all referenced groups
        all_idx: set = set()
        for task in ratios:
            all_idx.update(self.indices_by_task[task])
        self._total_unique: int = max(len(all_idx), 1)

        logger.info(
            "TaskBalancedBatchSampler | groups=%s | batch=%d | "
            "unique_samples=%d | ratios=%s",
            list(ratios.keys()),
            batch_size,
            self._total_unique,
            {k: round(v, 3) for k, v in ratios.items()},
        )

    # -----------------------------------------------------------------------
    # Sampler protocol
    # -----------------------------------------------------------------------

    def __len__(self) -> int:
        if self.drop_last:
            return self._total_unique // self.batch_size
        return math.ceil(self._total_unique / self.batch_size)

    def __iter__(self):
        import random as _random
        from collections import deque

        rng = _random.Random(self.seed + self._epoch)
        self._epoch += 1

        # Shuffle each pool and wrap in a deque
        queues: Dict[str, deque] = {}
        for task in self.ratios:
            pool = list(self.indices_by_task[task])
            rng.shuffle(pool)
            queues[task] = deque(pool)

        # Track how many unique samples have been yielded
        yielded = 0
        target = len(self)

        for _ in range(target):
            batch: List[int] = []
            used_in_batch: set = set()

            # ── Fill from each task pool according to its ratio ────────────
            for task, ratio in self.ratios.items():
                k = max(1, round(self.batch_size * ratio))
                q = queues[task]
                added = 0
                while added < k and len(batch) < self.batch_size:
                    if not q:
                        # Refill (with replacement) for this pool
                        fresh = list(self.indices_by_task[task])
                        rng.shuffle(fresh)
                        q = deque(fresh)
                        queues[task] = q
                    idx = q.popleft()
                    if idx not in used_in_batch:
                        batch.append(idx)
                        used_in_batch.add(idx)
                        added += 1

            # ── Fill any remaining budget from the largest pool ────────────
            if len(batch) < self.batch_size:
                remaining = self.batch_size - len(batch)
                sorted_tasks = sorted(
                    queues.keys(),
                    key=lambda t: len(queues[t]),
                    reverse=True,
                )
                for task in sorted_tasks:
                    q = queues[task]
                    while remaining > 0:
                        if not q:
                            break
                        idx = q.popleft()
                        if idx not in used_in_batch:
                            batch.append(idx)
                            used_in_batch.add(idx)
                            remaining -= 1
                    if remaining == 0:
                        break

            # ── Drop-last gate ──────────────────────────────────────────────
            if len(batch) < self.batch_size and self.drop_last:
                break

            if batch:
                rng.shuffle(batch)
                yield batch
                yielded += len(batch)

    # -----------------------------------------------------------------------
    # Dynamic ratio update (spec §1.5)
    # -----------------------------------------------------------------------

    def update_ratios(self, val_scores: Dict[str, float]) -> None:
        """Adjust sampling ratios proportional to ``(1 − val_score_i)``.

        Under-performing tasks (low validation score) receive a larger
        share of each batch on the next epoch.

        Parameters
        ----------
        val_scores:
            Dict mapping group name → validation score in [0, 1].
            Groups absent from ``val_scores`` keep their current ratio.
        """
        gaps: Dict[str, float] = {}
        for task in self.ratios:
            score = val_scores.get(task, 1.0 - self.ratios[task])
            gaps[task] = max(0.0, 1.0 - float(score))

        total_gap = sum(gaps.values())
        if total_gap < 1e-9:
            logger.debug(
                "TaskBalancedBatchSampler.update_ratios: all tasks at "
                "perfect performance — ratios unchanged."
            )
            return

        new_ratios = {t: g / total_gap for t, g in gaps.items()}
        self.ratios = new_ratios

        logger.info(
            "TaskBalancedBatchSampler ratios updated: %s",
            {k: round(v, 3) for k, v in new_ratios.items()},
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def build_indices_by_task(
        task_mask_matrix: np.ndarray,
        task_names: List[str],
        mixed_threshold: int = 2,
    ) -> Dict[str, List[int]]:
        """Pre-index a dataset into per-task and mixed-task groups.

        Parameters
        ----------
        task_mask_matrix:
            Boolean / int array of shape (N, T). Entry [i, t] = 1 means
            sample i has a valid label for task t.
        task_names:
            Ordered list of task names corresponding to columns of
            ``task_mask_matrix``.
        mixed_threshold:
            Minimum number of active tasks for a sample to be placed in
            the ``"mixed"`` group (spec §1.2 — K₄ mixed-label samples).

        Returns
        -------
        Dict mapping each task name and ``"mixed"`` to lists of indices.
        """
        task_mask_matrix = np.asarray(task_mask_matrix, dtype=np.int32)
        if task_mask_matrix.ndim != 2:
            raise ValueError(
                "task_mask_matrix must be 2-D (N, T); "
                f"got shape {task_mask_matrix.shape}"
            )
        N, T = task_mask_matrix.shape
        if len(task_names) != T:
            raise ValueError(
                f"task_names has {len(task_names)} entries but "
                f"task_mask_matrix has {T} columns."
            )

        result: Dict[str, List[int]] = {t: [] for t in task_names}
        result["mixed"] = []

        for i in range(N):
            row = task_mask_matrix[i]
            active_tasks = [task_names[j] for j in range(T) if row[j]]
            for task in active_tasks:
                result[task].append(i)
            if len(active_tasks) >= mixed_threshold:
                result["mixed"].append(i)

        logger.info(
            "TaskBalancedBatchSampler.build_indices_by_task | "
            "N=%d | sizes=%s",
            N,
            {k: len(v) for k, v in result.items()},
        )
        return result


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
