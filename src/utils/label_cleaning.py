"""
Label cleaning utilities.

Multilabel datasets often arrive with degenerate columns — labels that
contain only positives or only negatives in the training split. Examples
hit on the emotion task:

    emotion__emotion_0 → classes=1   (all-zero column)

These columns:
  * Provide no decision boundary (one class only).
  * Cannot be balanced (``pos`` or ``neg`` is zero → division by zero).
  * Pollute the gradient: ``BCEWithLogitsLoss`` happily produces a
    finite loss against an all-zero target, which the head obeys by
    pushing its logit toward ``-inf`` — wasting model capacity and
    biasing the shared encoder.

The fix is the same one used in production NLP pipelines: detect these
columns from the *training* split and drop them everywhere downstream
(dataset, loss, metrics). The training split is the source of truth so
that a rare positive that only appears in the validation set does not
silently get masked out at training time.
"""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, "list[list[float]]"]


def remove_single_class_columns(
    labels: ArrayLike,
    *,
    min_pos: int = 1,
    min_neg: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Drop multilabel columns that contain only one class.

    A column is kept iff it has at least ``min_pos`` positive samples
    AND at least ``min_neg`` negative samples after binarisation
    (``label > 0``). Soft labels are supported: anything strictly
    greater than zero counts as a positive observation.

    Args:
        labels: ``(N, C)`` array-like of multilabel targets.
        min_pos: minimum number of positives required to keep a column.
        min_neg: minimum number of negatives required to keep a column.

    Returns:
        ``(filtered_labels, valid_mask)`` where:
          * ``filtered_labels`` has shape ``(N, K)`` with ``K <= C``.
          * ``valid_mask`` is a boolean array of length ``C``; ``True``
            entries correspond to surviving columns.

    Notes:
        The mask is returned (rather than just the indices) so callers
        can also align companion arrays — e.g. a ``pos_weight`` tensor
        computed on the original column space — without an extra pass.
    """
    arr = np.asarray(labels)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(
            f"remove_single_class_columns expects a 2-D array; got shape {arr.shape}"
        )

    if arr.shape[0] == 0:
        # Empty input: keep no columns rather than guess.
        return arr, np.zeros(arr.shape[1], dtype=bool)

    binary = (arr > 0).astype(np.int64)
    pos = binary.sum(axis=0)
    neg = arr.shape[0] - pos

    valid_mask = (pos >= int(min_pos)) & (neg >= int(min_neg))

    return arr[:, valid_mask], valid_mask


def valid_indices_from_mask(mask: np.ndarray) -> List[int]:
    """Convert a boolean mask to a list of kept column indices."""
    return [int(i) for i, keep in enumerate(np.asarray(mask, dtype=bool)) if keep]
