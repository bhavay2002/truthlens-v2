"""Numerical helpers shared by the features layer.

Centralises the small numeric primitives that were previously inlined,
inconsistently, across ~13 feature extractors.

The headline helper is :func:`normalized_entropy`. The legacy pattern

    entropy_raw = -np.sum(probs * np.log(probs + EPS))
    entropy = entropy_raw / (np.log(len(probs)) + EPS)        # <-- BUG

silently produces garbage in two edge cases:

* ``len(probs) == 1`` -> ``np.log(1) == 0``, so the denominator collapses
  to ``EPS`` (~1e-8) and the entropy explodes to ~1e7 instead of being
  ``0.0`` (a single-bin distribution has zero entropy by definition).
* The variant ``np.log(len(probs) + EPS)`` is even worse for ``len == 0``:
  it returns a *negative* denominator (``log(EPS) < 0``), flipping the
  sign of the entropy.

:func:`normalized_entropy` always returns a value in ``[0.0, 1.0]`` and is
defined to be exactly ``0.0`` when there is fewer than two non-degenerate
bins, matching the information-theoretic convention.
"""

from __future__ import annotations

# Audit fix §3.1 — single source of truth for the two epsilons that used
# to be redeclared as module constants in 25+ files. Importing from here
# means a calibration sweep has exactly one place to touch.

EPS = 1e-8
MAX_CLIP = 1.0


from typing import Sequence, Union

import numpy as np


def normalized_entropy(
    probs: Union[Sequence[float], np.ndarray],
    eps: float = EPS,
) -> float:
    """Return Shannon entropy of ``probs`` normalised to ``[0, 1]``.

    ``probs`` is assumed (but not required) to be a probability vector.
    The mass is *not* renormalised here — that is the caller's job and
    keeps this helper a pure numeric primitive.

    Returns ``0.0`` when:
    * ``probs`` has fewer than 2 elements,
    * ``probs`` sums to (effectively) zero, or
    * the result is non-finite for any reason.
    """
    arr = np.asarray(probs, dtype=np.float64)
    n = arr.size
    if n < 2:
        return 0.0

    total = float(arr.sum())
    if total <= eps:
        return 0.0

    # Defensive: caller may pass un-normalised counts.
    if abs(total - 1.0) > 1e-6:
        arr = arr / total

    raw = -float(np.sum(arr * np.log(arr + eps)))
    norm = raw / np.log(n)

    if not np.isfinite(norm):
        return 0.0
    # Clamp to [0, 1]; the +eps inside the log can push raw slightly above
    # log(n) for near-uniform distributions.
    if norm < 0.0:
        return 0.0
    if norm > 1.0:
        return 1.0
    return float(norm)
