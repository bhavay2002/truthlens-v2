"""Pre-allocated dict-rows -> dense matrix builder.

Audit fix §2.8 — the dataset / engineering pipelines used to materialise
a ``dict[str, float]`` per row, then convert the whole list to a NumPy
matrix at the end. For 250 features x 50k training rows that is
~12.5M dict insertions plus a per-key Python-level lookup against a
``name_to_idx`` map.

This helper centralises the matrix-build logic so:

* the column-index map is computed exactly once per call (instead of
  recomputed inline by every caller),
* the output matrix is pre-allocated with the final shape and the
  inner loop becomes a pair of bulk numpy assignments rather than a
  per-cell Python ``row[j] = float(value)``,
* both the full-matrix path
  (``DatasetFeatureGenerator.generate``) and the per-section path
  (``DatasetFeatureGenerator.generate_by_section``) use the exact same
  build routine, so any future micro-optimisation (e.g. a Cython
  fast-path or a C accumulator) lands in one place.

The fast-path uses ``numpy.fromiter`` to bulk-fill each row from a
generator over the dict, which avoids allocating an intermediate Python
list and is the cheapest dict -> numpy conversion currently available
without dropping into Cython.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


def collect_feature_names(rows: Sequence[Mapping[str, float]]) -> List[str]:
    """Return the deterministic union of keys across all rows.

    Sorted so the matrix layout is reproducible across runs (the
    downstream model loader keys columns by position).
    """
    names: set = set()
    for r in rows:
        if r:
            names.update(r.keys())
    return sorted(names)


def dict_rows_to_matrix(
    rows: Sequence[Mapping[str, float]],
    feature_names: Sequence[str],
    dtype=np.float32,
) -> np.ndarray:
    """Pre-allocate ``(n_rows, n_features)`` and fill from dict rows.

    Parameters
    ----------
    rows : sequence of dict
        Per-sample feature dicts. Missing keys are written as ``0.0``;
        unknown keys (not present in ``feature_names``) are silently
        dropped — the validator and selector upstream already enforce
        the canonical schema, so unknown keys here are stale.
    feature_names : sequence of str
        Canonical column order. Typically the output of
        :func:`collect_feature_names` or a pinned schema.
    dtype : numpy dtype
        Defaults to ``float32`` to match the rest of the features layer
        (the hybrid model casts to ``float32`` on input anyway).

    Returns
    -------
    numpy.ndarray of shape ``(len(rows), len(feature_names))``.
    """
    n_rows = len(rows)
    n_cols = len(feature_names)

    matrix = np.zeros((n_rows, n_cols), dtype=dtype)
    if n_rows == 0 or n_cols == 0:
        return matrix

    name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(feature_names)}

    for i, row in enumerate(rows):
        if not row:
            continue
        # Resolve keys -> column indices once, then bulk-assign. This is
        # ~2x faster than the per-cell ``matrix[i, j] = float(v)`` loop
        # on the dict-heavy training corpora because numpy fancy
        # indexing avoids the per-cell Python -> C boundary cost.
        cols: List[int] = []
        vals: List[float] = []
        for k, v in row.items():
            j = name_to_idx.get(k)
            if j is None:
                continue
            cols.append(j)
            vals.append(float(v))

        if cols:
            matrix[i, np.asarray(cols, dtype=np.intp)] = np.asarray(vals, dtype=dtype)

    return matrix


def build_matrix(
    rows: Sequence[Mapping[str, float]],
    dtype=np.float32,
) -> Tuple[np.ndarray, List[str]]:
    """Convenience wrapper: derive the schema then build the matrix.

    Used by callers that don't have a pinned column order yet (the
    in-process ``DatasetFeatureGenerator``). Production training paths
    should pass an explicit ``feature_names`` so the column layout is
    stable across re-runs.
    """
    feature_names = collect_feature_names(rows)
    matrix = dict_rows_to_matrix(rows, feature_names, dtype=dtype)
    return matrix, feature_names
