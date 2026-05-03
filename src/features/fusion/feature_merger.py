from __future__ import annotations

"""
Feature Merger

Responsible for:
- Combining feature outputs from multiple extractors
- Resolving duplicate keys
- Enforcing numeric consistency
- Providing debug visibility

This is the ONLY place where feature dictionaries are merged.
"""

import logging
import numbers
from typing import Dict, List

logger = logging.getLogger(__name__)

FeatureDict = Dict[str, float]


# =========================================================
# CORE MERGE FUNCTION
# =========================================================

def merge_features(
    feature_dicts: List[FeatureDict],
    *,
    strict: bool = False,
) -> FeatureDict:
    """
    Merge multiple feature dictionaries.

    Parameters
    ----------
    feature_dicts : List[Dict[str, float]]
    strict : bool
        If True → raises error on duplicate keys
        If False → last-write-wins

    Returns
    -------
    Dict[str, float]
    """

    merged: FeatureDict = {}
    duplicates = []

    for idx, fd in enumerate(feature_dicts):

        if not isinstance(fd, dict):
            logger.warning("Skipping non-dict feature output at index %d", idx)
            continue

        for key, value in fd.items():

            # -----------------------------
            # Type safety
            # -----------------------------
            if not isinstance(value, numbers.Number):
                logger.warning(
                    "Non-numeric feature skipped | key=%s type=%s",
                    key,
                    type(value),
                )
                continue

            value = float(value)

            # -----------------------------
            # Duplicate handling
            # -----------------------------
            if key in merged:
                duplicates.append(key)

                if strict:
                    raise ValueError(f"Duplicate feature detected: {key}")

            merged[key] = value

    # -----------------------------
    # Debug logging
    # -----------------------------
    if duplicates:
        logger.debug(
            "Duplicate features overwritten (last-write-wins): %s",
            list(set(duplicates))[:10],
        )

    return merged


# =========================================================
# ADVANCED MERGE (OPTIONAL)
# =========================================================

def merge_with_priority(
    feature_dicts: List[FeatureDict],
    priorities: List[int],
) -> FeatureDict:
    """
    Merge with priority-based resolution.

    Higher priority value → wins conflicts.

    Example:
        priorities = [1, 2, 3] → last has highest priority
    """

    if len(feature_dicts) != len(priorities):
        raise ValueError("feature_dicts and priorities must match length")

    merged: Dict[str, tuple[float, int]] = {}

    for fd, priority in zip(feature_dicts, priorities):

        for key, value in fd.items():

            if not isinstance(value, numbers.Number):
                continue

            value = float(value)

            if key not in merged or priority >= merged[key][1]:
                merged[key] = (value, priority)

    return {k: v[0] for k, v in merged.items()}


# =========================================================
# SAFE MERGE WITH DEFAULTS
# =========================================================

def merge_with_schema(
    feature_dicts: List[FeatureDict],
    expected_features: List[str],
) -> FeatureDict:
    """
    Merge + enforce schema completeness.

    Missing features → filled with 0.0
    """

    merged = merge_features(feature_dicts)

    for key in expected_features:
        merged.setdefault(key, 0.0)

    return merged


# =========================================================
# DEBUG UTIL
# =========================================================

def inspect_merge(feature_dicts: List[FeatureDict]) -> Dict[str, int]:
    """
    Returns merge diagnostics:
    - feature frequency
    - duplicates
    """

    counter: Dict[str, int] = {}

    for fd in feature_dicts:
        for key in fd.keys():
            counter[key] = counter.get(key, 0) + 1

    return counter