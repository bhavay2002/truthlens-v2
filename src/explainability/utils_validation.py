from __future__ import annotations

import math
from typing import Sequence, List, Dict, Any, Tuple, Union

import numpy as np

EPS = 1e-12


def validate_tokens_scores(
    tokens: Sequence[str],
    scores: Sequence[float],

    #  NEW FEATURES
    enforce_range: bool = False,
    normalized: bool = False,
    allow_duplicates: bool = True,
    auto_fix: bool = False,
    return_fixed: bool = False,
) -> None | Tuple[List[str], List[float]]:
    """
    🔥 UPGRADED VALIDATOR

    Supports:
    - strict validation
    - normalization checks
    - duplicate handling
    - optional auto-fix mode
    """

    # =====================================================
    # TYPE CHECKS
    # =====================================================

    if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes)):
        raise TypeError("tokens must be a sequence of strings")

    if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes)):
        raise TypeError("scores must be a sequence of numeric values")

    if len(tokens) == 0 or len(scores) == 0:
        raise ValueError("Empty tokens or scores")

    if len(tokens) != len(scores):
        raise ValueError("tokens and scores must match length")

    # =====================================================
    # TOKEN VALIDATION
    # =====================================================

    for token in tokens:
        if not isinstance(token, str):
            raise TypeError("all tokens must be strings")

    # =====================================================
    # SCORE VALIDATION
    # =====================================================

    fixed_scores = []

    for score in scores:
        if not isinstance(score, (int, float)):
            if auto_fix:
                score = 0.0
            else:
                raise TypeError("all scores must be numeric")

        score = float(score)

        if not math.isfinite(score):
            if auto_fix:
                score = 0.0
            else:
                raise ValueError("scores must be finite")

        fixed_scores.append(score)

    scores_arr = np.array(fixed_scores, dtype=float)

    # =====================================================
    # RANGE CHECK
    # =====================================================

    if enforce_range:
        if auto_fix:
            scores_arr = np.clip(scores_arr, 0.0, 1.0)
        else:
            if np.any((scores_arr < 0.0) | (scores_arr > 1.0)):
                raise ValueError("scores must be in [0,1]")

    # =====================================================
    # NORMALIZATION CHECK
    # =====================================================

    if normalized:
        total = float(np.sum(scores_arr))

        if auto_fix:
            scores_arr = scores_arr / (total + EPS)
        else:
            if not (0.99 <= total <= 1.01):
                raise ValueError("scores must sum to 1")

    # =====================================================
    # DUPLICATE HANDLING
    # =====================================================

    if not allow_duplicates:
        token_map: Dict[str, float] = {}

        for t, s in zip(tokens, scores_arr):
            token_map[t] = token_map.get(t, 0.0) + s

        tokens = list(token_map.keys())
        scores_arr = np.array(list(token_map.values()), dtype=float)

    # =====================================================
    # SIGNAL CHECK (IMPORTANT)
    # =====================================================

    if len(scores_arr) > 1:
        if np.std(scores_arr) < 1e-6:
            # weak explanation signal
            if not auto_fix:
                raise ValueError("scores have near-zero variance")

    # =====================================================
    # OUTPUT
    # =====================================================

    if return_fixed:
        return list(tokens), scores_arr.tolist()

    return None