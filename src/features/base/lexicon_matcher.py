from __future__ import annotations

"""
Vectorized lexicon matchers.

Provides high-throughput primitives shared by every lexicon-style extractor
in src/features/{bias,emotion,propaganda}/.

Replaces the per-token Python loops (`for t in tokens: if t in lexicon: ...`)
that were the dominant CPU cost in the feature layer.

Two matchers:

  - `LexiconMatcher`         : unweighted set lookup, O(N) numpy isin
  - `WeightedLexiconMatcher` : Dict[str, float] weighted sum + negation-aware
                               scaling via a vectorized rolling-window mask

A precompiled regex (`pattern`) is also exposed for callers that prefer to
match against raw text (single C-level pass, ignores Python tokenization).
"""

import logging
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =========================================================
# TOKEN ARRAY UTIL
# =========================================================

def to_token_array(tokens: Iterable[str]) -> np.ndarray:
    """
    Materialize tokens as a contiguous numpy string array.

    Centralized so callers do not duplicate dtype handling.
    """
    if isinstance(tokens, np.ndarray):
        return tokens
    return np.asarray(list(tokens), dtype=object)


# =========================================================
# NEGATION MASK
# =========================================================

def compute_negation_mask(
    tokens: np.ndarray,
    negations: Set[str],
    window: int = 3,
) -> np.ndarray:
    """
    Vectorized rolling-window negation mask.

    mask[i] is True iff any token in tokens[i-window : i] is a negation.
    Uses a cumulative-sum trick → O(N) without a Python loop.
    """
    n = len(tokens)
    if n == 0 or not negations:
        return np.zeros(n, dtype=bool)

    neg_array = np.asarray(list(negations), dtype=object)
    is_neg = np.isin(tokens, neg_array).astype(np.int32)

    cumsum = np.concatenate(([0], np.cumsum(is_neg)))  # length n+1
    idx = np.arange(n)
    starts = np.maximum(0, idx - window)
    counts = cumsum[idx] - cumsum[starts]
    return counts > 0


# =========================================================
# UNWEIGHTED SET MATCHER
# =========================================================

class LexiconMatcher:
    """
    Vectorized matcher for unweighted lexicons (Set[str]).

    Two access modes, pick whichever your caller already has cheaply:

        matcher.count_in_text(text: str)      -> int  (precompiled regex)
        matcher.count_in_tokens(tokens: ndarray) -> int  (np.isin)
        matcher.hit_mask(tokens: ndarray)     -> bool ndarray of length len(tokens)
    """

    __slots__ = ("name", "vocab", "vocab_array", "pattern")

    def __init__(self, lexicon: Iterable[str], name: str = ""):
        self.name = name
        # frozenset for safety; vocab_array for vectorized lookup
        cleaned = [w.lower() for w in lexicon if isinstance(w, str) and w]
        self.vocab: frozenset = frozenset(cleaned)
        self.vocab_array: np.ndarray = (
            np.asarray(sorted(self.vocab), dtype=object)
            if self.vocab
            else np.empty(0, dtype=object)
        )
        if self.vocab:
            # \b…\b boundaries; case-insensitive at compile-time
            escaped = "|".join(re.escape(w) for w in sorted(self.vocab, key=len, reverse=True))
            self.pattern: Optional[re.Pattern] = re.compile(
                rf"\b(?:{escaped})\b", re.IGNORECASE
            )
        else:
            self.pattern = None

    # -----------------------------------------------------

    def count_in_text(self, text: str) -> int:
        if self.pattern is None or not text:
            return 0
        return len(self.pattern.findall(text))

    # -----------------------------------------------------

    def count_in_tokens(self, tokens: np.ndarray) -> int:
        if self.vocab_array.size == 0 or len(tokens) == 0:
            return 0
        return int(np.isin(tokens, self.vocab_array).sum())

    # -----------------------------------------------------

    def hit_mask(self, tokens: np.ndarray) -> np.ndarray:
        if self.vocab_array.size == 0 or len(tokens) == 0:
            return np.zeros(len(tokens), dtype=bool)
        return np.isin(tokens, self.vocab_array)


# =========================================================
# WEIGHTED DICT MATCHER
# =========================================================

class WeightedLexiconMatcher:
    """
    Vectorized matcher for weighted lexicons (Dict[str, float]).

    Provides:
        - weighted_sum(tokens)
                : Σ weight[token] over hits.
        - negation_aware_sum(tokens, negation_mask, factor=0.3)
                : same, but each hit's weight is multiplied by `factor`
                  if the precomputed negation_mask is True at that index.

    Negation masks are computed once per text via `compute_negation_mask`
    (cumsum trick) and reused across all weighted matchers for that text.
    """

    __slots__ = ("name", "_lookup", "weights")

    def __init__(self, lexicon, name: str = ""):
        self.name = name

        # Accept Dict[str, float] OR any iterable of strings (treated as
        # weight=1.0). This is intentional: some upstream extractors
        # ship placeholder lexicons until calibrated weights land, and
        # we never want a TypeError to take the pipeline down.
        if isinstance(lexicon, dict):
            items: List[Tuple[str, float]] = list(lexicon.items())
        elif lexicon is None:
            items = []
        else:
            try:
                items = [(w, 1.0) for w in lexicon]
            except TypeError:
                items = []

        cleaned: List[Tuple[str, float]] = [
            (w.lower(), float(v))
            for w, v in items
            if isinstance(w, str) and w
        ]

        if cleaned:
            words, weights = zip(*cleaned)
            # int index per word for O(1) lookup
            self._lookup: Dict[str, int] = {w: i for i, w in enumerate(words)}
            self.weights: np.ndarray = np.asarray(weights, dtype=np.float32)
        else:
            self._lookup = {}
            self.weights = np.empty(0, dtype=np.float32)

    # -----------------------------------------------------

    def _hit_indices(self, tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (positions, weight_indices) for tokens that are in the lexicon."""
        if not self._lookup or len(tokens) == 0:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty

        # Vectorized lookup using a Python comprehension over hits only
        # (cheap because we filter first via np.isin against the keys).
        vocab_array = np.asarray(list(self._lookup.keys()), dtype=object)
        mask = np.isin(tokens, vocab_array)
        if not mask.any():
            empty = np.empty(0, dtype=np.int64)
            return empty, empty

        positions = np.flatnonzero(mask)
        # tokens[positions] is small (only hits) — dict lookup is fine
        weight_idx = np.fromiter(
            (self._lookup[t] for t in tokens[positions]),
            dtype=np.int64,
            count=positions.size,
        )
        return positions, weight_idx

    # -----------------------------------------------------

    def weighted_sum(self, tokens: np.ndarray) -> float:
        positions, weight_idx = self._hit_indices(tokens)
        if positions.size == 0:
            return 0.0
        return float(self.weights[weight_idx].sum())

    # -----------------------------------------------------

    def negation_aware_sum(
        self,
        tokens: np.ndarray,
        negation_mask: np.ndarray,
        factor: float = 0.3,
    ) -> float:
        positions, weight_idx = self._hit_indices(tokens)
        if positions.size == 0:
            return 0.0

        scale = np.where(negation_mask[positions], np.float32(factor), np.float32(1.0))
        return float((self.weights[weight_idx] * scale).sum())
