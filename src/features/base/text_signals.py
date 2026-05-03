"""Shared structural text-signal helpers (audit fixes §2.3 + §3.2).

Three primitives — caps_ratio, exclamation_density, and a (lazy)
named-entity mask — were duplicated across
``src/features/bias/bias_features.py``,
``src/features/bias/bias_lexicon_features.py``,
``src/features/propaganda/propaganda_features.py``,
``src/features/propaganda/propaganda_lexicon_features.py``, and
``src/features/propaganda/manipulation_patterns.py``.

Beyond the obvious DRY problem, every duplicate counted *all* uppercase
runs — including proper nouns ("USA", "FBI", "NATO") that are uppercase
for orthographic reasons rather than as emphasis. This systematically
inflated the caps signal on geopolitical text and was the dominant
false-positive source for the bias and propaganda heads.

This module:

* Computes the signals **once** per :class:`FeatureContext` and stores
  the result on ``ctx.shared`` so every downstream extractor reads from
  the cache instead of recomputing.
* Subtracts a NER mask from the caps tally when spaCy is available
  (uses ``get_shared_nlp`` so we share the analysis-layer cache).
* Weights the headline (the first sentence) higher than the body —
  emphasis is the explicit signal for headline copy and a much weaker
  signal in the body, so a 2x weighting on the headline reflects the
  prior the audit calls for.

Falls back gracefully (no NER mask, no headline boost) when spaCy is
unavailable, so the helpers can never fail the pipeline.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

EPS = 1e-8

# Headline weighting: the first sentence is the headline / lede on most
# news copy. Caps emphasis there is much stronger evidence of bias
# framing than caps in the body, so we weight it higher (audit §3.2).
HEADLINE_WEIGHT = 2.0
BODY_WEIGHT = 1.0

_SHARED_KEY = "_text_signals"

# Pre-compiled splitter for lightweight headline extraction when spaCy
# is unavailable. ``[.!?]+`` matches any sentence-ending punctuation
# run; we keep the first non-empty segment as the headline.
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


# =========================================================
# NER MASK (lazy, shared)
# =========================================================

def _named_entity_uppercase_words(text: str) -> set[str]:
    """Return the lowercased word forms of all-caps named entities in *text*.

    These tokens should NOT count toward "caps emphasis". When spaCy is
    not installed we return an empty set; the caller then falls back to
    counting every uppercase token (the previous, audit-flagged, behavior).
    """
    try:
        from src.features.base.spacy_loader import get_shared_nlp
    except Exception:
        return set()

    nlp = get_shared_nlp("en_core_web_sm")
    if nlp is None:
        return set()

    try:
        # Cap at the same 100k char limit spaCy uses by default; longer
        # documents are rare in this pipeline, but defensive.
        doc = nlp(text[:100_000])
    except Exception:
        return set()

    out: set[str] = set()
    for ent in getattr(doc, "ents", ()):
        ent_text = (ent.text or "").strip()
        if not ent_text:
            continue
        # Walk word-by-word; only the all-caps tokens go on the deny list.
        for tok in ent_text.split():
            stripped = tok.strip(".,;:!?\"'()[]{}")
            if len(stripped) > 1 and stripped.isupper():
                out.add(stripped.lower())
    return out


# =========================================================
# CORE COMPUTATION (uncached)
# =========================================================

def _split_headline_body(text: str) -> tuple[str, str]:
    """Return ``(headline, body)`` using a cheap sentence split.

    The headline is the first non-empty sentence (or the whole text if
    no terminal punctuation is present). Headline weighting is intended
    for short ledes — we therefore cap it at the first 200 chars to
    avoid a single multi-clause sentence dominating the score.
    """
    if not text:
        return "", ""
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    if not parts:
        return "", text
    headline = parts[0][:200]
    body = " ".join(parts[1:])
    return headline, body


def _count_caps_emphasis(words: List[str], deny: set[str]) -> int:
    """Count uppercase tokens (>2 chars) that are NOT named entities."""
    n = 0
    for w in words:
        # Strip leading/trailing punctuation we may have inherited from
        # ``str.split`` so "USA," still matches the deny set.
        stripped = w.strip(".,;:!?\"'()[]{}")
        if len(stripped) <= 2 or not stripped.isupper():
            continue
        if stripped.lower() in deny:
            continue
        n += 1
    return n


def _compute_signals(text: str, n_tokens: int) -> Dict[str, float]:
    """Compute ``(caps_ratio, exclamation_density, question_density)`` once for *text*.

    ``n_tokens`` is the canonical word-token count (from
    :func:`ensure_tokens_word`); using it as the denominator keeps the
    signal comparable across extractors regardless of how they happen
    to split words internally.

    Audit fix §4.3 — ``question_density`` joins the cached signals so
    propaganda + manipulation extractors stop recomputing the same
    ``text.count('?') / n`` ratio from three different files.
    """
    if not text or n_tokens <= 0:
        return {
            "caps_ratio": 0.0,
            "exclamation_density": 0.0,
            "question_density": 0.0,
        }

    deny = _named_entity_uppercase_words(text)
    headline, body = _split_headline_body(text)

    headline_caps = _count_caps_emphasis(headline.split(), deny)
    body_caps = _count_caps_emphasis(body.split(), deny)

    headline_excl = headline.count("!")
    body_excl = body.count("!")

    headline_q = headline.count("?")
    body_q = body.count("?")

    weighted_caps = HEADLINE_WEIGHT * headline_caps + BODY_WEIGHT * body_caps
    weighted_excl = HEADLINE_WEIGHT * headline_excl + BODY_WEIGHT * body_excl
    weighted_q = HEADLINE_WEIGHT * headline_q + BODY_WEIGHT * body_q

    caps_ratio = float(weighted_caps) / (float(n_tokens) + EPS)
    excl_density = float(weighted_excl) / (float(n_tokens) + EPS)
    q_density = float(weighted_q) / (float(n_tokens) + EPS)

    # All three are bounded ratios; clip defensively in case of
    # pathological inputs (a 5-word headline of "BREAKING URGENT NEWS
    # NOW NOW" would otherwise cross 1.0 due to the headline weight).
    caps_ratio = min(caps_ratio, 1.0)
    excl_density = min(excl_density, 1.0)
    q_density = min(q_density, 1.0)

    return {
        "caps_ratio": caps_ratio,
        "exclamation_density": excl_density,
        "question_density": q_density,
    }


# =========================================================
# PUBLIC API (cache-aware)
# =========================================================

def get_text_signals(context: Any, n_tokens: int) -> Dict[str, float]:
    """Return the cached structural text signals for *context*.

    Reads/writes ``context.shared[_SHARED_KEY]`` so every extractor in
    the same batch pays for the spaCy NER pass at most once. Falls back
    to ``context.cache`` (per-sample) when ``shared`` is unavailable
    (e.g. single-sample inference path).
    """
    # Prefer the batch-shared cache; fall back to the per-sample cache.
    shared: Optional[Dict[str, Any]] = getattr(context, "shared", None)
    cache_dict: Dict[str, Any]
    if isinstance(shared, dict):
        cache_dict = shared
    else:
        # ``context.cache`` is guaranteed dict by FeatureContext.__post_init__
        cache_dict = getattr(context, "cache", None) or {}

    bucket = cache_dict.get(_SHARED_KEY)
    if bucket is None:
        bucket = {}
        cache_dict[_SHARED_KEY] = bucket
        # If we created the bucket on the per-sample cache, make sure it
        # is also visible via the .shared alias (no-op if shared is None).
        if shared is None and hasattr(context, "cache"):
            context.cache.setdefault(_SHARED_KEY, bucket)

    text = getattr(context, "text", "") or ""
    # Cache key includes the token count: switching tokenization strategy
    # mid-pipeline would otherwise return a stale ratio.
    key = (id(text), n_tokens)
    cached = bucket.get(key)
    if cached is not None:
        return cached

    signals = _compute_signals(text, n_tokens)
    bucket[key] = signals
    return signals


def caps_ratio(context: Any, n_tokens: int) -> float:
    """Convenience accessor for ``get_text_signals(...).get('caps_ratio')``."""
    return float(get_text_signals(context, n_tokens).get("caps_ratio", 0.0))


def exclamation_density(context: Any, n_tokens: int) -> float:
    """Convenience accessor for the cached exclamation density."""
    return float(get_text_signals(context, n_tokens).get("exclamation_density", 0.0))


__all__ = [
    "get_text_signals",
    "caps_ratio",
    "exclamation_density",
    "HEADLINE_WEIGHT",
    "BODY_WEIGHT",
]
