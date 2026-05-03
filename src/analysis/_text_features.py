from __future__ import annotations

import re
import logging
from collections import Counter, OrderedDict
from typing import Collection, List, Iterable, Dict, Tuple, Any

from spacy.tokens import Doc

logger = logging.getLogger(__name__)


# =========================================================
# CONSTANTS
# =========================================================

EPS = 1e-8
MAX_REGEX_CACHE = 512


# =========================================================
# TOKEN EXTRACTION (OPTIMIZED)
# =========================================================

def extract_alpha_lemmas(doc: Doc) -> List[str]:
    return [
        lem
        for token in doc
        if token.is_alpha and not token.is_stop
        for lem in (token.lemma_.lower(),)
        if lem.isalpha()
    ]


def extract_alpha_tokens(doc: Doc) -> List[str]:
    return [
        token.text.lower()
        for token in doc
        if token.is_alpha
    ]


# =========================================================
# COUNTERS
# =========================================================

def build_counter(tokens: List[str]) -> Counter:
    return Counter(tokens)


def word_count(tokens: List[str]) -> int:
    return len(tokens)


def unique_token_count(tokens: List[str]) -> int:
    return len(set(tokens))


# =========================================================
# TERM FEATURES
# =========================================================

def term_ratio(
    token_counts: Counter,
    n_tokens: int,
    lexicon: Collection[str],
) -> float:

    if n_tokens <= 0 or not lexicon:
        return 0.0

    hits = sum(token_counts.get(t, 0) for t in lexicon)
    return float(hits / (n_tokens + EPS))


def term_presence(
    token_counts: Counter,
    lexicon: Collection[str],
) -> int:
    return int(any(t in token_counts for t in lexicon))


# =========================================================
# REGEX CACHE
# =========================================================

# LRU cache: most-recently-used keys live at the end. We evict the
# least-recently-used (first) entry when the cache exceeds capacity.
_REGEX_CACHE: "OrderedDict[Tuple[str, ...], List[re.Pattern]]" = OrderedDict()


def _compile_patterns(phrases: Iterable[str]) -> List[re.Pattern]:

    key = tuple(sorted(p.strip().lower() for p in phrases if p))

    cached = _REGEX_CACHE.get(key)
    if cached is not None:
        # mark this entry as most-recently-used
        _REGEX_CACHE.move_to_end(key)
        return cached

    compiled: List[re.Pattern] = []

    for phrase in key:
        if not phrase:
            continue

        if " " in phrase:
            pattern = re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)")
        else:
            pattern = re.compile(r"\b" + re.escape(phrase) + r"\b")

        compiled.append(pattern)

    _REGEX_CACHE[key] = compiled

    # Evict least-recently-used entries until we're back within budget.
    while len(_REGEX_CACHE) > MAX_REGEX_CACHE:
        _REGEX_CACHE.popitem(last=False)

    return compiled


# =========================================================
# PHRASE FEATURES
# =========================================================

def phrase_match_count(
    text_lower: str,
    phrases: Collection[str],
    *,
    word_boundary: bool = True,
) -> int:

    if not text_lower or not phrases:
        return 0

    if not word_boundary:
        return sum(1 for phrase in phrases if phrase and phrase in text_lower)

    patterns = _compile_patterns(phrases)

    return sum(1 for pattern in patterns if pattern.search(text_lower))


# PERF-A2: shared per-context phrase-hit cache. Each analyzer holds a
# stable reference to its own lexicon set, so `id(phrases)` is a safe and
# cheap key. Repeated calls with the same lexicon during a single
# request reuse the prior scan instead of re-running the regex sweep.
def cached_phrase_match_count(
    ctx: Any,
    phrases: Collection[str],
    *,
    key: Any = None,
    word_boundary: bool = True,
) -> int:

    if ctx is None or not phrases:
        return phrase_match_count(
            getattr(ctx, "text_lower", "") or "",
            phrases,
            word_boundary=word_boundary,
        )

    if key is None:
        key = id(phrases)

    shared = getattr(ctx, "shared", None)
    if shared is None:
        return phrase_match_count(
            ctx.text_lower or "",
            phrases,
            word_boundary=word_boundary,
        )

    cache = shared.setdefault("phrase_hits", {})
    cached = cache.get(key)
    if cached is not None:
        return cached

    value = phrase_match_count(
        ctx.text_lower or "",
        phrases,
        word_boundary=word_boundary,
    )
    cache[key] = value
    return value


def phrase_frequency(
    text_lower: str,
    phrases: Collection[str],
) -> int:

    if not text_lower or not phrases:
        return 0

    patterns = _compile_patterns(phrases)

    return sum(len(pattern.findall(text_lower)) for pattern in patterns)


# =========================================================
# NORMALIZATION
# =========================================================

# NUM-A1: shared, well-guarded normalized-Shannon-entropy helper.
# Promotes the n<=1 / max-entropy<EPS / sum<EPS guards from
# propaganda_pattern_detector so every analyzer that wants a normalized
# entropy gets the same numerically-stable behavior. Returns a value in
# [0, 1] (or 0.0 for degenerate inputs).
def safe_normalized_entropy(values: Iterable[float]) -> float:

    import numpy as _np  # local import to keep module import light

    arr = _np.asarray(list(values), dtype=_np.float32)

    if arr.size == 0:
        return 0.0

    total = float(arr.sum())
    if total < EPS:
        return 0.0

    probs = arr / total

    n = arr.size
    if n <= 1:
        return 0.0

    entropy = -float(_np.sum(probs * _np.log(probs + EPS)))
    max_entropy = float(_np.log(n))

    if max_entropy < EPS:
        return 0.0

    return entropy / max_entropy


def normalize_lexicon_terms(terms: Collection[str]) -> set[str]:
    return {
        t.replace("_", " ").strip().lower()
        for t in terms
        if isinstance(t, str) and t.strip()
    }


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.lower().strip()


# =========================================================
#  CONTEXT-AWARE DOC ACCESS (NEW)
# =========================================================

def get_doc_from_context(context: Any, task: str):
    """
    Retrieve spaCy doc using shared cache.

    Integrates with spacy_loader.get_doc()
    """
    from src.analysis.spacy_loader import get_doc
    return get_doc(context, task)


# =========================================================
# FEATURE BUILDER
# =========================================================

def build_text_features(
    doc: Doc,
    *,
    lexicons: Dict[str, Collection[str]] | None = None,
) -> Dict[str, float]:

    tokens = extract_alpha_lemmas(doc)
    counts = build_counter(tokens)

    n_tokens = len(tokens)

    features: Dict[str, float] = {
        "word_count": float(n_tokens),
        "unique_words": float(unique_token_count(tokens)),
        "type_token_ratio": float(unique_token_count(tokens) / (n_tokens + EPS)),
    }

    if lexicons:
        for name, lex in lexicons.items():
            ratio = term_ratio(counts, n_tokens, lex)
            presence = term_presence(counts, lex)

            features[f"lexicon_{name}_ratio"] = ratio
            features[f"lexicon_{name}_presence"] = float(presence)

    return features


# =========================================================
# BATCH FEATURE EXTRACTION
# =========================================================

def build_features_batch(
    docs: Iterable[Doc],
    *,
    lexicons: Dict[str, Collection[str]] | None = None,
) -> List[Dict[str, float]]:

    return [
        build_text_features(doc, lexicons=lexicons)
        for doc in docs
    ]


# =========================================================
# VECTOR CONVERSION
# =========================================================

def features_to_vector(
    features: Dict[str, float],
    schema: List[str],
) -> List[float]:

    return [float(features.get(k, 0.0)) for k in schema]


# =========================================================
# 🔥 CONTEXT PIPELINE (NEW)
# =========================================================

def extract_features_from_context(
    context: Any,
    *,
    task: str,
    lexicons: Dict[str, Collection[str]] | None = None,
) -> Dict[str, float]:
    """
    Full pipeline:
    context → shared spaCy doc → features
    """

    doc = get_doc_from_context(context, task)
    return build_text_features(doc, lexicons=lexicons)


# =========================================================
# CACHE CONTROL
# =========================================================

def clear_regex_cache():
    _REGEX_CACHE.clear()