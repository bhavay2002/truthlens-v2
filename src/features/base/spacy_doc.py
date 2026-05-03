"""Shared per-context spaCy ``Doc`` cache.

Audit fix §2.7 — the entity-graph and interaction-graph extractors used
to call ``EntityGraphBuilder.build_graph(text)`` and
``NarrativeGraphBuilder.build_graph(text)`` independently on the same
text. Both delegate to ``self.nlp(text)`` under the hood, so the spaCy
parser ran twice for every document — and on long-form articles the
parser is the dominant cost in the entire feature layer.

The helper here parses the document once, caches the resulting ``Doc``
on ``FeatureContext.cache['spacy_doc']``, and returns the cached object
on every subsequent call within the same request. Extractors that are
already ``Doc``-aware (the two graph features, the syntactic extractor,
the narrative-role extractor, the emotion-target extractor) all share
that single parse.

The cache is intentionally per-``FeatureContext`` and not process-wide:

* ``Doc`` objects are large (~10x the size of the underlying text) and
  pinning them across requests would defeat the whole point of feature
  caching.
* Two requests with the same text are extremely rare in production
  (every URL/article is unique); when they do collide the
  :class:`CacheManager` already memoises the *output* feature dict, so
  we never re-parse anyway.

Failure mode: if spaCy isn't installed, or the requested model failed to
load on a previous attempt, this returns ``None`` and callers fall back
to their existing regex / heuristic path. We **never** raise — feature
extractors are expected to degrade gracefully when spaCy is missing.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.features.base.spacy_loader import get_shared_nlp

logger = logging.getLogger(__name__)


_DOC_CACHE_KEY = "spacy_doc"


def ensure_spacy_doc(
    context: Any,
    model_name: str = "en_core_web_sm",
    text: Optional[str] = None,
) -> Optional[Any]:
    """Return ``context.cache['spacy_doc']``, parsing once on first access.

    Parameters
    ----------
    context : FeatureContext
        The per-request context. Must expose a ``cache`` dict (the
        canonical :class:`FeatureContext` always does).
    model_name : str
        spaCy model to use. Defaults to ``en_core_web_sm`` to match the
        rest of the features layer.
    text : str, optional
        Override for ``context.text``. Almost always omitted; exposed
        only so the helper composes with the few extractors that strip
        whitespace before parsing.

    Returns
    -------
    spacy.tokens.Doc or None
        The parsed document, or ``None`` if spaCy / the model is
        unavailable. Callers are expected to treat ``None`` as "no
        spaCy" and run their fallback path.
    """
    cache = getattr(context, "cache", None)
    if cache is not None:
        cached = cache.get(_DOC_CACHE_KEY)
        if cached is not None:
            return cached

    nlp = get_shared_nlp(model_name)
    if nlp is None:
        return None

    src = text if text is not None else (getattr(context, "text", "") or "")
    if not src:
        return None

    try:
        doc = nlp(src)
    except Exception as exc:
        logger.warning("ensure_spacy_doc: parse failed (%s)", exc)
        return None

    if cache is not None:
        cache[_DOC_CACHE_KEY] = doc
    return doc


def set_spacy_doc(context: Any, doc: Any) -> None:
    """Seed the shared cache from an extractor that already parsed the doc.

    Use this from the first ``Doc``-aware extractor in a pipeline (e.g.
    the syntactic extractor's ``extract_batch`` after ``nlp.pipe``) so
    the graph extractors that run later in the same request never hit
    the spaCy parser at all.
    """
    cache = getattr(context, "cache", None)
    if cache is not None and doc is not None:
        cache[_DOC_CACHE_KEY] = doc
