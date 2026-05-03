"""Shared sentence segmentation + heuristic entity extraction.

Audit fix §4 — five different sentence splitters
(``_simple_sentence_split``, ``_sentence_split``, ``_split_sentences``,
``_SENT_SPLIT_RE``) and two identical ``_heuristic_entities`` helpers
were duplicated across ``text/syntactic_features.py``,
``emotion/emotion_trajectory_features.py``, ``graph/entity_graph_features.py``
and ``graph/interaction_graph_features.py``. They diverged subtly (some
collapsed multi-character runs, some did not; some preserved the
trailing punctuation, some did not), which made the per-sentence stats
the trajectory / graph extractors emit slightly inconsistent across
modules.

This module owns the canonical implementations; downstream modules just
import. The implementations are deliberately conservative — when a
callable spaCy ``Doc`` is available the caller should prefer
``doc.sents`` and only fall back here when spaCy is unavailable.
"""

from __future__ import annotations

import re
from typing import List

# Greedy splitter on the canonical sentence-terminating punctuation. Runs
# of ``!``, ``?``, ``.`` collapse so that ``"Wait!! Stop?"`` yields two
# sentences rather than four. Newlines also break sentences because
# many news bodies arrive un-punctuated paragraph-per-line.
_SENT_SPLIT_RE = re.compile(r"[.!?\n]+")

# Capitalised-token heuristic. Used by the graph extractors when spaCy
# NER is unavailable. ASCII-only on purpose — the heuristic targets
# proper-noun heads (``Putin``, ``Pentagon``) and the upstream
# tokenisation in those modules is also ASCII.
_HEURISTIC_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]+\b")


def split_sentences(text: str) -> List[str]:
    """Split ``text`` into stripped, non-empty sentences.

    Cheap regex-based splitter for code paths that cannot afford a
    spaCy parse. For high-quality segmentation prefer
    ``ensure_spacy_doc(ctx).sents``.
    """
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def heuristic_entities(sentence: str) -> List[str]:
    """Return distinct capitalised tokens treated as proper-noun candidates.

    Ordering is intentionally not preserved — the graph builders only
    consume the set of unique entities per sentence, and dropping the
    ``set(...)`` round-trip from the call sites is the whole point of
    centralising the helper.
    """
    if not sentence:
        return []
    return list(set(_HEURISTIC_ENTITY_RE.findall(sentence)))
