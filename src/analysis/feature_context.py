from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from typing import List, Dict, Any, Optional

from spacy.tokens import Doc, Span

from src.analysis._text_features import extract_alpha_lemmas
from src.analysis.spacy_loader import get_doc


# =========================================================
# FEATURE CONTEXT (FINAL ARCHITECTURE)
# =========================================================

@dataclass(slots=True)
class FeatureContext:
    """
    Unified context object for all analysis + feature pipelines.

    Design Goals:
    - lazy spaCy execution (via get_doc)
    - shared cache across analyzers
    - zero recomputation
    - batch-friendly
    """

    # -----------------------------------------------------
    # CORE INPUT
    # -----------------------------------------------------

    text: str

    # -----------------------------------------------------
    # SHARED SYSTEM (CRITICAL)
    # -----------------------------------------------------

    shared: Dict[str, Any] = field(default_factory=dict)
    cache: Dict[str, Any] = field(default_factory=dict)

    # -----------------------------------------------------
    # LAZY COMPUTED FIELDS
    # -----------------------------------------------------

    text_lower: Optional[str] = None
    tokens: Optional[List[str]] = None
    token_counts: Optional[Counter] = None
    n_tokens: Optional[int] = None

    sentences: Optional[List[Span]] = None
    entities: Optional[List[Span]] = None

    pos_counts: Optional[Dict[str, int]] = None
    dep_counts: Optional[Dict[str, int]] = None

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __post_init__(self):

        if not isinstance(self.text, str):
            raise TypeError("FeatureContext.text must be a string")

        if self.shared is None:
            self.shared = {}

        if self.cache is None:
            self.cache = {}

    # =========================================================
    # 🔥 DOC ACCESS (LAZY + SHARED)
    # =========================================================

    def get_doc(self, task: str = "syntax") -> Doc:
        return get_doc(self, task)

    # =========================================================
    # FACTORY: SEED FROM PRECOMPUTED DOC
    # =========================================================

    @classmethod
    def from_doc(cls, doc: Doc, mode: str = "safe") -> "FeatureContext":
        """
        Build a FeatureContext from an already-processed spaCy Doc.

        Pre-seeds the shared spaCy cache only for the tasks that the
        provided pipeline actually supports (driven by ``mode``):

        - ``"safe"`` (default): full pipeline — seed both ``"syntax"``
          and ``"ner"`` slots so analyzers reuse the doc without a
          re-parse.
        - ``"fast"`` / unknown modes: the doc was produced by a stripped
          pipeline (no NER / tagger / parser / lemmatizer), so we do
          NOT seed those task slots. Downstream analyzers that ask for
          ``get_doc(ctx, "ner"|"syntax")`` will then lazily re-parse
          with the correct task pipeline instead of silently iterating
          over an empty entity / dep / pos view (CRIT-A7).
        """
        if doc is None:
            raise ValueError("doc must not be None")

        ctx = cls(text=doc.text)
        seeded = ctx.shared.setdefault("spacy_docs", {})

        if mode == "safe":
            seeded["syntax"] = doc
            seeded["ner"] = doc
        # For "fast" or any non-safe mode the doc lacks the components
        # that "syntax"/"ner" require, so leave those slots empty and
        # let `get_doc` re-parse on demand via the task-specific NLP.

        return ctx

    # =========================================================
    # TOKEN FEATURES (LAZY)
    # =========================================================

    def ensure_tokens(self, task: str = "syntax") -> None:

        if self.tokens is not None:
            return

        doc = self.get_doc(task)

        self.text_lower = doc.text.lower()

        tokens = extract_alpha_lemmas(doc)

        self.tokens = tokens
        self.token_counts = Counter(tokens)
        self.n_tokens = len(tokens)

    # =========================================================
    # SENTENCE / ENTITY CACHE (LAZY)
    # =========================================================

    def ensure_structure(self, task: str = "syntax") -> None:

        if self.sentences is not None and self.entities is not None:
            return

        doc = self.get_doc(task)

        self.sentences = list(doc.sents)
        self.entities = list(doc.ents)

    # =========================================================
    # POS / DEP CACHE (LAZY)
    # =========================================================

    def ensure_linguistics(self, task: str = "syntax") -> None:

        if self.pos_counts is not None and self.dep_counts is not None:
            return

        doc = self.get_doc(task)

        pos_counts: Dict[str, int] = {}
        dep_counts: Dict[str, int] = {}

        for token in doc:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
            dep_counts[token.dep_] = dep_counts.get(token.dep_, 0) + 1

        self.pos_counts = pos_counts
        self.dep_counts = dep_counts

    # =========================================================
    # SAFE ACCESS HELPERS
    # =========================================================

    def safe_n_tokens(self) -> int:
        return self.n_tokens or 0

    def safe_tokens(self) -> List[str]:
        return self.tokens or []

    def safe_counts(self) -> Counter:
        return self.token_counts or Counter()

    # PERF-A1: cache `text.count(symbol)` once per (ctx, symbol). Several
    # analyzers compute exclamation/question densities; before this, each
    # one re-scanned the raw text per request. Stored on `shared` rather
    # than as a slot field so we don't have to widen the dataclass slots.
    def punct_count(self, symbol: str) -> int:
        cache = self.shared.setdefault("punct_counts", {})
        cached = cache.get(symbol)
        if cached is not None:
            return cached
        value = (self.text or "").count(symbol)
        cache[symbol] = value
        return value

    # =========================================================
    # UTILITIES
    # =========================================================

    def has_entities(self) -> bool:
        return bool(self.entities)

    def sentence_count(self) -> int:
        return len(self.sentences) if self.sentences else 0

    def entity_count(self) -> int:
        return len(self.entities) if self.entities else 0

    def get_pos_ratio(self, pos_tag: str) -> float:
        if not self.pos_counts:
            return 0.0
        total = max(sum(self.pos_counts.values()), 1)
        return self.pos_counts.get(pos_tag, 0) / total

    def get_dep_ratio(self, dep_tag: str) -> float:
        if not self.dep_counts:
            return 0.0
        total = max(sum(self.dep_counts.values()), 1)
        return self.dep_counts.get(dep_tag, 0) / total

    # =========================================================
    # DEBUG
    # =========================================================

    def summary(self) -> Dict[str, Any]:

        return {
            "text_length": len(self.text),
            "n_tokens": self.n_tokens or 0,
            "n_sentences": len(self.sentences) if self.sentences else 0,
            "n_entities": len(self.entities) if self.entities else 0,
            "vocab_size": len(self.token_counts) if self.token_counts else 0,
        }