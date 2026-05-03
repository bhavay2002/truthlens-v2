from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from src.features.base.spacy_loader import get_shared_nlp

logger = logging.getLogger(__name__)
EPS = 1e-12


# =========================================================
# FEATURES
# =========================================================

@dataclass(slots=True)
class NarrativeGraphFeatures:

    narrative_graph_nodes: float
    narrative_graph_edges: float
    narrative_graph_avg_degree: float
    narrative_graph_density: float
    narrative_graph_isolated_nodes: float
    narrative_graph_components: float

    # 🔥 NEW
    narrative_graph_entropy: float
    narrative_graph_centralization: float
    narrative_graph_flow_strength: float

    def to_dict(self) -> Dict[str, float]:
        # ``slots=True`` strips ``__dict__``; build via ``__slots__``.
        return {f: getattr(self, f) for f in self.__slots__}


# =========================================================
# HELPERS
# =========================================================

def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def _regex_keyword_spans(
    sentence: str,
    sentence_offset: int,
    min_len: int,
    top_k: int,
) -> List[Tuple[str, int, int]]:
    """Regex fallback (used only when spaCy is unavailable).

    Returns ``(token, start_char, end_char)`` triples — char offsets are
    absolute (relative to the original ``text``), not relative to the
    sentence, so callers can use them directly for span recovery.
    """
    counts: Counter = Counter()
    positions: Dict[str, Tuple[int, int]] = {}

    for m in re.finditer(r"\b[a-zA-Z]+\b", sentence):
        tok = m.group(0).lower()
        if len(tok) < min_len:
            continue
        counts[tok] += 1
        if tok not in positions:
            positions[tok] = (sentence_offset + m.start(), sentence_offset + m.end())

    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    out: List[Tuple[str, int, int]] = []
    for tok, _ in ranked[:top_k]:
        s, e = positions[tok]
        out.append((tok, s, e))
    return out


# =========================================================
# BUILDER
# =========================================================

class NarrativeGraphBuilder:
    """Build a sentence-level narrative co-occurrence graph.

    G-S1 / G-T2 fix
    ----------------
    The previous implementation used a hand-rolled regex tokenizer
    (``\b[a-zA-Z]+\b``) that disagreed with the spaCy tokenizer used by
    every other layer of the project. That meant a "node" in the
    narrative graph could not be reliably mapped back to either a spaCy
    token or to a model subword — and nothing tied the chosen keywords
    to a real linguistic unit.

    We now drive keyword extraction off the **shared spaCy pipeline**
    (the same one used by ``EntityGraphBuilder`` and the rest of
    ``src/analysis/``):

      * **Nodes** are noun-chunks and named entities — real linguistic
        units, with stable spans into the source text.
      * **Edges** are added on two principles:
          1. *Shared sentence membership* — every pair of keywords that
             co-occur in the same sentence is linked (real
             co-occurrence, not a top-k×top-k bigram chain).
          2. *Temporal succession* — every keyword in sentence ``i`` is
             also linked to every keyword in sentence ``i+1`` (preserves
             the chain semantics of the previous implementation, but on
             real linguistic units instead of regex tokens).

    The regex tokenizer is kept solely as a fallback for environments
    where ``en_core_web_sm`` is missing — ``get_shared_nlp`` returns
    ``None`` in that case and the builder transparently falls back.

    The public API (``build_graph(text) -> Dict[str, Dict[str, float]]``
    and ``extract_graph_features(graph) -> NarrativeGraphFeatures``) is
    unchanged. A new ``build_graph_with_spans(text)`` returns both the
    graph **and** the per-keyword character offsets so the API /
    explainer layer can highlight nodes back into the source text.
    """

    def __init__(
        self,
        min_token_length: int = 4,
        max_keywords_per_sentence: int = 4,
        model: str = "en_core_web_sm",
    ):

        if min_token_length < 1:
            raise ValueError("min_token_length must be >= 1")

        if max_keywords_per_sentence < 1:
            raise ValueError("max_keywords_per_sentence must be >= 1")

        self.min_token_length = min_token_length
        self.max_keywords_per_sentence = max_keywords_per_sentence

        # G-T2: share the *same* spaCy instance every other layer uses
        # so node IDs in this graph match entity IDs in the entity
        # graph and tokens used by the analysis layer.
        self.nlp = get_shared_nlp(model)

        # ``doc.sents`` requires either a parser, a senter, or a
        # sentencizer. The shared loader may return a blank pipeline
        # when ``en_core_web_sm`` is missing — add the cheap
        # rule-based sentencizer in that case so iteration doesn't
        # raise ``E030``.
        if self.nlp is not None and not (
            self.nlp.has_pipe("parser")
            or self.nlp.has_pipe("senter")
            or self.nlp.has_pipe("sentencizer")
        ):
            try:
                self.nlp.add_pipe("sentencizer")
            except Exception:  # pragma: no cover — defensive only
                logger.warning("Could not add sentencizer to spaCy pipeline")

        logger.info(
            "NarrativeGraphBuilder initialized (spacy=%s)",
            "yes" if self.nlp is not None else "no/regex-fallback",
        )

    # =====================================================
    # KEYWORD EXTRACTION  (G-S1 / G-T2)
    # =====================================================

    def _sentence_keywords_spacy(
        self,
        sent,
    ) -> List[Tuple[str, int, int]]:
        """Extract ``(keyword, start_char, end_char)`` from a spaCy sentence.

        Uses noun-chunks (preferred — multi-word phrases are real
        narrative units) plus named entities.

        G-S10: the lemmatised-content-token fallback below was
        previously documented as "the blank pipeline path", but its
        trigger (``if not items``) also fires when a real spaCy model
        produces zero noun-chunks *and* zero entities for a sentence
        — short interjections, headlines, social-media fragments. So
        treat it as a **graceful degradation path** that runs in
        production any time both higher-quality extractors come back
        empty, not just when the blank pipeline is loaded.
        """
        items: List[Tuple[str, int, int]] = []
        seen: Set[str] = set()

        # ---- noun chunks (real linguistic units) ----
        try:
            for nc in sent.noun_chunks:
                key = nc.text.lower().strip()
                if (
                    len(key) >= self.min_token_length
                    and key not in seen
                ):
                    items.append((key, nc.start_char, nc.end_char))
                    seen.add(key)
        except (NotImplementedError, ValueError):
            # blank pipelines have no parser → no noun_chunks
            pass

        # ---- named entities ----
        for ent in sent.ents:
            key = ent.text.lower().strip()
            if (
                len(key) >= self.min_token_length
                and key not in seen
            ):
                items.append((key, ent.start_char, ent.end_char))
                seen.add(key)

        # ---- fallback: lemmatised content tokens ----
        # G-S10: fires either when the loaded pipeline is blank (no
        # parser → no noun_chunks, no NER → no ents) *or* when a real
        # pipeline produces nothing for this specific sentence. Both
        # cases are graceful degradation, not error states.
        if not items:
            for tok in sent:
                if not tok.is_alpha:
                    continue
                if getattr(tok, "is_stop", False):
                    continue
                key = (
                    (tok.lemma_ or tok.text).lower().strip()
                )
                if (
                    len(key) >= self.min_token_length
                    and key not in seen
                ):
                    items.append((key, tok.idx, tok.idx + len(tok.text)))
                    seen.add(key)

        return items[: self.max_keywords_per_sentence]

    # =====================================================
    # 🔥 BUILD GRAPH (WEIGHTED + SPAN-AWARE)
    # =====================================================

    def build_graph_with_spans(
        self,
        text: str,
        *,
        doc: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Build the narrative graph **and** return per-keyword char offsets.

        Returns a dict::

            {
                "graph":  Dict[str, Dict[str, float]],
                "spans":  List[
                    {"keyword": str, "start_char": int,
                     "end_char": int, "sentence_index": int}
                ],
                "tokenizer": "spacy" | "regex",
            }

        The ``spans`` list lets the explainer / API layer map a node
        like ``"obama"`` back to a highlightable region of the source
        text — addresses G-T1 / G-T2.
        """

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Invalid text")

        graph: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        spans: List[Dict[str, Any]] = []

        # -------- collect per-sentence keyword lists --------
        sentence_keywords: List[List[str]] = []

        # G-P8: when the caller already parsed the text via spaCy
        # (e.g. ``GraphPipeline._run_with_doc`` shares one ``Doc`` with
        # the entity builder), accept it directly so we don't re-run
        # the parser. Falls back to the previous self-parse path.
        if doc is None and self.nlp is not None:
            doc = self.nlp(text)

        if doc is not None:

            for s_idx, sent in enumerate(doc.sents):
                kws = self._sentence_keywords_spacy(sent)
                sentence_keywords.append([k for k, _, _ in kws])

                for k, s, e in kws:
                    spans.append(
                        {
                            "keyword": k,
                            "start_char": int(s),
                            "end_char": int(e),
                            "sentence_index": s_idx,
                        }
                    )

            tokenizer_used = "spacy"

        else:
            # ---- regex fallback (no spaCy available) ----
            offset = 0
            for s_idx, sentence in enumerate(_split_sentences(text)):
                pos = text.find(sentence, offset)
                if pos < 0:
                    pos = offset
                offset = pos + len(sentence)

                triples = _regex_keyword_spans(
                    sentence,
                    sentence_offset=pos,
                    min_len=self.min_token_length,
                    top_k=self.max_keywords_per_sentence,
                )
                sentence_keywords.append([k for k, _, _ in triples])

                for k, s, e in triples:
                    spans.append(
                        {
                            "keyword": k,
                            "start_char": int(s),
                            "end_char": int(e),
                            "sentence_index": s_idx,
                        }
                    )

            tokenizer_used = "regex"

        # -------- build edges: (1) intra-sentence + (2) succession --------
        prev_keywords: List[str] = []

        for kws in sentence_keywords:

            if not kws:
                # empty sentence — keep prev_keywords so that succession
                # can still bridge across sentences with no salient nodes
                continue

            # ensure node entries exist (defaultdict factory survives the
            # later ``dict(v)`` conversion below)
            for k in kws:
                if k not in graph:
                    graph[k] = defaultdict(float)

            # (1) intra-sentence co-occurrence — real shared-membership edges
            #
            # G-S5: write every edge into a single canonical
            # (sorted-tuple) direction so the downstream
            # ``canonicalize_weighted`` symmetrise step (which takes
            # ``max(w_uv, w_vu)``) preserves the true co-occurrence
            # count instead of returning ``max(uv, vu)`` when the
            # same pair was independently written in both directions
            # by different sentence pairs.
            for i, u in enumerate(kws):
                for v in kws[i + 1:]:
                    if u != v:
                        a, b = (u, v) if u < v else (v, u)
                        graph[a][b] += 1.0

            # (2) temporal succession — preserves the prior chain semantics,
            # but on real linguistic units instead of regex tokens
            if prev_keywords:
                for src in prev_keywords:
                    for tgt in kws:
                        if src != tgt:
                            a, b = (src, tgt) if src < tgt else (tgt, src)
                            graph[a][b] += 1.0

            prev_keywords = kws

        return {
            "graph": {k: dict(v) for k, v in graph.items()},
            "spans": spans,
            "tokenizer": tokenizer_used,
        }

    def build_graph(self, text: str) -> Dict[str, Dict[str, float]]:
        """Backward-compatible entrypoint — returns just the graph dict.

        Existing callers (``GraphPipeline``, ``GraphFeatureExtractor``,
        ``analyze_article``, ``interaction_graph_features``) keep
        working unchanged. New callers that need span alignment should
        call :meth:`build_graph_with_spans`.
        """
        return self.build_graph_with_spans(text)["graph"]

    def build_graph_with_doc(
        self,
        text: str,
        doc: Any,
    ) -> Dict[str, Dict[str, float]]:
        """G-P8: build the narrative graph from a pre-parsed spaCy ``Doc``.

        Used by :class:`GraphPipeline` so the entity and narrative
        builders share a single ``nlp(text)`` pass instead of running
        the parser twice on the same string.
        """
        return self.build_graph_with_spans(text, doc=doc)["graph"]

    # =====================================================
    # FEATURES
    # =====================================================

    def extract_graph_features(
        self,
        graph: Dict[str, Dict[str, float]],
    ) -> NarrativeGraphFeatures:

        if not isinstance(graph, dict):
            raise ValueError("graph must be dict")

        nodes = set(graph.keys())

        # G-S4: the input graph is the post-``canonicalize_weighted``
        # double-entry adjacency, so every undirected edge appears in
        # *both* directions (``graph[u][v]`` and ``graph[v][u]``).
        # The previous implementation counted ordered pairs into
        # ``edges`` and appended each weight twice, which doubled the
        # reported edge count, doubled the weight mass used by the
        # entropy / flow_strength metrics, and broke parity with
        # ``GraphAnalyzer.compute_graph_metrics`` (which divides by 2).
        # Dedupe on a sorted-tuple key so each undirected edge
        # contributes exactly once.
        edge_weights: Dict[Tuple[str, str], float] = {}
        degrees: List[int] = []

        for src, nbrs in graph.items():

            for tgt, w in nbrs.items():
                if src != tgt:
                    key = (src, tgt) if src < tgt else (tgt, src)
                    if key not in edge_weights:
                        edge_weights[key] = float(w)

            degrees.append(len(nbrs))

            nodes.update(nbrs.keys())

        weights = list(edge_weights.values())

        n = len(nodes)
        e = len(edge_weights)

        degrees_arr = np.array(degrees, dtype=float) if degrees else np.array([])

        avg_degree = float(np.mean(degrees_arr)) if degrees_arr.size else 0.0
        density = float(e / (n * (n - 1) + EPS)) if n > 1 else 0.0

        isolated = sum(1 for d in degrees if d == 0)

        components = self._weak_components(graph)

        # =================================================
        # 🔥 NEW METRICS
        # =================================================

        # entropy (distribution of edges)
        if weights:
            w = np.array(weights, dtype=float)
            p = w / (np.sum(w) + EPS)
            entropy = float(-np.sum(p * np.log(p + EPS)))
        else:
            entropy = 0.0

        # centralization
        if degrees_arr.size:
            centralization = float(
                (np.max(degrees_arr) - np.mean(degrees_arr)) / (n - 1 + EPS)
            )
        else:
            centralization = 0.0

        # flow strength (temporal continuity)
        flow_strength = float(np.mean(weights)) if weights else 0.0

        return NarrativeGraphFeatures(
            narrative_graph_nodes=float(n),
            narrative_graph_edges=float(e),
            narrative_graph_avg_degree=avg_degree,
            narrative_graph_density=density,
            narrative_graph_isolated_nodes=float(isolated),
            narrative_graph_components=float(components),
            narrative_graph_entropy=entropy,
            narrative_graph_centralization=centralization,
            narrative_graph_flow_strength=flow_strength,
        )

    # =====================================================
    # COMPONENTS
    # =====================================================

    def _weak_components(self, graph: Dict[str, Dict[str, float]]) -> int:

        undirected: Dict[str, Set[str]] = defaultdict(set)

        for u, nbrs in graph.items():
            for v in nbrs:
                undirected[u].add(v)
                undirected[v].add(u)

        visited: Set[str] = set()
        count = 0

        for start in undirected:

            if start in visited:
                continue

            count += 1

            q = deque([start])
            visited.add(start)

            while q:
                node = q.popleft()
                for nbr in undirected[node]:
                    if nbr not in visited:
                        visited.add(nbr)
                        q.append(nbr)

        return count


# =========================================================
# VECTOR
# =========================================================

def narrative_graph_vector(features: Dict[str, float]) -> np.ndarray:

    keys: Iterable[str] = (
        "narrative_graph_nodes",
        "narrative_graph_edges",
        "narrative_graph_avg_degree",
        "narrative_graph_density",
        "narrative_graph_isolated_nodes",
        "narrative_graph_components",
        "narrative_graph_entropy",
        "narrative_graph_centralization",
        "narrative_graph_flow_strength",
    )

    return np.array(
        [float(features.get(k, 0.0)) for k in keys],
        dtype=np.float32,
    )
