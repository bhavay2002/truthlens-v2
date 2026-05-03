from __future__ import annotations

import logging
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from spacy.tokens import Doc

from src.features.base.spacy_loader import get_shared_nlp

logger = logging.getLogger(__name__)
EPS = 1e-12


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_graph(graph: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    normalized: Dict[str, Dict[str, float]] = {}

    for node, neighbors in graph.items():
        nk = node.strip().lower()
        normalized[nk] = {}

        for nbr, w in neighbors.items():
            nbrk = nbr.strip().lower()
            if nbrk and nbrk != nk:
                normalized[nk][nbrk] = float(w)

    return normalized


def to_undirected(graph: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    undirected: Dict[str, Dict[str, float]] = defaultdict(dict)

    for node, neighbors in graph.items():
        for nbr, w in neighbors.items():
            undirected[node][nbr] = undirected[node].get(nbr, 0.0) + w
            undirected[nbr][node] = undirected[nbr].get(node, 0.0) + w

    return dict(undirected)


def unique_edges(graph: Dict[str, Dict[str, float]]) -> List[Tuple[str, str]]:
    seen = set()
    edges = []

    for a, neighbors in graph.items():
        for b in neighbors:
            edge = tuple(sorted((a, b)))
            if edge not in seen:
                seen.add(edge)
                edges.append(edge)

    return edges


# =========================================================
# FEATURES
# =========================================================

@dataclass
class EntityGraphFeatures:
    nodes: float
    edges: float
    avg_degree: float
    density: float
    dominant_degree: float
    degree_variance: float
    clustering_coeff: float
    centrality_mean: float

    def to_dict(self):
        return self.__dict__


# =========================================================
# BUILDER
# =========================================================

class EntityGraphBuilder:
    """
    Audit fix §1.7 — this builder used to maintain a private
    ``_NLP_CACHE`` and call :func:`spacy.load` directly. That meant
    every process held two copies of ``en_core_web_sm`` (one in this
    cache, one in :mod:`src.analysis.spacy_loader`) and the parser
    warmed twice on first call. The shared loader resolves both.
    """

    def __init__(self, model: str = "en_core_web_sm"):
        nlp = get_shared_nlp(model)
        if nlp is None:
            # ``get_shared_nlp`` already logged a warning; fall back to
            # an empty pipeline so build_graph still produces an empty
            # graph instead of raising.
            import spacy as _spacy  # local import keeps the module hot path lean
            logger.warning("Fallback to blank spaCy model (model=%s)", model)
            nlp = _spacy.blank("en")

        # ``doc.sents`` requires either a parser, a senter, or a
        # sentencizer. The shared loader returns the blank pipeline
        # when ``en_core_web_sm`` is unavailable, and that blank model
        # has none of the three — so iterating ``doc.sents`` raised
        # ``E030``. Add a cheap rule-based sentencizer if the active
        # pipeline doesn't provide sentence boundaries already.
        if not nlp.has_pipe("parser") \
                and not nlp.has_pipe("senter") \
                and not nlp.has_pipe("sentencizer"):
            try:
                nlp.add_pipe("sentencizer")
            except Exception:  # pragma: no cover — defensive only
                logger.warning("Could not add sentencizer to spaCy pipeline")
        self.nlp = nlp

    # =====================================================
    # GRAPH BUILD (🔥 WEIGHTED + SPAN-AWARE)
    # =====================================================

    # G-D1: factored helper so the pipeline's batched ``run_batch``
    # path (which has already parsed the document via ``nlp.pipe``) can
    # build the entity graph from a pre-parsed ``Doc`` without paying
    # the ``self.nlp(text)`` cost a second time. Keeping the per-doc
    # graph-building logic in one place avoids the prior drift between
    # ``EntityGraphBuilder.build_graph_with_spans`` and the inline
    # ``GraphPipeline._entity_graph_from_doc`` re-implementation.
    def build_graph_with_doc(
        self,
        doc: Doc,
    ) -> Dict[str, Any]:
        """Build the weighted entity graph from a pre-parsed spaCy ``Doc``.

        Returns the same shape as :meth:`build_graph_with_spans` —
        ``{"graph": Dict[str, Dict[str, float]], "spans": List[...]}``
        — so callers can swap one for the other without a translation
        layer.
        """

        graph: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        spans: List[Dict[str, Any]] = []

        for s_idx, sent in enumerate(doc.sents):

            seen: Set[str] = set()
            ents: List[str] = []

            for ent in sent.ents:
                key = ent.text.lower().strip()
                if not key:
                    continue

                spans.append(
                    {
                        "entity": key,
                        "raw_text": ent.text,
                        "start_char": int(ent.start_char),
                        "end_char": int(ent.end_char),
                        "sentence_index": s_idx,
                        "label": getattr(ent, "label_", "") or "",
                    }
                )

                if key not in seen:
                    ents.append(key)
                    seen.add(key)

            for i, a in enumerate(ents):
                for b in ents[i + 1:]:
                    # Single-direction co-occurrence; symmetrised by
                    # ``canonicalize_weighted`` below.
                    graph[a][b] += 1.0

        from src.graph.graph_analysis import canonicalize_weighted

        return {
            "graph": canonicalize_weighted(graph),
            "spans": spans,
        }

    def build_graph_with_spans(self, text: str) -> Dict[str, Any]:
        """Build the entity graph **and** return per-mention char spans.

        G-T1 fix: previously the builder discarded ``ent.start_char`` /
        ``ent.end_char`` and only kept ``ent.text.lower().strip()`` as
        the node id, so a downstream explainer holding
        ``node_importance: {"obama": 0.4}`` had no way to map ``"obama"``
        back to a highlightable region of the source text. We now carry
        the alignment alongside the graph.

        Returns::

            {
                "graph":  Dict[str, Dict[str, float]],
                "spans":  List[
                    {"entity": str,             # node id (lower-cased)
                     "raw_text": str,           # surface form as it appears
                     "start_char": int,
                     "end_char": int,
                     "sentence_index": int,
                     "label": str},             # spaCy NER label (e.g. PERSON)
                ],
            }

        The graph itself is the canonical weighted form so existing
        consumers keep working. ``spans`` is opt-in — only the new
        ``GraphPipeline`` surface and explainer read it.

        Note on G-S2: edges are written **once** here
        (``graph[a][b] += 1.0``); symmetrisation is delegated to
        ``canonicalize_weighted``, which is idempotent (it takes the
        ``max`` of the two directions). This is what prevents the old
        4×-amplification bug — see the comment on ``canonicalize_weighted``.
        """

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Invalid text")

        # G-D1: delegate to the doc-only variant so the per-doc graph
        # building logic lives in exactly one place.
        return self.build_graph_with_doc(self.nlp(text))

    def build_graph(self, text: str) -> Dict[str, Dict[str, float]]:
        """Backward-compatible entrypoint — returns just the graph dict.

        Existing callers stay unchanged. New callers that need span
        alignment should call :meth:`build_graph_with_spans`.
        """
        return self.build_graph_with_spans(text)["graph"]

    # =====================================================
    # FEATURES
    # =====================================================

    def extract_features(self, graph: Dict[str, Dict[str, float]]) -> EntityGraphFeatures:

        # G-C5 / G-P1: route through the canonical, idempotent
        # representation so weights survive and so we don't have to
        # repeat the symmetrise / normalise pass that the pipeline has
        # already done at the top.
        from src.graph.graph_analysis import (
            canonicalize_weighted,
            _average_clustering_sparse,
        )

        g = canonicalize_weighted(graph)

        nodes = list(g.keys())
        n = len(nodes)

        e = sum(len(v) for v in g.values()) // 2

        degree_vals = [len(g[u]) for u in nodes]

        avg_degree = float(np.mean(degree_vals)) if degree_vals else 0.0
        dominant = max(degree_vals, default=0)

        density = (2 * e) / (n * (n - 1) + EPS) if n > 1 else 0.0
        variance = float(np.var(degree_vals)) if degree_vals else 0.0

        # G-P2: sparse triangle count via A^2 ⊙ A — replaces the
        # ``O(N · k^2)`` Python double-loop above (measured ~0.3s on a
        # 300-node graph; sparse variant is ~10ms).
        clustering = _average_clustering_sparse(g, nodes)

        # =================================================
        # CENTRALITY (degree proxy)
        # =================================================
        centrality = [deg / (n - 1 + EPS) for deg in degree_vals] if n > 1 else degree_vals
        centrality_mean = float(np.mean(centrality)) if centrality else 0.0

        return EntityGraphFeatures(
            nodes=float(n),
            edges=float(e),
            avg_degree=avg_degree,
            density=float(density),
            dominant_degree=float(dominant),
            degree_variance=variance,
            clustering_coeff=clustering,
            centrality_mean=centrality_mean,
        )

    # G-C1: ``GraphFeatureExtractor.extract_from_graphs`` calls
    # ``extract_graph_features`` (the name used by NarrativeGraphBuilder)
    # — used to raise AttributeError on the very first request. Alias
    # keeps the two builders API-compatible.
    def extract_graph_features(
        self,
        graph: Dict[str, Dict[str, float]],
    ) -> EntityGraphFeatures:
        return self.extract_features(graph)


# =========================================================
# VECTOR
# =========================================================

def graph_to_vector(features: Dict[str, float]) -> np.ndarray:

    keys = [
        "nodes",
        "edges",
        "avg_degree",
        "density",
        "dominant_degree",
        "degree_variance",
        "clustering_coeff",
        "centrality_mean",
    ]

    return np.array([features.get(k, 0.0) for k in keys], dtype=np.float32)

# Alias maintained for backward compatibility with src.graph.graph_features.
ordered_entity_graph_vector = graph_to_vector

