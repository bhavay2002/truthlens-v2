from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Mapping, Set, Tuple, Union

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)
EPS = 1e-12

# G-C5: canonical graph type is **weighted** ``Dict[str, Dict[str, float]]``.
# A plain ``Dict[str, List[str]]`` is still accepted as input (treated as
# weight=1.0 per edge) so existing callers don't break.
WeightedGraph = Dict[str, Dict[str, float]]
GraphLike = Union[Dict[str, Dict[str, float]], Dict[str, List[str]]]


# =========================================================
# CANONICALIZATION  (G-C5 + G-P1)
# =========================================================

def canonicalize_weighted(graph: GraphLike) -> WeightedGraph:
    """Return a normalized, symmetric, weighted view of ``graph``.

    The function is **idempotent**: calling it twice on the same input
    produces the same output (and the second call is a no-op besides
    a single allocation). This is what lets the pipeline canonicalize
    once at the top and pass the result to every downstream consumer
    without worrying about double-symmetrising weights (G-S2 / G-P1).

    Accepted inputs:
      * ``Dict[str, Dict[str, float]]`` — weighted edges (canonical form)
      * ``Dict[str, List[str]]``        — unweighted, treated as weight 1.0

    Output guarantees:
      * keys (and neighbour names) are ``str.strip().lower()``
      * no self-loops
      * graph is symmetric: ``out[u][v] == out[v][u]``
      * isolated nodes from the input are preserved
      * all weights are positive ``float`` values
      * if both ``out[u][v]`` and ``out[v][u]`` already existed in the
        input we take the **max** of the two so re-canonicalising never
        amplifies a value (the builder writes both directions equally,
        so max == either value in practice).
    """

    if not isinstance(graph, dict):
        raise TypeError("graph must be dict")

    # ----- pass 1: directed weights with normalised string keys -----
    directed: Dict[str, Dict[str, float]] = {}

    for node, nbrs in graph.items():

        if not isinstance(node, str):
            continue

        nk = node.strip().lower()
        if not nk:
            continue

        directed.setdefault(nk, {})

        if isinstance(nbrs, dict):
            items = nbrs.items()
        else:
            items = ((n, 1.0) for n in nbrs)

        for nbr, w in items:

            if not isinstance(nbr, str):
                continue

            nbrk = nbr.strip().lower()

            if not nbrk or nbrk == nk:
                continue

            try:
                wv = float(w)
            except (TypeError, ValueError):
                continue

            if wv <= 0.0:
                continue

            # Take max if duplicate keys after normalisation.
            directed[nk][nbrk] = max(directed[nk].get(nbrk, 0.0), wv)

    # ----- pass 2: symmetrise (max of both directions, idempotent) -----
    out: WeightedGraph = {n: {} for n in directed}

    for u, nbrs in directed.items():
        for v, w in nbrs.items():

            sym_w = max(w, directed.get(v, {}).get(u, 0.0))

            out[u][v] = sym_w
            out.setdefault(v, {})
            out[v][u] = sym_w

    return out


# =========================================================
# LEGACY HELPERS  (kept for back-compat with old call sites)
# =========================================================

def normalize_graph(graph: GraphLike) -> Dict[str, List[str]]:
    """Lossy unweighted view — kept for back-compat callers.

    New code should use :func:`canonicalize_weighted`. This wrapper
    discards weights but preserves topology, so it remains safe for
    consumers that don't read weight values.
    """
    canon = canonicalize_weighted(graph)
    return {n: sorted(v.keys()) for n, v in canon.items()}


def to_undirected(graph: GraphLike) -> Dict[str, List[str]]:
    """Symmetric unweighted view — kept for back-compat."""
    return normalize_graph(graph)  # canonicalize_weighted already symmetrises


def unique_edges(graph: GraphLike) -> List[Tuple[str, str]]:
    """List of unique unordered edge pairs."""
    canon = canonicalize_weighted(graph)
    edges: Set[Tuple[str, str]] = set()
    for u, nbrs in canon.items():
        for v in nbrs:
            edges.add(tuple(sorted((u, v))))
    return list(edges)


# =========================================================
# CLUSTERING — sparse matrix triangle count (G-P2)
# =========================================================

def _build_sparse_adjacency(
    graph: WeightedGraph,
    nodes: List[str],
    *,
    binary: bool = True,
) -> sp.csr_matrix:
    """Build a CSR adjacency matrix from a canonical weighted graph.

    ``binary=True`` returns a 0/1 adjacency (used for triangle counting
    and other topology-only metrics); ``binary=False`` preserves edge
    weights (used for the spectral embedding).
    """
    idx = {n: i for i, n in enumerate(nodes)}
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for u, nbrs in graph.items():
        if u not in idx:
            continue
        iu = idx[u]
        for v, w in nbrs.items():
            if v not in idx:
                continue
            rows.append(iu)
            cols.append(idx[v])
            data.append(1.0 if binary else float(w))

    n = len(nodes)
    if not data:
        return sp.csr_matrix((n, n), dtype=np.float64)

    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
    # Defensive symmetrise — canonicalize_weighted should already
    # guarantee this but be robust to malformed direct callers.
    A = A.maximum(A.T)
    return A


def _average_clustering_sparse(graph: WeightedGraph, nodes: List[str]) -> float:
    """Average local clustering coefficient via sparse matrix ops.

    Replaces the previous ``O(N · k^2)`` Python loop. For a node ``u``
    with neighbour set ``N(u)``:

        C(u) = 2 · | { (i, j) in N(u) x N(u) : i < j and (i, j) is an edge } |
                / ( deg(u) · (deg(u) - 1) )

    Globally we use ``triangles_per_node = diag(A^2 ⊙ A) / 2`` which
    counts, for each node, the number of edges among its neighbours.
    """
    n = len(nodes)
    if n < 3:
        return 0.0

    A = _build_sparse_adjacency(graph, nodes, binary=True)

    if A.nnz == 0:
        return 0.0

    A2 = A @ A
    # Element-wise product with A then per-row sum -> # of closed
    # triangles each node participates in (each triangle counted 2x).
    triangles_per_node = np.asarray((A2.multiply(A)).sum(axis=1)).flatten() / 2.0

    degrees = np.asarray(A.sum(axis=1)).flatten()
    triplets = degrees * (degrees - 1) / 2.0

    coef = np.zeros(n, dtype=np.float64)
    nz = triplets > 0
    coef[nz] = triangles_per_node[nz] / triplets[nz]

    return float(np.mean(coef))


# =========================================================
# CORE METRICS
# =========================================================

def compute_graph_metrics(graph: GraphLike) -> Dict[str, float]:

    g = canonicalize_weighted(graph)

    nodes = list(g.keys())
    n = len(nodes)

    # Edge count — symmetric graph means each undirected edge appears
    # twice in the adjacency lists; divide by 2.
    e = sum(len(v) for v in g.values()) // 2

    degrees = np.array([len(g[u]) for u in nodes], dtype=float)
    weighted_degrees = np.array([sum(g[u].values()) for u in nodes], dtype=float)

    # -------------------------
    # BASIC
    # -------------------------
    avg_degree = float(np.mean(degrees)) if n > 0 else 0.0
    max_degree = float(np.max(degrees)) if n > 0 else 0.0
    min_degree = float(np.min(degrees)) if n > 0 else 0.0
    var_degree = float(np.var(degrees)) if n > 0 else 0.0

    density = float((2 * e) / (n * (n - 1) + EPS)) if n > 1 else 0.0

    centralization = (
        float((max_degree - avg_degree) / (n - 1 + EPS)) if n > 1 else 0.0
    )

    # -------------------------
    # CLUSTERING (G-P2: sparse triangle count, no Python loops)
    # -------------------------
    clustering = _average_clustering_sparse(g, nodes)

    # -------------------------
    # CENTRALITY (degree-based)
    # -------------------------
    centrality = degrees / (n - 1 + EPS) if n > 1 else degrees
    centrality_mean = float(np.mean(centrality)) if n > 0 else 0.0
    centrality_var = float(np.var(centrality)) if n > 0 else 0.0

    # -------------------------
    # ENTROPY  (G-C5: now uses *weighted* degree distribution so
    # edge weights actually influence the metric instead of being
    # silently discarded by the consumer.)
    # -------------------------
    if weighted_degrees.sum() > 0:
        p = weighted_degrees / (weighted_degrees.sum() + EPS)
        entropy = float(-np.sum(p * np.log(p + EPS)))
    else:
        entropy = 0.0

    return {
        "graph_nodes": float(n),
        "graph_edges": float(e),
        "graph_avg_degree": avg_degree,
        "graph_max_degree": max_degree,
        "graph_min_degree": min_degree,
        "graph_degree_variance": var_degree,
        "graph_density": density,
        "graph_centralization": centralization,
        "graph_clustering": clustering,
        "graph_centrality_mean": centrality_mean,
        "graph_centrality_variance": centrality_var,
        "graph_entropy": entropy,
    }


# =========================================================
# DATACLASS
# =========================================================

@dataclass
class GraphMetrics:
    graph_nodes: float
    graph_edges: float
    graph_avg_degree: float
    graph_max_degree: float
    graph_min_degree: float
    graph_degree_variance: float
    graph_density: float
    graph_centralization: float
    graph_clustering: float
    graph_centrality_mean: float
    graph_centrality_variance: float
    graph_entropy: float

    def to_dict(self) -> Dict[str, float]:
        return self.__dict__


# =========================================================
# ANALYZER
# =========================================================

class GraphAnalyzer:

    def __init__(self):
        logger.info("GraphAnalyzer initialized")

    def analyze(self, graph: GraphLike) -> GraphMetrics:

        if not isinstance(graph, dict):
            raise TypeError("graph must be dictionary")

        metrics = compute_graph_metrics(graph)

        return GraphMetrics(**metrics)


# =========================================================
# VECTOR
# =========================================================

def graph_to_vector(features: Dict[str, float]) -> np.ndarray:

    keys = [
        "graph_nodes",
        "graph_edges",
        "graph_avg_degree",
        "graph_max_degree",
        "graph_min_degree",
        "graph_degree_variance",
        "graph_density",
        "graph_centralization",
        "graph_clustering",
        "graph_centrality_mean",
        "graph_centrality_variance",
        "graph_entropy",
    ]

    return np.array([features.get(k, 0.0) for k in keys], dtype=np.float32)

# Alias maintained for backward compatibility with src.graph.graph_features.
ordered_graph_metrics_vector = graph_to_vector
