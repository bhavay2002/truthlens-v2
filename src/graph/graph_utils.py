from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)
EPS = 1e-12

Graph = Dict[str, List[str]]
AdjacencySet = Dict[str, Set[str]]
WeightedGraph = Dict[str, Dict[str, float]]
EdgePair = Tuple[str, str]


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_adjacency(graph: Graph) -> AdjacencySet:

    if not isinstance(graph, dict):
        raise ValueError("graph must be dict")

    adj: AdjacencySet = {}

    for node, neighbors in graph.items():

        node_key = str(node).strip().lower()

        adj[node_key] = {
            str(n).strip().lower()
            for n in neighbors
            if isinstance(n, str)
            and n.strip()
            and str(n).strip().lower() != node_key
        }

    return adj


# =========================================================
# WEIGHTED SUPPORT (🔥 NEW)
# =========================================================

def normalize_weighted_graph(graph: WeightedGraph) -> WeightedGraph:

    out: WeightedGraph = {}

    for node, nbrs in graph.items():
        nk = node.strip().lower()
        out[nk] = {}

        for nbr, w in nbrs.items():
            if isinstance(nbr, str):
                nk2 = nbr.strip().lower()
                if nk2 and nk2 != nk:
                    out[nk][nk2] = float(max(0.0, w))

    return out


# =========================================================
# UNDIRECTED
# =========================================================

def to_undirected_graph(adjacency: AdjacencySet) -> AdjacencySet:

    undirected: AdjacencySet = {
        node: set(neighbors) for node, neighbors in adjacency.items()
    }

    for node, neighbors in list(undirected.items()):
        for nbr in neighbors:
            undirected.setdefault(nbr, set()).add(node)

    return undirected


def to_undirected_weighted(graph: WeightedGraph) -> WeightedGraph:

    undirected: WeightedGraph = {}

    for u, nbrs in graph.items():
        undirected.setdefault(u, {})

        for v, w in nbrs.items():

            undirected.setdefault(v, {})

            undirected[u][v] = undirected[u].get(v, 0.0) + w
            undirected[v][u] = undirected[v].get(u, 0.0) + w

    return undirected


# =========================================================
# CLEANUP
# =========================================================

def remove_self_loops(adjacency: AdjacencySet) -> AdjacencySet:

    return {
        node: {nbr for nbr in nbrs if nbr != node}
        for node, nbrs in adjacency.items()
    }


# =========================================================
# EDGE UTILITIES
# =========================================================

def unique_edge_pairs(adjacency: AdjacencySet) -> Set[EdgePair]:

    edges: Set[EdgePair] = set()

    for u, nbrs in adjacency.items():
        for v in nbrs:
            if u != v:
                edges.add(tuple(sorted((u, v))))

    return edges


def edge_count_undirected(adjacency: AdjacencySet) -> int:
    return len(unique_edge_pairs(adjacency))


def edge_count_directed(adjacency: AdjacencySet) -> int:
    return sum(len(nbrs) for nbrs in adjacency.values())


# =========================================================
# NODE UTILITIES
# =========================================================

def node_set(adjacency: AdjacencySet) -> Set[str]:

    nodes = set(adjacency.keys())

    for nbrs in adjacency.values():
        nodes.update(nbrs)

    return nodes


# =========================================================
# DEGREE
# =========================================================

def degree_distribution(adjacency: AdjacencySet) -> Dict[int, int]:

    degrees = [len(nbrs) for nbrs in adjacency.values()]
    return dict(Counter(degrees))


def degree_vector(adjacency: AdjacencySet) -> np.ndarray:

    return np.array([len(nbrs) for nbrs in adjacency.values()], dtype=float)


# =========================================================
# ADVANCED METRICS (🔥 NEW)
# =========================================================

def graph_density(adjacency: AdjacencySet) -> float:

    n = len(node_set(adjacency))
    e = edge_count_undirected(adjacency)

    if n < 2:
        return 0.0

    return float((2 * e) / (n * (n - 1) + EPS))


def graph_entropy(adjacency: AdjacencySet) -> float:

    deg = degree_vector(adjacency)

    if deg.size == 0 or np.sum(deg) == 0:
        return 0.0

    p = deg / (np.sum(deg) + EPS)

    return float(-np.sum(p * np.log(p + EPS)))


def graph_centralization(adjacency: AdjacencySet) -> float:

    deg = degree_vector(adjacency)

    if deg.size == 0:
        return 0.0

    max_d = np.max(deg)
    mean_d = np.mean(deg)

    n = len(deg)

    if n < 2:
        return 0.0

    return float((max_d - mean_d) / (n - 1 + EPS))


# =========================================================
# SUMMARY
# =========================================================

def graph_summary(adjacency: AdjacencySet) -> Dict[str, float]:

    nodes = node_set(adjacency)

    return {
        "nodes": float(len(nodes)),
        "edges": float(edge_count_undirected(adjacency)),
        "density": graph_density(adjacency),
        "entropy": graph_entropy(adjacency),
        "centralization": graph_centralization(adjacency),
    }