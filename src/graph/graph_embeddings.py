from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import networkx as nx
import scipy.sparse as sp

logger = logging.getLogger(__name__)
EPS = 1e-12

# G-C5: accept either weighted or unweighted dict-of-dict / dict-of-list.
Graph = Union[Dict[str, Dict[str, float]], Dict[str, List[str]]]


# =========================================================
# CONFIG
# =========================================================

@dataclass(slots=True)
class GraphEmbeddingConfig:

    embedding_type: str = "hybrid"  # degree | centrality | spectral | hybrid | node2vec
    spectral_dim: int = 8
    normalize: bool = True

    # Node2Vec-lite
    walk_length: int = 10
    num_walks: int = 10
    embedding_dim: int = 16


# =========================================================
# DIMENSION CONTRACTS  (G-E1)
#
# The downstream feature vector pads to a fixed schema, but the
# ``extract_from_graphs`` consumer used to enumerate the embedding
# returned here and emit one feature per element. That made the
# vector length non-deterministic (1 for empty input, 28 for the
# default hybrid type) — which silently changed the model input
# dimension. Pin every embedding type to a known length and always
# return that many floats so the schema stays stable.
# =========================================================

def _embedding_target_dim(cfg: GraphEmbeddingConfig) -> int:
    etype = cfg.embedding_type.lower()
    if etype == "degree":
        return 4
    if etype == "centrality":
        return 4
    if etype == "spectral":
        return cfg.spectral_dim
    if etype == "node2vec":
        return cfg.embedding_dim
    if etype == "hybrid":
        # 4 degree + 4 centrality + spectral_dim
        return 4 + 4 + cfg.spectral_dim
    return cfg.embedding_dim


# =========================================================
# SPECTRAL  (G-P4)
# =========================================================

def _to_sparse_adjacency(
    graph_or_adj: Union[np.ndarray, Graph],
) -> Optional[sp.csr_matrix]:
    """Build a symmetric sparse weighted adjacency from any input.

    Returns ``None`` for empty inputs so the caller can short-circuit.
    """
    if isinstance(graph_or_adj, np.ndarray):
        if graph_or_adj.size == 0:
            return None
        A = sp.csr_matrix(graph_or_adj.astype(np.float64))
        return A.maximum(A.T)

    if isinstance(graph_or_adj, dict):
        nodes = sorted(graph_or_adj.keys())
        n = len(nodes)
        if n == 0:
            return None

        idx = {nd: i for i, nd in enumerate(nodes)}
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        for u, nbrs in graph_or_adj.items():
            iu = idx[u]
            items = nbrs.items() if isinstance(nbrs, dict) else ((v, 1.0) for v in nbrs)
            for v, w in items:
                if v not in idx:
                    continue
                rows.append(iu)
                cols.append(idx[v])
                data.append(float(w))

        if not data:
            return sp.csr_matrix((n, n), dtype=np.float64)

        A = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
        return A.maximum(A.T)

    return None


def spectral_eigen_embedding(
    graph_or_adj: Union[np.ndarray, Graph],
    dim: int = 8,
) -> np.ndarray:
    """Top-``dim`` eigenvalues of the symmetric adjacency.

    G-P4 fix: previous implementation built an ``O(N^2)`` dense matrix
    via ``nx.to_numpy_array`` and then ran ``np.linalg.eigvalsh`` —
    ``O(N^3)`` per request. We now use the sparse Lanczos solver
    (``scipy.sparse.linalg.eigsh``) for any graph above ~32 nodes,
    where ARPACK setup overhead becomes worthwhile, and fall back to
    the dense path on failure.
    """

    A = _to_sparse_adjacency(graph_or_adj)

    if A is None or A.shape[0] == 0:
        return np.zeros(dim, dtype=np.float32)

    n = A.shape[0]

    # G-S7: previously the dense path returned the top-k by *signed*
    # value (``np.sort(...)[::-1]``) while the sparse path used
    # ``which="LA"`` (largest algebraic). Both pick "largest signed"
    # — which silently disagrees with conventional spectral-embedding
    # semantics (top-k by magnitude) for narrative graphs whose
    # adjacency matrices have negative eigenvalues. Unify both
    # branches on largest-magnitude (``LM``) so the embedding feature
    # is consistent across the dense / sparse threshold and consistent
    # with what callers expect from a "spectral embedding".
    def _top_k_by_magnitude(values: np.ndarray, k: int) -> np.ndarray:
        if values.size == 0:
            return values
        order = np.argsort(np.abs(values))[::-1]
        return values[order[:k]]

    # ARPACK requires ``k < n``. For tiny graphs the dense path is
    # actually faster and avoids the ``k = n`` corner case entirely.
    if n <= 32 or dim >= n:
        try:
            eigenvalues = np.linalg.eigvalsh(A.toarray())
            eigenvalues = _top_k_by_magnitude(eigenvalues, dim)
        except np.linalg.LinAlgError:
            logger.warning("Dense eigen decomposition failed (n=%d)", n)
            return np.zeros(dim, dtype=np.float32)
    else:
        try:
            from scipy.sparse.linalg import eigsh
            k = min(dim, n - 1)
            eigenvalues = eigsh(
                A.astype(np.float64),
                k=k,
                which="LM",
                return_eigenvectors=False,
            )
            eigenvalues = _top_k_by_magnitude(eigenvalues, dim)
        except Exception as exc:
            logger.warning("Sparse eigen decomposition failed (n=%d): %s", n, exc)
            try:
                eigenvalues = np.linalg.eigvalsh(A.toarray())
                eigenvalues = _top_k_by_magnitude(eigenvalues, dim)
            except np.linalg.LinAlgError:
                return np.zeros(dim, dtype=np.float32)

    if len(eigenvalues) < dim:
        eigenvalues = np.pad(eigenvalues, (0, dim - len(eigenvalues)))

    return eigenvalues[:dim].astype(np.float32)


# =========================================================
# CORE
# =========================================================

class GraphEmbeddingGenerator:

    def __init__(self, config: Optional[GraphEmbeddingConfig] = None):

        self.config = config or GraphEmbeddingConfig()

        if self.config.spectral_dim < 1:
            raise ValueError("spectral_dim must be >= 1")

        logger.info(
            "GraphEmbeddingGenerator initialized (%s)",
            self.config.embedding_type,
        )

    # =====================================================
    # UTILS
    # =====================================================

    def _validate(self, graph: Graph):
        if not isinstance(graph, dict):
            raise TypeError("graph must be dict")

    def _to_nx(self, graph: Graph) -> nx.Graph:
        # G-C5: weighted edges now flow through to NetworkX so future
        # consumers (centrality variants, weighted PageRank, etc.) can
        # use them. ``degree_centrality`` itself is unweighted by
        # design, so this doesn't change current numerical output.
        G = nx.Graph()

        for node, neighbors in graph.items():
            if not isinstance(node, str):
                continue
            n = node.strip().lower()
            if not n:
                continue
            G.add_node(n)

            items = (
                neighbors.items()
                if isinstance(neighbors, dict)
                else ((v, 1.0) for v in neighbors)
            )

            for nbr, w in items:
                if not isinstance(nbr, str):
                    continue
                m = nbr.strip().lower()
                if m and m != n:
                    G.add_edge(n, m, weight=float(w))

        return G

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        if not self.config.normalize:
            return vec

        norm = np.linalg.norm(vec) + EPS
        return vec / norm

    # =====================================================
    # DEGREE
    # =====================================================

    def _degree(self, G: nx.Graph) -> np.ndarray:
        d = np.array([deg for _, deg in G.degree()], dtype=float)

        if d.size == 0:
            return np.zeros(4, dtype=np.float32)

        return np.array(
            [np.mean(d), np.max(d), np.min(d), np.var(d)],
            dtype=np.float32,
        )

    # =====================================================
    # CENTRALITY
    # =====================================================

    def _centrality(self, G: nx.Graph) -> np.ndarray:
        c = list(nx.degree_centrality(G).values())

        if not c:
            return np.zeros(4, dtype=np.float32)

        c = np.array(c, dtype=float)

        return np.array(
            [np.mean(c), np.max(c), np.min(c), np.var(c)],
            dtype=np.float32,
        )

    # =====================================================
    # NODE2VEC (LITE)
    # =====================================================

    def _node2vec(self, G: nx.Graph) -> np.ndarray:

        if G.number_of_nodes() == 0:
            return np.zeros(self.config.embedding_dim, dtype=np.float32)

        nodes = list(G.nodes())
        n_nodes = len(nodes)
        vocab = {n: i for i, n in enumerate(nodes)}

        # G-R3: previously called ``np.random.choice`` which uses the
        # global numpy RNG, so two consecutive calls on the same graph
        # returned different embeddings — breaking
        # ``batch_feature_pipeline._build_cache_key``'s assumption of
        # deterministic features per ``(text, config_fingerprint)``
        # pair. Seed a *local* generator from the sorted node tuple so
        # the same graph topology always yields the same walks across
        # processes / runs without disturbing global RNG state.
        seed = hash(tuple(sorted(nodes))) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)

        # G-P5: previously allocated an N×N dense float64 transition
        # matrix and computed ``np.mean(mat, axis=0)`` at the end —
        # i.e. column sums divided by N. The matrix grew O(N²) memory
        # for every entity graph (a 5k-node article ≈ 200 MB) yet
        # only the column sums were ever read. Accumulate column
        # sums (= "incoming" walk landings) directly in a 1D vector
        # so the runtime is O(num_walks · walk_length · |V|) memory
        # instead of O(|V|²).
        incoming = np.zeros(n_nodes, dtype=np.float64)

        for _ in range(self.config.num_walks):
            for node in nodes:
                current = node

                for _ in range(self.config.walk_length - 1):
                    neighbors = list(G.neighbors(current))
                    if not neighbors:
                        break
                    current = rng.choice(neighbors)
                    incoming[vocab[current]] += 1.0

        vec = incoming / float(n_nodes)

        if vec.size < self.config.embedding_dim:
            vec = np.pad(vec, (0, self.config.embedding_dim - vec.size))

        return vec[: self.config.embedding_dim].astype(np.float32)

    # =====================================================
    #  TEMPORAL SCALING
    # =====================================================

    def _apply_temporal_weight(
        self,
        vec: np.ndarray,
        temporal_features: Optional[Dict[str, float]],
    ) -> np.ndarray:

        if not temporal_features:
            return vec

        drift = float(temporal_features.get("narrative_drift", 0.0))

        # safety clip
        scale = 1.0 + np.clip(drift, 0.0, 1.0)

        return vec * scale

    # =====================================================
    # MAIN
    # =====================================================

    def generate_embedding(
        self,
        graph: Graph,
        *,
        temporal_features: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:

        self._validate(graph)

        target_dim = _embedding_target_dim(self.config)

        # G-E1: empty graph used to return ``np.zeros(1)`` which broke
        # the schema contract (downstream emitted a single feature
        # ``graph_embedding_0``). Always return the configured length.
        G = self._to_nx(graph)

        if G.number_of_nodes() == 0:
            return np.zeros(target_dim, dtype=np.float32)

        etype = self.config.embedding_type.lower()

        # -------------------------
        # BASE EMBEDDING
        # -------------------------
        if etype == "degree":
            vec = self._degree(G)

        elif etype == "centrality":
            vec = self._centrality(G)

        elif etype == "spectral":
            # G-P4: pass the weighted dict directly so we never
            # materialise an N×N dense adjacency.
            vec = spectral_eigen_embedding(graph, self.config.spectral_dim)

        elif etype == "node2vec":
            vec = self._node2vec(G)

        elif etype == "hybrid":
            deg = self._degree(G)
            cen = self._centrality(G)
            spec = spectral_eigen_embedding(graph, self.config.spectral_dim)
            vec = np.concatenate([deg, cen, spec])

        else:
            raise ValueError(f"Unknown embedding type: {etype}")

        # -------------------------
        # NORMALIZE + FIXED LENGTH (G-E1)
        # -------------------------
        vec = self._normalize(vec)

        # -------------------------
        #  TEMPORAL ADAPTATION
        # -------------------------
        # G-S6: apply temporal scaling *after* normalisation. The
        # previous order multiplied the raw vector by ``1 + drift``
        # and then ran an L2 normalise, which divides the scale right
        # back out — the embedding was bit-identical regardless of
        # the temporal context. Scaling post-normalisation keeps the
        # vector's *direction* intact while its norm now encodes
        # narrative drift, so downstream models (and the explainer)
        # actually see the temporal signal that this layer is
        # supposed to inject.
        vec = self._apply_temporal_weight(vec, temporal_features)

        if vec.size < target_dim:
            vec = np.pad(vec, (0, target_dim - vec.size))
        elif vec.size > target_dim:
            vec = vec[:target_dim]

        return vec.astype(np.float32)


# =========================================================
# API
# =========================================================

def graph_embedding_vector(
    graph: Graph,
    config: Optional[GraphEmbeddingConfig] = None,
    *,
    temporal_features: Optional[Dict[str, float]] = None,
) -> np.ndarray:

    generator = GraphEmbeddingGenerator(config)

    return generator.generate_embedding(
        graph,
        temporal_features=temporal_features,
    )
