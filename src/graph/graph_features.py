from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.graph.entity_graph import EntityGraphBuilder, ordered_entity_graph_vector
from src.graph.graph_analysis import (
    GraphAnalyzer,
    canonicalize_weighted,
    ordered_graph_metrics_vector,
)
from src.graph.narrative_graph_builder import (
    NarrativeGraphBuilder,
    narrative_graph_vector,
)
from src.graph.graph_embeddings import graph_embedding_vector, GraphEmbeddingConfig

logger = logging.getLogger(__name__)
EPS = 1e-12


# =========================================================
# DEVICE TRANSFER HELPER  (G-G1)
# =========================================================
#
# The graph layer is intentionally CPU-only (good — spaCy parsing is
# the bottleneck, not the linear algebra) but the resulting feature
# vectors used to be returned as plain ``np.float32`` arrays. The
# downstream consumer (``HybridTruthLensModel.forward``) then did one
# host→device copy *per sample* inside the forward pass, which on
# batched inference is N independent CUDA syncs.
#
# ``to_pinned_tensor`` lets the batch-inference / feature-preparer
# layer:
#   1. accumulate per-sample numpy vectors on the CPU,
#   2. ``np.stack`` them once,
#   3. wrap the stacked array in a pinned-memory tensor here,
#   4. do a single ``.to(device, non_blocking=True)`` at the model
#      boundary.
#
# This collapses N syncs into 1 and overlaps the copy with kernel
# launches when the model is GPU-resident. Pinning is a no-op on
# CPU-only deployments, so this is safe to call unconditionally.
# ``torch`` import is deferred so ``src.graph`` keeps loading on
# torch-less environments (e.g. data-only utility scripts).


def to_pinned_tensor(vec):
    """Wrap a numpy array (or list of arrays) in a pinned CPU tensor.

    Accepts a single ``np.ndarray`` or a list/tuple of equally shaped
    arrays — in the second case the arrays are stacked along axis 0
    so the caller gets a single ``(N, D)`` tensor ready for one
    ``.to(device, non_blocking=True)`` copy at the model edge.

    Returns the tensor unchanged on platforms without CUDA support
    (``pin_memory`` raises on those — we swallow it and return a
    plain CPU tensor instead).
    """
    import torch  # local import keeps ``src.graph`` torch-optional
    import numpy as _np

    if isinstance(vec, (list, tuple)):
        if not vec:
            return torch.empty(0, dtype=torch.float32)
        arr = _np.stack([_np.asarray(v, dtype=_np.float32) for v in vec], axis=0)
    else:
        arr = _np.asarray(vec, dtype=_np.float32)

    t = torch.from_numpy(arr)

    # ``pin_memory`` requires CUDA *or* a CPU-pinned allocator; on
    # pure-CPU builds it raises ``RuntimeError``. Treat that as a
    # no-op so callers can use this helper unconditionally.
    if torch.cuda.is_available():
        try:
            t = t.pin_memory()
        except RuntimeError:
            pass

    return t


# =========================================================
# CONFIG
# =========================================================

@dataclass(slots=True)
class GraphFeatureExtractorConfig:
    enable_entity_graph: bool = True
    enable_narrative_graph: bool = True
    enable_embeddings: bool = True
    embedding_config: Optional[GraphEmbeddingConfig] = None
    normalize_features: bool = True


# =========================================================
# UTIL
# =========================================================

def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec) + EPS
    return vec / norm


def merge_feature_blocks_strict(*blocks: Dict[str, float]) -> Dict[str, float]:
    merged: Dict[str, float] = {}

    for block in blocks:
        for k, v in block.items():
            if k in merged:
                raise ValueError(f"Duplicate feature key: {k}")
            merged[k] = float(v)

    return merged


# =========================================================
# MAIN
# =========================================================

class GraphFeatureExtractor:

    def __init__(self, config: Optional[GraphFeatureExtractorConfig] = None):

        self.config = config or GraphFeatureExtractorConfig()

        self.entity_builder = (
            EntityGraphBuilder() if self.config.enable_entity_graph else None
        )

        self.narrative_builder = (
            NarrativeGraphBuilder() if self.config.enable_narrative_graph else None
        )

        self.analyzer = GraphAnalyzer()

        logger.info("GraphFeatureExtractor initialized")

    # =====================================================
    # FULL PIPELINE
    # =====================================================

    def extract_features(self, text: str) -> Dict[str, float]:

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Invalid text")

        entity_graph = None
        narrative_graph = None

        if self.entity_builder:
            entity_graph = self.entity_builder.build_graph(text)

        if self.narrative_builder:
            narrative_graph = self.narrative_builder.build_graph(text)
            # G-P6: ensure narrative graph is canonical so downstream
            # ``extract_graph_features`` and analyzer see a symmetric,
            # double-entry adjacency (parity with the pipeline path
            # which canonicalizes once in ``_run_with_doc``).
            if narrative_graph:
                narrative_graph = canonicalize_weighted(narrative_graph)

        return self.extract_from_graphs(entity_graph, narrative_graph)

    # =====================================================
    # CORE LOGIC
    # =====================================================

    def extract_from_graphs(
        self,
        entity_graph: Optional[Dict[str, Dict[str, float]]] = None,
        narrative_graph: Optional[Dict[str, Dict[str, float]]] = None,
        *,
        entity_metrics: Optional[Dict[str, float]] = None,
        narrative_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Compose the graph-level feature dict.

        G-R2: ``entity_metrics`` / ``narrative_metrics`` are optional
        pre-computed dicts. When provided, the internal
        ``GraphAnalyzer.analyze`` call is skipped — the pipeline already
        runs it once at the top, so this avoids a second pass over the
        same graph (Python loop, ~O(N+E) per call). When *not* provided
        the legacy behaviour is preserved so direct callers
        (``extract_features(text)`` / tests) keep working.
        """

        blocks: List[Dict[str, float]] = []

        # -------------------------
        # ENTITY GRAPH
        # -------------------------
        if entity_graph and self.entity_builder:

            entity_features = (
                self.entity_builder.extract_graph_features(entity_graph).to_dict()
            )

            metrics = (
                entity_metrics
                if entity_metrics is not None
                else self.analyzer.analyze(entity_graph).to_dict()
            )

            blocks.append(entity_features)
            blocks.append(metrics)

            # 🔥 embeddings
            # G-P7: emit the entire embedding block as a single dict
            # rather than one ``{key: val}`` dict per scalar — the old
            # loop produced ``embedding_dim`` python dicts plus an
            # equal number of ``merge_feature_blocks_strict`` passes.
            if self.config.enable_embeddings:
                emb = graph_embedding_vector(
                    entity_graph,
                    self.config.embedding_config,
                )
                blocks.append(
                    {
                        f"graph_embedding_{i}": float(val)
                        for i, val in enumerate(emb)
                    }
                )

        # -------------------------
        # NARRATIVE GRAPH
        # -------------------------
        if narrative_graph and self.narrative_builder:

            narrative_features = (
                self.narrative_builder.extract_graph_features(narrative_graph).to_dict()
            )

            blocks.append(narrative_features)

            # G-R2 / G-K2: surface narrative metrics too when
            # supplied. They share generic ``graph_*`` keys with the
            # entity-metrics block (both come from
            # ``GraphAnalyzer.analyze``), which previously caused
            # ``merge_feature_blocks_strict`` to raise on every dual-
            # graph call. Re-prefix with ``narrative_metric_`` to
            # disambiguate while keeping the entity metrics block at
            # its canonical ``graph_*`` keys (preserved so
            # ``ordered_graph_metrics_vector`` keeps working).
            if narrative_metrics:
                blocks.append(
                    {
                        f"narrative_metric_{k}": v
                        for k, v in narrative_metrics.items()
                    }
                )

        if not blocks:
            return {}

        return merge_feature_blocks_strict(*blocks)

    # =====================================================
    # VECTOR
    # =====================================================

    def extract_feature_vector(self, text: str) -> np.ndarray:

        features = self.extract_features(text)
        return self.extract_feature_vector_from_features(features)

    def extract_feature_vector_from_features(
        self,
        features: Dict[str, float],
    ) -> np.ndarray:

        if not features:
            return np.zeros(0, dtype=np.float32)

        vectors: List[np.ndarray] = []

        # -------------------------
        # ENTITY + METRICS
        # -------------------------
        try:
            vectors.append(ordered_entity_graph_vector(features))
            vectors.append(ordered_graph_metrics_vector(features))
        except Exception:
            logger.warning("Skipping entity/metrics vector")

        # -------------------------
        # EMBEDDINGS
        # -------------------------
        emb_keys = sorted(
            [k for k in features if k.startswith("graph_embedding_")]
        )

        if emb_keys:
            emb_vec = np.array(
                [features[k] for k in emb_keys],
                dtype=np.float32,
            )
            vectors.append(emb_vec)

        # -------------------------
        # NARRATIVE
        # -------------------------
        # G-S8: use the canonical ``narrative_graph_vector`` helper so
        # this block is always present at a fixed, schema-aligned
        # shape — the previous all-or-nothing ``if all(...)`` silently
        # dropped the entire block whenever a single key was missing
        # (e.g. zero-edge graphs that legitimately omit
        # ``flow_strength``), shrinking the vector mid-batch and
        # corrupting any downstream concatenation.
        if self.narrative_builder is not None or any(
            k.startswith("narrative_graph_") for k in features
        ):
            vectors.append(narrative_graph_vector(features))

        if not vectors:
            return np.zeros(0, dtype=np.float32)

        # G-S9: per-block L2 instead of one global L2 across the
        # heterogeneous concat. The blocks have wildly different
        # native scales (entity counts ~1-50, density 0-1, embedding
        # ~0-1, narrative counts 0-100) and global normalization
        # collapses small-scale signals into rounding noise whenever
        # a large-scale block dominates the norm. Per-block
        # normalization keeps each block's relative geometry intact
        # before concatenation.
        if self.config.normalize_features:
            vectors = [_normalize_vector(v) for v in vectors]

        vec = np.concatenate(vectors).astype(np.float32)

        return vec