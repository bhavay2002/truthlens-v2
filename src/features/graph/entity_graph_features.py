from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature
from src.features.base.numerics import EPS, MAX_CLIP
from src.features.base.segmentation import (
    heuristic_entities as _heuristic_entities,
    split_sentences as _sentence_split,
)
from src.features.base.spacy_doc import ensure_spacy_doc

logger = logging.getLogger(__name__)


# Audit fix §4 — local ``_sentence_split`` / ``_heuristic_entities``
# duplicates removed; the canonical implementations now live in
# ``src/features/base/segmentation.py`` and are shared with the
# trajectory and interaction-graph extractors.


# ---------------------------------------------------------
# Feature
# ---------------------------------------------------------

@dataclass
@register_feature
class EntityGraphFeatures(BaseFeature):

    name: str = "entity_graph_features"
    group: str = "graph"
    description: str = "Normalized entity graph structural features"

    _builder: object | None = field(default=None, init=False)
    _analyzer: object | None = field(default=None, init=False)

    # -----------------------------------------------------

    def initialize(self) -> None:
        if self._builder is not None:
            return
        try:
            from src.graph.entity_graph import EntityGraphBuilder
            from src.graph.graph_analysis import GraphAnalyzer

            self._builder = EntityGraphBuilder()
            self._analyzer = GraphAnalyzer()

        except Exception as e:
            logger.warning("Graph system unavailable → fallback: %s", e)
            self._builder = None
            self._analyzer = None

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return self._empty()

        self.initialize()

        # =====================================================
        # GRAPH PIPELINE
        # =====================================================

        if self._builder and self._analyzer:

            # Audit fix §2.7 — reuse the per-context cached spaCy
            # ``Doc`` if any earlier extractor (e.g. the syntactic
            # extractor's ``extract_batch``) has already parsed the
            # text. ``EntityGraphBuilder.build_graph_with_doc`` skips
            # the ``self.nlp(text)`` re-parse and returns the same
            # graph shape as ``build_graph``. Falls back to the
            # text-only path when no cached doc is available so the
            # behaviour is identical when this extractor runs alone.
            doc = ensure_spacy_doc(context)

            if doc is not None and hasattr(self._builder, "build_graph_with_doc"):
                result = self._builder.build_graph_with_doc(doc)
                graph = result["graph"] if isinstance(result, dict) and "graph" in result else result
            else:
                graph = self._builder.build_graph(text)

            metrics = self._builder.extract_graph_features(graph).to_dict()
            gmetrics = self._analyzer.analyze(graph).to_dict()

            nodes = float(metrics.get("entity_graph_nodes", 0.0))
            edges = float(metrics.get("entity_graph_edges", 0.0))

        else:
            # fallback
            sentences = _sentence_split(text)

            entities = set()
            edges = 0

            for s in sentences:
                ents = _heuristic_entities(s)
                entities.update(ents)

                n = len(ents)
                if n > 1:
                    edges += (n * (n - 1)) // 2

            nodes = float(len(entities))
            gmetrics = {}

        # =====================================================
        # NORMALIZATION
        # =====================================================

        max_edges = nodes * (nodes - 1) / 2.0 if nodes > 1 else 1.0

        # §11.4 — for multigraphs (parallel edges allowed) `edges` can exceed
        # `max_edges`, producing density > 1.0.  Clamp to [0, 1] so the value
        # is always a valid ratio that the scaling stage can handle.
        density = min(edges / (max_edges + EPS), 1.0)

        # normalized degree
        avg_degree = (2.0 * edges) / (nodes + EPS)
        degree_norm = avg_degree / (nodes + EPS)

        # sparsity
        sparsity = 1.0 - density

        # =====================================================
        # ENTROPY (CRITICAL)
        # =====================================================

        probs = np.array([density, sparsity], dtype=np.float32)

        if probs.sum() > 0:
            probs = probs / (probs.sum() + EPS)
            entropy = -np.sum(probs * np.log(probs + EPS))
        else:
            entropy = 0.0

        # =====================================================
        # INTENSITY
        # =====================================================

        intensity = float(np.linalg.norm([density, degree_norm]))

        # =====================================================
        # OUTPUT
        # =====================================================

        # Audit fix §1.1 — emit raw log-magnitudes and let
        # FeatureScalingPipeline learn the normalisation. The previous
        # ``/ 10.0`` divisor implicitly assumed a corpus where
        # ``log1p(nodes) ~ 10`` (≈ 22k nodes), saturating short docs at
        # near-zero and clipping long docs at 1.0.

        return {
            "graph_nodes_log": self._safe_unbounded(float(np.log1p(nodes))),
            "graph_edges_log": self._safe_unbounded(float(np.log1p(edges))),

            "graph_density": self._safe(density),
            "graph_sparsity": self._safe(sparsity),

            "graph_degree_norm": self._safe(degree_norm),

            "graph_entropy": self._safe(entropy),
            "graph_intensity": self._safe(intensity),
        }

    # -----------------------------------------------------

    def _empty(self) -> Dict[str, float]:
        # §11.1 — consistent fixed-key zero dict so the schema validator sees
        # a stable shape for empty-text inputs.
        return {
            "graph_nodes_log":   0.0,
            "graph_edges_log":   0.0,
            "graph_density":     0.0,
            "graph_sparsity":    0.0,
            "graph_degree_norm": 0.0,
            "graph_entropy":     0.0,
            "graph_intensity":   0.0,
        }

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))

    def _safe_unbounded(self, v: float) -> float:
        """Drop NaN / negative values without applying an upper clip.

        Audit fix §1.1 — raw log-magnitudes flow through to the
        :class:`FeatureScalingPipeline` for corpus-aware normalisation.
        """
        if not np.isfinite(v) or v < 0:
            return 0.0
        return float(v)