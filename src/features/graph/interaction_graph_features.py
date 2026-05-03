from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature
from src.features.base.numerics import EPS, MAX_CLIP
from src.features.base.segmentation import (
    heuristic_entities as _heuristic_entities,
    split_sentences as _split_sentences,
)
from src.features.base.spacy_doc import ensure_spacy_doc

logger = logging.getLogger(__name__)


# Audit fix §4 — local ``_split_sentences`` / ``_heuristic_entities``
# duplicates removed; both come from ``base/segmentation.py`` so the
# entity / interaction graph builders never disagree on segmentation.


# ---------------------------------------------------------
# Feature
# ---------------------------------------------------------

@dataclass
@register_feature
class InteractionGraphFeatures(BaseFeature):

    name: str = "interaction_graph_features"
    group: str = "graph"
    description: str = "Normalized interaction graph features"

    _builder: object | None = field(default=None, init=False)
    _analyzer: object | None = field(default=None, init=False)

    # -----------------------------------------------------

    def initialize(self) -> None:
        if self._builder is not None:
            return
        try:
            from src.graph.narrative_graph_builder import NarrativeGraphBuilder
            from src.graph.graph_analysis import GraphAnalyzer

            self._builder = NarrativeGraphBuilder()
            self._analyzer = GraphAnalyzer()

        except Exception as e:
            logger.warning("Graph fallback mode: %s", e)
            self._builder = None
            self._analyzer = None

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return self._empty()

        self.initialize()

        # =====================================================
        # GRAPH BUILD
        # =====================================================

        if self._builder and self._analyzer:

            # Audit fix §2.7 — share the per-context cached spaCy
            # ``Doc`` with the entity-graph extractor (and the
            # syntactic extractor's ``extract_batch`` upstream).
            # ``NarrativeGraphBuilder.build_graph_with_doc(text, doc)``
            # returns the same ``Dict[str, Dict[str, float]]`` as
            # ``build_graph`` and skips the duplicate ``self.nlp(text)``
            # parse that previously dominated this extractor.
            doc = ensure_spacy_doc(context)

            if doc is not None and hasattr(self._builder, "build_graph_with_doc"):
                graph = self._builder.build_graph_with_doc(text, doc)
            else:
                graph = self._builder.build_graph(text)

            metrics = self._builder.extract_graph_features(graph).to_dict()
            gmetrics = self._analyzer.analyze(graph).to_dict()

            nodes = float(metrics.get("narrative_graph_nodes", 0.0))
            edges = float(metrics.get("narrative_graph_edges", 0.0))
            components = float(metrics.get("narrative_graph_components", 1.0))
            clustering = float(gmetrics.get("graph_clustering_estimate", 0.0))

        else:
            # fallback
            sentences = _split_sentences(text)

            nodes_set = set()
            edges_set = set()

            for s in sentences:
                ents = _heuristic_entities(s)
                nodes_set.update(ents)

                for pair in itertools.combinations(sorted(set(ents)), 2):
                    edges_set.add(pair)

            nodes = float(len(nodes_set))
            edges = float(len(edges_set))
            components = 1.0
            clustering = 0.0

        # =====================================================
        # NORMALIZATION
        # =====================================================

        max_edges = nodes * (nodes - 1) / 2.0 if nodes > 1 else 1.0

        # §11.4 — clamp density to [0, 1] for multigraphs where edges may
        # exceed max_edges (parallel edges between the same node pair).
        density = min(edges / (max_edges + EPS), 1.0)
        sparsity = 1.0 - density

        # normalized degree
        avg_degree = (2.0 * edges) / (nodes + EPS)
        degree_norm = avg_degree / (nodes + EPS)

        # component ratio
        component_ratio = components / (nodes + EPS)

        # =====================================================
        # ENTROPY (CRITICAL)
        # =====================================================

        probs = np.array([density, sparsity, clustering], dtype=np.float32)

        if probs.sum() > 0:
            probs = probs / (probs.sum() + EPS)
            entropy = -np.sum(probs * np.log(probs + EPS))
        else:
            entropy = 0.0

        # =====================================================
        # INTENSITY
        # =====================================================

        intensity = float(np.linalg.norm([density, degree_norm, clustering]))

        # =====================================================
        # OUTPUT
        # =====================================================

        # Audit fix §1.1 — raw log-magnitudes for size; ratios stay
        # bounded. The hand-tuned ``/ 10.0`` divisor was a population
        # constant masquerading as per-row normalisation.

        return {
            "interaction_nodes_log": self._safe_unbounded(float(np.log1p(nodes))),
            "interaction_edges_log": self._safe_unbounded(float(np.log1p(edges))),

            "interaction_density": self._safe(density),
            "interaction_sparsity": self._safe(sparsity),

            "interaction_degree_norm": self._safe(degree_norm),
            "interaction_clustering": self._safe(clustering),

            "interaction_component_ratio": self._safe(component_ratio),

            "interaction_entropy": self._safe(entropy),
            "interaction_intensity": self._safe(intensity),
        }

    # -----------------------------------------------------

    def _empty(self) -> Dict[str, float]:
        # §11.1 — consistent fixed-key zero dict for empty-text inputs.
        return {
            "interaction_nodes_log":      0.0,
            "interaction_edges_log":      0.0,
            "interaction_density":        0.0,
            "interaction_sparsity":       0.0,
            "interaction_degree_norm":    0.0,
            "interaction_clustering":     0.0,
            "interaction_component_ratio": 0.0,
            "interaction_entropy":        0.0,
            "interaction_intensity":      0.0,
        }

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))

    def _safe_unbounded(self, v: float) -> float:
        """Drop NaN / negative values without applying an upper clip.

        Audit fix §1.1 — raw magnitudes are scaled population-wide by
        the :class:`FeatureScalingPipeline`, not per-row.
        """
        if not np.isfinite(v) or v < 0:
            return 0.0
        return float(v)