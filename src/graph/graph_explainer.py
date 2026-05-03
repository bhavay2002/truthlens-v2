from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

from src.graph.graph_analysis import canonicalize_weighted
from src.graph.temporal_graph import TemporalGraphAnalyzer

logger = logging.getLogger(__name__)
EPS = 1e-12

GraphLike = Union[Dict[str, Dict[str, float]], Dict[str, List[str]]]


# =========================================================
# OUTPUT STRUCTURE
# =========================================================

@dataclass
class GraphExplanation:

    node_importance: Dict[str, float]
    edge_importance: Dict[str, float]
    temporal_importance: Dict[str, float]

    overall_score: float

    def to_dict(self) -> Dict:
        return {
            "node_importance": self.node_importance,
            "edge_importance": self.edge_importance,
            "temporal_importance": self.temporal_importance,
            "overall_score": self.overall_score,
        }


# =========================================================
# CORE EXPLAINER
# =========================================================

class GraphExplainer:

    def __init__(
        self,
        *,
        node_weight: float = 0.4,
        edge_weight: float = 0.3,
        temporal_weight: float = 0.3,
    ):
        # Kept for the ``explain_from_text`` shortcut path. The main
        # ``explain(...)`` API now prefers a pre-computed temporal dict
        # supplied by the pipeline, so this analyzer is rarely invoked
        # twice per request anymore (G-R2).
        self.temporal = TemporalGraphAnalyzer()

        # G-CFG2: mixing weights used to be hardcoded ``0.4 / 0.3 / 0.3``
        # inside ``_overall_score``. They now flow from
        # ``GraphConfig`` via the pipeline so a single edit in
        # ``config/config.yaml`` retunes the explainer's score blend
        # without code changes. Validation lives in ``GraphConfigLoader``
        # (must sum to 1.0); we add a defensive guard here too in case
        # a caller bypasses the config layer.
        weight_sum = node_weight + edge_weight + temporal_weight
        if not (0.99 <= weight_sum <= 1.01):
            raise ValueError(
                "explainer weights must sum to 1.0 "
                f"(got {weight_sum:.4f})"
            )
        self.node_weight = float(node_weight)
        self.edge_weight = float(edge_weight)
        self.temporal_weight = float(temporal_weight)

        logger.info(
            "GraphExplainer initialized (weights node=%.2f edge=%.2f temporal=%.2f)",
            self.node_weight,
            self.edge_weight,
            self.temporal_weight,
        )

    # =====================================================
    # NODE IMPORTANCE  (G-C5: weight-aware)
    # =====================================================

    def _node_importance(
        self,
        graph: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:

        if not graph:
            return {}

        # Weighted degree — heavy edges contribute more to importance,
        # which is the entire point of building a weighted graph upstream.
        weighted_degrees = {n: sum(nbrs.values()) for n, nbrs in graph.items()}

        total = sum(weighted_degrees.values()) + EPS

        if total <= EPS:
            # Fallback to unweighted normalisation when the canonical
            # form has no weighted edges (e.g. an isolated-nodes graph).
            n = len(graph)
            return {k: 1.0 / n for k in graph}

        return {k: float(v / total) for k, v in weighted_degrees.items()}

    # =====================================================
    # EDGE IMPORTANCE  (G-C5: weight-aware)
    # =====================================================

    def _edge_importance(
        self,
        graph: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:

        edge_scores: Dict[Tuple[str, str], float] = {}

        for src, nbrs in graph.items():
            for tgt, w in nbrs.items():

                if src == tgt:
                    continue

                edge = tuple(sorted((src, tgt)))

                # Symmetric weighted graph — ``w`` already equals the
                # value at ``graph[tgt][src]`` (canonicalize_weighted
                # symmetrises with ``max``); take it once per pair.
                edge_scores[edge] = max(edge_scores.get(edge, 0.0), float(w))

        if not edge_scores:
            return {}

        total = sum(edge_scores.values()) + EPS

        return {
            f"{a}->{b}": float(v / total)
            for (a, b), v in edge_scores.items()
        }

    # =====================================================
    # TEMPORAL IMPORTANCE
    # =====================================================

    def _normalise_temporal(
        self,
        features: Mapping[str, Any],
    ) -> Dict[str, float]:

        # Coerce to a flat ``Dict[str, float]`` ignoring non-numeric values.
        clean: Dict[str, float] = {}
        for k, v in features.items():
            try:
                clean[str(k)] = float(v)
            except (TypeError, ValueError):
                continue

        if not clean:
            return {}

        vals = np.array(list(clean.values()), dtype=float)

        if np.sum(vals) == 0:
            return clean

        vals = vals / (np.sum(vals) + EPS)
        return dict(zip(clean.keys(), vals.tolist()))

    def _temporal_importance(
        self,
        text: Optional[str],
    ) -> Dict[str, float]:

        if not text:
            return {}

        return self._normalise_temporal(self.temporal.analyze(text).to_dict())

    # =====================================================
    # OVERALL SCORE
    # =====================================================

    def _overall_score(
        self,
        node_imp: Dict[str, float],
        edge_imp: Dict[str, float],
        temporal_imp: Dict[str, float],
    ) -> float:

        node_score = float(np.mean(list(node_imp.values()))) if node_imp else 0.0
        edge_score = float(np.mean(list(edge_imp.values()))) if edge_imp else 0.0
        temp_score = float(np.mean(list(temporal_imp.values()))) if temporal_imp else 0.0

        return float(
            self.node_weight * node_score
            + self.edge_weight * edge_score
            + self.temporal_weight * temp_score
        )

    # =====================================================
    # PUBLIC API
    # =====================================================

    def explain(
        self,
        *,
        entity_graph: Optional[GraphLike] = None,
        narrative_graph: Optional[GraphLike] = None,
        text: Optional[str] = None,
        temporal_features: Optional[Mapping[str, Any]] = None,
    ) -> GraphExplanation:
        """Build a graph-level explanation.

        G-C3 fix: previously raised ``TypeError`` when the pipeline
        passed ``temporal_features=`` (it wasn't in the signature).
        Now accepted and used directly, avoiding a second
        ``TemporalGraphAnalyzer.analyze`` call per request (G-R2).
        ``text=`` remains accepted for the standalone path.
        """

        graph = entity_graph if entity_graph else (narrative_graph or {})

        # G-C5: canonicalize once so the importance functions see a
        # weighted symmetric dict whether the caller passed weighted,
        # unweighted, or already-canonical input.
        canon = canonicalize_weighted(graph) if graph else {}

        node_imp = self._node_importance(canon)
        edge_imp = self._edge_importance(canon)

        if temporal_features is not None:
            temporal_imp = self._normalise_temporal(temporal_features)
        else:
            temporal_imp = self._temporal_importance(text)

        score = self._overall_score(node_imp, edge_imp, temporal_imp)

        return GraphExplanation(
            node_importance=node_imp,
            edge_importance=edge_imp,
            temporal_importance=temporal_imp,
            overall_score=score,
        )

    # =====================================================
    # TEXT-ONLY SHORTCUT
    # =====================================================

    def explain_from_text(self, text: str) -> Dict:

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Invalid text")

        temporal_imp = self._temporal_importance(text)

        return {
            "temporal_importance": temporal_imp,
            "overall_score": float(np.mean(list(temporal_imp.values())) if temporal_imp else 0.0),
        }
