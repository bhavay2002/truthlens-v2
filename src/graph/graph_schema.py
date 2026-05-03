from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, field_validator


# =========================================================
# BASE VALIDATION
# =========================================================

def _validate_numeric(v: Optional[float]) -> Optional[float]:
    if v is None:
        return v

    if isinstance(v, bool):
        raise TypeError("Must be numeric")

    v = float(v)

    if not math.isfinite(v):
        raise ValueError("Must be finite")

    return v


# =========================================================
# GRAPH STRUCTURES
# =========================================================

class GraphStructure(BaseModel):
    """
    Adjacency graph structure
    """

    model_config = ConfigDict(extra="forbid")

    nodes: List[str]
    edges: Dict[str, List[str]]

    @field_validator("nodes")
    @classmethod
    def validate_nodes(cls, v):
        if v is None:
            raise ValueError("nodes cannot be None")
        # G-C2: allow an empty node list so an article with no detected
        # entities still serialises through the schema (callers wrap an
        # empty graph in ``Optional[GraphStructure]`` rather than passing
        # ``None`` so downstream UI can distinguish "no content" from
        # "graph disabled").
        return list(v)

    @field_validator("edges")
    @classmethod
    def validate_edges(cls, v):
        if not isinstance(v, dict):
            raise TypeError("edges must be dict")
        return v


# =========================================================
# GRAPH FEATURES
# =========================================================

class GraphFeatureModel(BaseModel):
    """
    Unified graph feature container
    """

    model_config = ConfigDict(extra="allow")

    # core metrics
    graph_nodes: float
    graph_edges: float
    graph_density: float

    # optional extended features
    graph_entropy: Optional[float] = None
    graph_centralization: Optional[float] = None
    graph_clustering: Optional[float] = None

    @field_validator("*", mode="before")
    @classmethod
    def validate_numeric(cls, v):
        return _validate_numeric(v)


# =========================================================
# TEMPORAL FEATURES
# =========================================================

class TemporalFeatureModel(BaseModel):
    """
    Temporal narrative features
    """

    model_config = ConfigDict(extra="allow")

    entity_recurrence: float
    entity_transition_rate: float
    topic_shift_score: float
    narrative_drift: float

    # advanced
    temporal_entropy: Optional[float] = None
    narrative_volatility: Optional[float] = None
    temporal_consistency: Optional[float] = None

    @field_validator("*", mode="before")
    @classmethod
    def validate_numeric(cls, v):
        return _validate_numeric(v)


# =========================================================
# GRAPH EMBEDDINGS
# =========================================================

class GraphEmbeddingModel(BaseModel):
    """
    Graph embedding vector
    """

    model_config = ConfigDict(extra="forbid")

    embedding: List[float]

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v):
        if not v:
            raise ValueError("embedding cannot be empty")
        return [float(x) for x in v]


# =========================================================
# GRAPH EXPLANATION
# =========================================================

class GraphExplanationModel(BaseModel):
    """
    Explainability for graph module
    """

    model_config = ConfigDict(extra="forbid")

    node_importance: Dict[str, float]
    edge_importance: Dict[str, float]
    temporal_importance: Dict[str, float]

    overall_score: float

    @field_validator("overall_score", mode="before")
    @classmethod
    def validate_score(cls, v):
        return _validate_numeric(v)

    @field_validator("node_importance", "edge_importance", "temporal_importance")
    @classmethod
    def validate_maps(cls, v):
        if not isinstance(v, dict):
            raise TypeError("Must be dict")
        return {str(k): float(vv) for k, vv in v.items()}


# =========================================================
# FINAL GRAPH OUTPUT
# =========================================================

class GraphOutput(BaseModel):
    """
    Full graph module output (single unified object)
    """

    model_config = ConfigDict(extra="forbid")

    # structures
    entity_graph: Optional[GraphStructure] = None
    narrative_graph: Optional[GraphStructure] = None

    # features
    features: Optional[Dict[str, float]] = None
    temporal_features: Optional[TemporalFeatureModel] = None

    # G-C2 / G-C4: per-graph metric maps. The pipeline computes these
    # via ``GraphAnalyzer.analyze`` and used to silently drop them
    # (the consumer read ``entity_graph_metrics`` while the producer
    # only had ``entity_metrics`` in the local scope and threw on the
    # ``GraphOutput(...)`` call). Now first-class fields.
    entity_metrics: Optional[Dict[str, float]] = None
    narrative_metrics: Optional[Dict[str, float]] = None

    # G-C6: per-mention character spans + tokenizer discriminator.
    # ``GraphPipeline._run_with_doc`` already populates ``entity_spans``,
    # ``narrative_spans`` and ``narrative_tokenizer`` on the result
    # dict (so the API / explainer can highlight node IDs back into
    # source text), but ``GraphOutput`` was declared with
    # ``extra="forbid"`` and did not list the keys — meaning any
    # consumer that typed its input as ``GraphOutput`` could not read
    # them, and constructing ``GraphOutput`` *with* them would raise
    # ``ValidationError``. They flow through the typed envelope now.
    entity_spans: Optional[List[Dict[str, Any]]] = None
    narrative_spans: Optional[List[Dict[str, Any]]] = None
    narrative_tokenizer: Optional[str] = None

    # embeddings
    embeddings: Optional[GraphEmbeddingModel] = None

    # explainability
    explanation: Optional[GraphExplanationModel] = None

    # metadata
    metadata: Optional[Dict[str, str]] = None

    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        if v is None:
            return v

        if not isinstance(v, dict):
            raise TypeError("features must be dict")

        return {str(k): float(vv) for k, vv in v.items()}

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v):
        if v is None:
            return v

        if not isinstance(v, dict):
            raise TypeError("metadata must be dict")

        return {str(k): str(vv) for k, vv in v.items()}