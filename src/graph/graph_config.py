from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


# =========================================================
# YAML LOADER
# =========================================================

def load_yaml_as_dict(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML dict at {path}")

    return data


# =========================================================
# PARSER
# =========================================================
#
# G-CFG1 / G-CFG2 / G-CFG3 (audit §8): the previous parser only
# accepted a small set of fields and silently discarded everything
# else, so the YAML ``graph:`` block could never fully drive runtime
# behaviour. Every knob the pipeline actually uses now flows through
# here, with the same defaults that used to be baked into the various
# ``__init__`` methods. New keys are documented in ``config/config.yaml``.
# =========================================================

def parse_graph_config(config_data: Dict[str, Any]) -> Dict[str, Any]:

    graph = config_data.get("graph", config_data)

    return {
        # ---- core toggles ----
        "enable_entity_graph": bool(graph.get("enable_entity_graph", True)),
        "enable_narrative_graph": bool(graph.get("enable_narrative_graph", True)),
        "enable_temporal_graph": bool(graph.get("enable_temporal_graph", True)),
        "enable_graph_explainer": bool(graph.get("enable_graph_explainer", True)),

        # ---- runtime behaviour (G-CFG3) ----
        "return_vector": bool(graph.get("return_vector", True)),
        "run_analysis_modules": bool(graph.get("run_analysis_modules", True)),
        "batch_size": int(graph.get("batch_size", 32)),

        # ---- keyword / token extraction (G-CFG2) ----
        "min_keyword_length": int(graph.get("min_keyword_length", 4)),
        "max_keywords_per_sentence": int(graph.get("max_keywords_per_sentence", 4)),
        "temporal_min_token_length": int(
            graph.get("temporal_min_token_length", 4)
        ),

        # ---- graph behavior ----
        "use_weighted_edges": bool(graph.get("use_weighted_edges", True)),
        "normalize_graph": bool(graph.get("normalize_graph", True)),

        # ---- thresholds ----
        "min_edge_weight": float(graph.get("min_edge_weight", 0.0)),
        "max_edge_weight": float(graph.get("max_edge_weight", 10.0)),

        # ---- scaling ----
        "feature_scale": float(graph.get("feature_scale", 1.0)),

        # ---- embeddings (G-CFG2) ----
        "enable_graph_embeddings": bool(graph.get("enable_graph_embeddings", False)),
        "embedding_type": str(graph.get("embedding_type", "hybrid")),
        "spectral_dim": int(graph.get("spectral_dim", 8)),
        "embedding_dim": int(graph.get("embedding_dim", 16)),
        "walk_length": int(graph.get("walk_length", 10)),
        "num_walks": int(graph.get("num_walks", 10)),

        # ---- explainer mixing weights (G-CFG2) ----
        "explainer_node_weight": float(graph.get("explainer_node_weight", 0.4)),
        "explainer_edge_weight": float(graph.get("explainer_edge_weight", 0.3)),
        "explainer_temporal_weight": float(
            graph.get("explainer_temporal_weight", 0.3)
        ),
    }


# =========================================================
# DATACLASS
# =========================================================

@dataclass(slots=True)
class GraphConfig:

    # toggles
    enable_entity_graph: bool = True
    enable_narrative_graph: bool = True
    enable_temporal_graph: bool = True
    enable_graph_explainer: bool = True

    # runtime behaviour (G-CFG3)
    return_vector: bool = True
    run_analysis_modules: bool = True
    batch_size: int = 32

    # extraction (G-CFG2)
    min_keyword_length: int = 4
    max_keywords_per_sentence: int = 4
    temporal_min_token_length: int = 4

    # graph behavior
    use_weighted_edges: bool = True
    normalize_graph: bool = True

    # thresholds
    min_edge_weight: float = 0.0
    max_edge_weight: float = 10.0

    # scaling
    feature_scale: float = 1.0

    # embeddings (G-CFG2)
    enable_graph_embeddings: bool = False
    embedding_type: str = "hybrid"
    spectral_dim: int = 8
    embedding_dim: int = 16
    walk_length: int = 10
    num_walks: int = 10

    # explainer mixing weights (G-CFG2)
    explainer_node_weight: float = 0.4
    explainer_edge_weight: float = 0.3
    explainer_temporal_weight: float = 0.3


# =========================================================
# LOADER
# =========================================================

class GraphConfigLoader:

    def __init__(self):
        logger.info("GraphConfigLoader initialized")

    def load_from_yaml(self, path: str | Path) -> GraphConfig:

        p = Path(path)

        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")

        data = load_yaml_as_dict(p)
        return self._parse(data)

    def load_from_dict(self, config: Dict[str, Any]) -> GraphConfig:

        if not isinstance(config, dict):
            raise TypeError("config must be dict")

        return self._parse(config)

    # =====================================================
    # INTERNAL
    # =====================================================

    def _parse(self, config_data: Dict[str, Any]) -> GraphConfig:

        parsed = parse_graph_config(config_data)

        cfg = GraphConfig(**parsed)

        self._validate(cfg)

        logger.info("GraphConfig loaded")

        return cfg

    # =====================================================
    # VALIDATION
    # =====================================================

    def _validate(self, cfg: GraphConfig) -> None:

        if cfg.min_keyword_length < 1:
            raise ValueError("min_keyword_length must be >= 1")

        if cfg.max_keywords_per_sentence < 1:
            raise ValueError("max_keywords_per_sentence must be >= 1")

        if cfg.temporal_min_token_length < 1:
            raise ValueError("temporal_min_token_length must be >= 1")

        if cfg.min_edge_weight < 0:
            raise ValueError("min_edge_weight must be >= 0")

        if cfg.max_edge_weight <= cfg.min_edge_weight:
            raise ValueError("max_edge_weight must be > min_edge_weight")

        if not (0.0 < cfg.feature_scale <= 10.0):
            raise ValueError("feature_scale must be in (0, 10]")

        if cfg.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        if cfg.spectral_dim < 1:
            raise ValueError("spectral_dim must be >= 1")

        if cfg.embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1")

        if cfg.walk_length < 1:
            raise ValueError("walk_length must be >= 1")

        if cfg.num_walks < 1:
            raise ValueError("num_walks must be >= 1")

        # G-CFG2: explainer mixing weights are a convex combination —
        # validate that, otherwise the score can leave [0, 1] silently.
        weight_sum = (
            cfg.explainer_node_weight
            + cfg.explainer_edge_weight
            + cfg.explainer_temporal_weight
        )

        if not (0.99 <= weight_sum <= 1.01):
            raise ValueError(
                "explainer_*_weight must sum to 1.0 "
                f"(got {weight_sum:.4f})"
            )

        for name, val in (
            ("explainer_node_weight", cfg.explainer_node_weight),
            ("explainer_edge_weight", cfg.explainer_edge_weight),
            ("explainer_temporal_weight", cfg.explainer_temporal_weight),
        ):
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")

        if not isinstance(cfg.enable_entity_graph, bool):
            raise TypeError("enable_entity_graph must be bool")

        if not isinstance(cfg.enable_narrative_graph, bool):
            raise TypeError("enable_narrative_graph must be bool")

        if not isinstance(cfg.enable_temporal_graph, bool):
            raise TypeError("enable_temporal_graph must be bool")

        if not isinstance(cfg.enable_graph_explainer, bool):
            raise TypeError("enable_graph_explainer must be bool")

        if cfg.embedding_type.lower() not in {
            "degree",
            "centrality",
            "spectral",
            "hybrid",
            "node2vec",
        }:
            raise ValueError(
                f"embedding_type must be one of degree|centrality|spectral|hybrid|node2vec "
                f"(got {cfg.embedding_type!r})"
            )


# =========================================================
# UTILITIES
# =========================================================

def clip_edge_weight(value: float, cfg: GraphConfig) -> float:
    return float(max(cfg.min_edge_weight, min(value, cfg.max_edge_weight)))


def scale_feature(value: float, cfg: GraphConfig) -> float:
    return float(value * cfg.feature_scale)


# =========================================================
# DEFAULT YAML LOOKUP  (G-CFG1)
# =========================================================
#
# The pipeline used to reach for ``GraphPipelineConfig()`` (a
# different, smaller dataclass with no YAML knowledge) every time it
# was instantiated. ``load_default_graph_config`` gives every entry
# point a single, defensive way to pick up ``config/config.yaml`` —
# falls back to the dataclass defaults when the YAML or the ``graph:``
# block is missing, so import-time consumers keep working in
# environments without a config file.

_DEFAULT_YAML = Path("config/config.yaml")


def load_default_graph_config(path: str | Path | None = None) -> GraphConfig:
    """Load the project's graph config, with a quiet fallback."""

    p = Path(path) if path is not None else _DEFAULT_YAML

    if not p.exists():
        logger.debug("No graph YAML at %s — using defaults", p)
        return GraphConfig()

    try:
        return GraphConfigLoader().load_from_yaml(p)
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("Failed to load graph config from %s: %s", p, exc)
        return GraphConfig()
