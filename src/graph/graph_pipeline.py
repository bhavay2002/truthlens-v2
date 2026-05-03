from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.graph.entity_graph import EntityGraphBuilder
from src.graph.narrative_graph_builder import NarrativeGraphBuilder
from src.graph.graph_analysis import GraphAnalyzer, canonicalize_weighted
from src.graph.graph_features import GraphFeatureExtractor, GraphFeatureExtractorConfig
from src.graph.temporal_graph import TemporalGraphAnalyzer
from src.graph.graph_explainer import GraphExplainer
from src.graph.graph_schema import GraphOutput, GraphStructure
from src.graph.graph_embeddings import GraphEmbeddingConfig
from src.graph.graph_config import GraphConfig, load_default_graph_config

# G-D4: ``AnalysisIntegrationRunner`` pulls in 15 analysis modules at
# import time. Top-level import made every consumer that touched
# ``src.graph.graph_pipeline`` pay that cost — even when
# ``run_analysis_modules: false``. Imported lazily inside ``__init__``
# below, only when the runner is actually being constructed.

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass(slots=True)
class GraphPipelineConfig:

    enable_entity_graph: bool = True
    enable_narrative_graph: bool = True
    enable_temporal_graph: bool = True

    enable_graph_explainer: bool = True

    return_vector: bool = True
    run_analysis_modules: bool = True

    # G-P3: batch size used by ``run_batch`` when calling ``nlp.pipe``.
    batch_size: int = 32

    # G-CFG2: tunables previously baked into ``__init__`` defaults of
    # the various builders. Surfaced here so a YAML ``graph:`` block
    # can drive them end-to-end without monkey-patching.
    min_keyword_length: int = 4
    max_keywords_per_sentence: int = 4
    temporal_min_token_length: int = 4

    enable_graph_embeddings: bool = False
    embedding_type: str = "hybrid"
    spectral_dim: int = 8
    embedding_dim: int = 16
    walk_length: int = 10
    num_walks: int = 10

    explainer_node_weight: float = 0.4
    explainer_edge_weight: float = 0.3
    explainer_temporal_weight: float = 0.3

    # =====================================================
    # G-CFG1: translate the YAML-aware ``GraphConfig`` into
    # the runtime config the pipeline actually consumes.
    # =====================================================
    @classmethod
    def from_graph_config(cls, cfg: "GraphConfig") -> "GraphPipelineConfig":
        return cls(
            enable_entity_graph=cfg.enable_entity_graph,
            enable_narrative_graph=cfg.enable_narrative_graph,
            enable_temporal_graph=cfg.enable_temporal_graph,
            enable_graph_explainer=cfg.enable_graph_explainer,
            return_vector=cfg.return_vector,
            run_analysis_modules=cfg.run_analysis_modules,
            batch_size=cfg.batch_size,
            min_keyword_length=cfg.min_keyword_length,
            max_keywords_per_sentence=cfg.max_keywords_per_sentence,
            temporal_min_token_length=cfg.temporal_min_token_length,
            enable_graph_embeddings=cfg.enable_graph_embeddings,
            embedding_type=cfg.embedding_type,
            spectral_dim=cfg.spectral_dim,
            embedding_dim=cfg.embedding_dim,
            walk_length=cfg.walk_length,
            num_walks=cfg.num_walks,
            explainer_node_weight=cfg.explainer_node_weight,
            explainer_edge_weight=cfg.explainer_edge_weight,
            explainer_temporal_weight=cfg.explainer_temporal_weight,
        )

    @classmethod
    def from_yaml(cls, path: str | None = None) -> "GraphPipelineConfig":
        return cls.from_graph_config(load_default_graph_config(path))


# =========================================================
# SCHEMA HELPERS  (G-C2)
# =========================================================

def _to_graph_structure(
    graph: Optional[Dict[str, Dict[str, float]]],
) -> Optional[GraphStructure]:
    """Adapt a weighted adjacency dict to the ``GraphStructure`` schema.

    Used only at the boundary where ``GraphOutput`` is constructed.
    Returns ``None`` for empty input so the optional field stays absent
    rather than carrying a dummy node.
    """
    if not graph:
        return None

    nodes = sorted(
        set(graph.keys())
        | {n for nbrs in graph.values() for n in (nbrs.keys() if isinstance(nbrs, dict) else nbrs)}
    )

    if not nodes:
        return None

    edges: Dict[str, List[str]] = {}
    for n in nodes:
        nbrs = graph.get(n, {})
        if isinstance(nbrs, dict):
            edges[n] = sorted(nbrs.keys())
        else:
            edges[n] = sorted(nbrs)

    return GraphStructure(nodes=nodes, edges=edges)


# =========================================================
# PIPELINE
# =========================================================

class GraphPipeline:

    def __init__(self, config: Optional[GraphPipelineConfig] = None):

        # G-CFG1: when no explicit config is supplied, hydrate from the
        # YAML ``graph:`` block so a single edit in ``config/config.yaml``
        # actually drives runtime behaviour. ``load_default_graph_config``
        # falls back to dataclass defaults if the YAML is missing.
        if config is None:
            config = GraphPipelineConfig.from_yaml()

        self.config = config

        # -------------------------
        # Builders — G-CFG2: every tunable now flows from self.config.
        # -------------------------
        self.entity_graph_builder = (
            EntityGraphBuilder() if self.config.enable_entity_graph else None
        )

        self.narrative_graph_builder = (
            NarrativeGraphBuilder(
                min_token_length=self.config.min_keyword_length,
                max_keywords_per_sentence=self.config.max_keywords_per_sentence,
            )
            if self.config.enable_narrative_graph
            else None
        )

        self.temporal_analyzer = (
            TemporalGraphAnalyzer(
                min_token_length=self.config.temporal_min_token_length,
            )
            if self.config.enable_temporal_graph
            else None
        )

        self.graph_analyzer = GraphAnalyzer()

        self.graph_feature_extractor = GraphFeatureExtractor(
            GraphFeatureExtractorConfig(
                enable_entity_graph=self.config.enable_entity_graph,
                enable_narrative_graph=self.config.enable_narrative_graph,
                enable_embeddings=self.config.enable_graph_embeddings,
                embedding_config=GraphEmbeddingConfig(
                    embedding_type=self.config.embedding_type,
                    spectral_dim=self.config.spectral_dim,
                    embedding_dim=self.config.embedding_dim,
                    walk_length=self.config.walk_length,
                    num_walks=self.config.num_walks,
                ),
            )
        )

        self.graph_explainer = (
            GraphExplainer(
                node_weight=self.config.explainer_node_weight,
                edge_weight=self.config.explainer_edge_weight,
                temporal_weight=self.config.explainer_temporal_weight,
            )
            if self.config.enable_graph_explainer
            else None
        )

        # G-D4: lazy import — only pay the cost of dragging in the 15
        # analysis modules when the caller actually opted into them.
        if self.config.run_analysis_modules:
            from src.analysis.integration_runner import AnalysisIntegrationRunner
            self.analysis_runner = AnalysisIntegrationRunner()
        else:
            self.analysis_runner = None

        logger.info("GraphPipeline initialized")

    # =====================================================
    # CONFIG FINGERPRINT (audit fix #1.2)
    # =====================================================

    def config_fingerprint(self) -> str:

        try:
            payload = asdict(self.config)
        except TypeError:
            payload = {
                k: getattr(self.config, k)
                for k in dir(self.config)
                if not k.startswith("_")
                and not callable(getattr(self.config, k))
            }

        # G-CFG4: ``batch_feature_pipeline._build_cache_key`` keys on
        # this fingerprint. Previously it hashed only the dataclass
        # fields, so a spaCy model swap (``en_core_web_sm`` →
        # ``en_core_web_trf`` → blank fallback) silently re-used cached
        # graphs from the *previous* model — a structurally different
        # parse producing identical cache hits. Mix in the active
        # model identity (name + version + pipe_names) so the cache
        # key invalidates whenever the parser changes.
        nlp_meta: Dict[str, Any] = {}
        if self.entity_graph_builder is not None:
            nlp = getattr(self.entity_graph_builder, "nlp", None)
            if nlp is not None:
                meta = getattr(nlp, "meta", {}) or {}
                nlp_meta = {
                    "name": meta.get("name", ""),
                    "version": meta.get("version", ""),
                    "lang": meta.get("lang", ""),
                    "pipe_names": list(getattr(nlp, "pipe_names", []) or []),
                }
        payload["__nlp__"] = nlp_meta

        raw = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    # =====================================================
    # VALIDATION
    # =====================================================

    def _validate_text(self, text: str):

        if not isinstance(text, str):
            raise TypeError("text must be string")

        if not text.strip():
            raise ValueError("text must be non-empty")

    # =====================================================
    # ENTITY GRAPH (factored so run / run_batch share code)
    # =====================================================

    def _entity_graph_from_doc(
        self,
        doc,
    ) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
        """Build the weighted entity graph from a pre-parsed spaCy ``Doc``.

        G-D1: now a thin shim that delegates to
        :meth:`EntityGraphBuilder.build_graph_with_doc` so the per-doc
        graph-construction logic lives in exactly one place. This used
        to be a 40-line copy of ``build_graph_with_spans`` minus the
        ``self.nlp(text)`` call — two implementations to keep in sync.
        """
        if self.entity_graph_builder is None:
            return {}, []

        payload = self.entity_graph_builder.build_graph_with_doc(doc)
        return payload["graph"], payload.get("spans", [])

    # =====================================================
    # MAIN
    # =====================================================

    def run(self, text: str) -> Dict[str, Any]:

        self._validate_text(text)

        if self.entity_graph_builder is not None:
            doc = self.entity_graph_builder.nlp(text)
        else:
            doc = None

        return self._run_with_doc(text, doc)

    # =====================================================
    # G-P3: batched variant
    # =====================================================

    def run_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Vectorised counterpart of :meth:`run`.

        Uses ``spacy.Language.pipe`` to parse the full batch in a single
        call instead of one-by-one; meaningful speedup for batch
        inference (`batch_inference.py`, `feature_pipeline.batch_extract`)
        which previously paid the per-doc spaCy overhead N times.
        Narrative / temporal stages still run per-doc — they're pure
        Python regex and dominated by entity-graph parsing in profile.
        """

        if not texts:
            return []

        for t in texts:
            self._validate_text(t)

        if self.entity_graph_builder is not None:
            docs = list(
                self.entity_graph_builder.nlp.pipe(
                    texts, batch_size=self.config.batch_size
                )
            )
        else:
            docs = [None] * len(texts)

        # G-E5: previously a list comprehension — one bad doc (e.g. an
        # explainer crash on a malformed sub-graph that slipped past
        # validation) tore down the entire batch and the caller lost
        # the 31 successful results. Isolate per-doc failures with a
        # sentinel ``{"error": str}`` matching what
        # ``inference_pipeline._fail_safe`` already expects, so the
        # rest of the batch reaches the model.
        results: List[Dict[str, Any]] = []
        for t, d in zip(texts, docs):
            try:
                results.append(self._run_with_doc(t, d))
            except Exception as exc:  # noqa: BLE001 — boundary trap
                logger.exception("Per-doc graph pipeline failed; isolating")
                results.append({"error": f"{type(exc).__name__}: {exc}"})
        return results

    # =====================================================
    # SHARED IMPL
    # =====================================================

    def _run_with_doc(self, text: str, doc) -> Dict[str, Any]:

        entity_graph: Optional[Dict[str, Dict[str, float]]] = None
        narrative_graph: Optional[Dict[str, Dict[str, float]]] = None
        temporal_features: Optional[Dict[str, float]] = None

        # G-T1 / G-T2: per-mention character spans for entity & narrative
        # nodes, surfaced through the result dict so the API / explainer
        # layer can map node IDs back to highlightable text regions.
        entity_spans: List[Dict[str, Any]] = []
        narrative_spans: List[Dict[str, Any]] = []
        narrative_tokenizer: Optional[str] = None

        # -------------------------------------------
        # ENTITY GRAPH  (already parsed) — G-T1: spans surfaced
        # -------------------------------------------
        if self.entity_graph_builder is not None and doc is not None:
            entity_graph, entity_spans = self._entity_graph_from_doc(doc)

        # -------------------------------------------
        # NARRATIVE GRAPH — G-S1 / G-T2: spaCy-aligned, span-aware
        # G-P8: pass the already-parsed ``Doc`` so the narrative
        # builder doesn't run the spaCy parser a second time on the
        # same text. Falls back to self-parse when no doc is shared
        # (direct callers, tests).
        # -------------------------------------------
        if self.narrative_graph_builder is not None:
            narrative_payload = (
                self.narrative_graph_builder.build_graph_with_spans(
                    text, doc=doc
                )
            )
            narrative_graph = narrative_payload["graph"]
            narrative_spans = narrative_payload.get("spans", [])
            narrative_tokenizer = narrative_payload.get("tokenizer")

            # G-P1: canonicalize once at the top so every downstream
            # consumer (analyzer, embedding, explainer) skips repeating
            # the symmetrise / normalise pass.
            if narrative_graph:
                narrative_graph = canonicalize_weighted(narrative_graph)

        # -------------------------------------------
        # TEMPORAL GRAPH
        # G-T4: pass the shared ``Doc`` so the analyzer extracts
        # entity ids from spaCy NEs / noun-chunks (matching the
        # entity-graph node space) instead of regex word tokens.
        # -------------------------------------------
        if self.temporal_analyzer is not None:
            temporal_features = self.temporal_analyzer.analyze(
                text, doc=doc
            ).to_dict()

        # -------------------------------------------
        # GRAPH METRICS
        # -------------------------------------------
        entity_metrics = (
            self.graph_analyzer.analyze(entity_graph).to_dict()
            if entity_graph
            else {}
        )

        narrative_metrics = (
            self.graph_analyzer.analyze(narrative_graph).to_dict()
            if narrative_graph
            else {}
        )

        # -------------------------------------------
        # GRAPH FEATURES — G-R2: pass through pre-computed metrics so
        # ``extract_from_graphs`` does not run ``GraphAnalyzer.analyze``
        # a second time on the same graph.
        # -------------------------------------------
        features = self.graph_feature_extractor.extract_from_graphs(
            entity_graph=entity_graph,
            narrative_graph=narrative_graph,
            entity_metrics=entity_metrics,
            narrative_metrics=narrative_metrics,
        )

        # merge temporal features
        if temporal_features:
            features.update(temporal_features)

        # -------------------------------------------
        # GRAPH EXPLANATION  (G-C3)
        # -------------------------------------------
        explanation = None

        if self.graph_explainer is not None:
            try:
                explanation = self.graph_explainer.explain(
                    entity_graph=entity_graph,
                    narrative_graph=narrative_graph,
                    temporal_features=temporal_features,
                )
            except Exception:
                logger.exception("Graph explanation failed; continuing without it")
                explanation = None

        # -------------------------------------------
        # FEATURE VECTOR
        # -------------------------------------------
        vector = None

        if self.config.return_vector:
            try:
                vector = self.graph_feature_extractor.extract_feature_vector_from_features(
                    features
                )
            except Exception as exc:
                logger.exception("Vector creation failed")
                raise RuntimeError("Graph vector failed") from exc

        # -------------------------------------------
        # ANALYSIS MODULES
        # -------------------------------------------
        analysis_modules = None

        if self.analysis_runner is not None:
            analysis_modules = self.analysis_runner.analyze_text(text)

        # -------------------------------------------
        # FINAL WRAP (GraphOutput)  — G-C2
        # -------------------------------------------
        explanation_dict = explanation.to_dict() if explanation is not None else None

        try:
            graph_output = GraphOutput(
                # G-C2: previously passed raw weighted dicts as
                # ``entity_graph=``/``narrative_graph=`` and the schema
                # expected ``GraphStructure(nodes, edges)`` — pydantic
                # raised ``ValidationError`` on every request. Now
                # adapted at the boundary.
                entity_graph=_to_graph_structure(entity_graph),
                narrative_graph=_to_graph_structure(narrative_graph),
                temporal_features=temporal_features,
                entity_metrics=entity_metrics or None,
                narrative_metrics=narrative_metrics or None,
                # G-C6: spans + tokenizer now flow through the typed
                # envelope as well as the raw result dict, so consumers
                # that type their input as ``GraphOutput`` can reach
                # them without dropping back to untyped ``Dict``.
                entity_spans=entity_spans or None,
                narrative_spans=narrative_spans or None,
                narrative_tokenizer=narrative_tokenizer,
                features=features,
                embeddings=None,
                explanation=explanation_dict,
            )
        except Exception:
            # Defensive: never let a schema mismatch take down the
            # entire request — the rest of the result dict is still
            # useful even if the typed envelope failed to build.
            logger.exception("GraphOutput construction failed; returning raw dicts")
            graph_output = None

        result: Dict[str, Any] = {
            "graph_output": graph_output,
            "graph_features": features,
            # G-C4: previously the consumer
            # (`feature_pipeline._merge_graph_features`) read
            # ``entity_graph_metrics`` / ``narrative_graph_metrics`` but
            # the producer never emitted those keys, so per-graph metrics
            # were silently dropped before reaching the model. Now
            # surfaced as first-class result keys with the names the
            # consumer already expects.
            "entity_graph_metrics": entity_metrics,
            "narrative_graph_metrics": narrative_metrics,
            # G-T1 / G-T2: per-mention character spans so the API /
            # explainer can highlight node IDs back into the source text.
            "entity_spans": entity_spans,
            "narrative_spans": narrative_spans,
            "narrative_tokenizer": narrative_tokenizer,
        }

        if vector is not None:
            result["graph_feature_vector"] = vector

        if analysis_modules is not None:
            result["analysis_modules"] = analysis_modules

        if explanation is not None:
            result["graph_explanation"] = explanation_dict

        logger.debug(
            "GraphPipeline completed: %d features",
            len(features),
        )

        return result


# =========================================================
# SINGLETON  (G-R1)
# =========================================================
#
# ``GraphPipeline`` instantiation pulls in 6 builders + 15 analysis
# modules (``AnalysisIntegrationRunner``). The audit found 7 callsites
# across the codebase, each holding its own copy — same parsers, same
# spaCy reference, same module registry. Switching the default to a
# singleton collapses that to one per process while still allowing
# tests / advanced callers to inject their own ``GraphPipeline`` for
# isolation. Reset via ``reset_default_pipeline()`` between tests.

_DEFAULT_PIPELINE: Optional[GraphPipeline] = None
# G-T3: previously the comment claimed the GIL made the
# ``if _DEFAULT_PIPELINE is None`` + assignment effectively atomic.
# It does for the bytecode of those two ops in isolation, but two
# threads can both pass the None-check before either assigns — the
# loser's ``GraphPipeline()`` (6 builders + 15 analysis modules) is
# constructed and immediately GC'd. Not a correctness bug — both
# pipelines are functionally equivalent — but a real perf cliff at
# FastAPI worker startup under concurrent first requests. Double-
# checked locking with ``threading.Lock`` fixes it cheaply.
_DEFAULT_PIPELINE_LOCK = threading.Lock()


def get_default_pipeline() -> GraphPipeline:
    """Return the process-wide ``GraphPipeline`` singleton.

    Lazily constructed on first call so import order does not force
    ``AnalysisIntegrationRunner`` to load before its dependencies are
    ready. Thread-safe via double-checked locking.
    """
    global _DEFAULT_PIPELINE
    # Fast path: already initialised — avoid the lock entirely.
    if _DEFAULT_PIPELINE is not None:
        return _DEFAULT_PIPELINE
    with _DEFAULT_PIPELINE_LOCK:
        # Re-check under the lock so we only ever construct once.
        if _DEFAULT_PIPELINE is None:
            _DEFAULT_PIPELINE = GraphPipeline()
    return _DEFAULT_PIPELINE


def reset_default_pipeline() -> None:
    """Drop the cached singleton — used by tests."""
    global _DEFAULT_PIPELINE
    with _DEFAULT_PIPELINE_LOCK:
        _DEFAULT_PIPELINE = None
