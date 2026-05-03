from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Callable, Optional

import numpy as np
import torch

from src.features.base.base_feature import FeatureContext
from src.features.pipelines.feature_pipeline import FeaturePipeline

from src.graph.entity_graph import EntityGraphBuilder
from src.graph.graph_analysis import GraphAnalyzer
from src.graph.narrative_graph_builder import NarrativeGraphBuilder
from src.graph.graph_pipeline import GraphPipeline, get_default_pipeline

from src.analysis.bias_profile_builder import BiasProfileBuilder
from src.analysis.integration_runner import AnalysisIntegrationRunner

from src.aggregation.aggregation_pipeline import AggregationPipeline
from src.aggregation.truthlens_score_calculator import TruthLensScoreCalculator

from src.inference.prediction_service import PredictionService

from src.inference.report_generator import ReportGenerator
from src.explainability.explanation_report_generator import ExplanationReportGenerator

from src.utils import ensure_non_empty_text

logger = logging.getLogger(__name__)


# =========================================================
# HELPER
# =========================================================

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# =========================================================
# MAIN ANALYZER
# =========================================================

@dataclass
class ArticleAnalyzer:

    # Core pipelines
    feature_pipeline: FeaturePipeline
    entity_graph_builder: EntityGraphBuilder
    graph_analyzer: GraphAnalyzer
    profile_builder: BiasProfileBuilder
    score_calculator: TruthLensScoreCalculator

    # Optional modules
    narrative_graph_builder: Optional[NarrativeGraphBuilder] = None
    graph_pipeline: Optional[GraphPipeline] = None
    analysis_runner: Optional[AnalysisIntegrationRunner] = None
    aggregation_pipeline: Optional[AggregationPipeline] = None

    # 🔥 NEW unified inference
    prediction_service: Optional[PredictionService] = None

    # Reporting
    report_generator: Optional[ReportGenerator] = None
    explanation_report_generator: Optional[ExplanationReportGenerator] = None

    # Optional external predict_fn
    predict_fn: Optional[Callable[[str], Dict[str, Any]]] = None

    # =====================================================
    # INIT
    # =====================================================

    def __post_init__(self):

        self.feature_pipeline.initialize()

        self.narrative_graph_builder = self.narrative_graph_builder or NarrativeGraphBuilder()
        # G-R1: callers can still inject their own ``GraphPipeline``;
        # the default falls through to the process-wide singleton.
        self.graph_pipeline = self.graph_pipeline or get_default_pipeline()
        self.analysis_runner = self.analysis_runner or AnalysisIntegrationRunner()
        self.aggregation_pipeline = self.aggregation_pipeline or AggregationPipeline()

        self.report_generator = self.report_generator or ReportGenerator()
        self.explanation_report_generator = self.explanation_report_generator or ExplanationReportGenerator()

        # 🔥 NEW: full inference system — only build if engine is available.
        # PredictionService requires an InferenceEngine; skip silently when
        # no model is present so the rest of the analysis still works.
        if self.prediction_service is None:
            try:
                from src.inference.inference_engine import InferenceEngine, InferenceConfig
                from src.utils.settings import load_settings
                _settings = load_settings()
                self.prediction_service = PredictionService(
                    engine=InferenceEngine(
                        InferenceConfig(
                            model_path=str(_settings.model.path),
                            device="auto",
                            enable_full_pipeline=False,
                        )
                    )
                )
            except Exception:
                self.prediction_service = None

    # =====================================================
    # FEATURE SPLIT
    # =====================================================

    def _extract_feature_sections(self, features: Dict[str, float]):

        sections = {
            "bias": {},
            "emotion": {},
            "narrative": {},
            "discourse": {},
        }

        for k, v in features.items():
            if k.startswith("bias_"):
                sections["bias"][k] = v
            elif k.startswith("emotion_"):
                sections["emotion"][k] = v
            elif k.startswith("narrative_"):
                sections["narrative"][k] = v
            elif k.startswith("discourse_"):
                sections["discourse"][k] = v

        return sections

    # =====================================================
    # 🔥 NEW PREDICTION (FULL SYSTEM)
    # =====================================================

    def _run_prediction(self, text: str) -> Dict[str, Any]:

        if not self.prediction_service:
            return {}

        try:
            # REC-1: ``PredictionService.predict`` returns the basic
            # ``{label, confidence, fake_probability}`` blob (and is
            # cache-backed). It does NOT contain ``predictions`` /
            # ``probabilities`` / ``logits`` arrays — those keys read as
            # ``None`` previously, polluting the report. Surface the
            # actual fields and keep the raw blob under ``raw_output``
            # for downstream consumers.
            result = self.prediction_service.predict(text)

            return {
                "label": result.get("label"),
                "confidence": result.get("confidence"),
                "fake_probability": result.get("fake_probability"),
                "raw_output": result,
            }

        except Exception as e:
            logger.warning("PredictionService failed: %s", e)
            return {}

    # =====================================================
    # MAIN ANALYSIS
    # =====================================================

    def analyze(self, text: str) -> Dict[str, Any]:

        ensure_non_empty_text(text, name="text")

        context = FeatureContext(text=text)

        # ---------------- FEATURES ----------------
        fused_features = self.feature_pipeline.extract(context)
        feature_sections = self._extract_feature_sections(fused_features)

        # ---------------- GRAPH ----------------
        entity_graph = self.entity_graph_builder.build_graph(text)
        graph_features = self.entity_graph_builder.extract_graph_features(entity_graph)
        graph_metrics = self.graph_analyzer.analyze(entity_graph)

        # ---------------- ANALYSIS ----------------
        analysis_modules = self.analysis_runner.analyze_text(text)

        # ---------------- PROFILE ----------------
        profile = self.profile_builder.build_profile(
            bias_features=feature_sections["bias"],
            emotion_features=feature_sections["emotion"],
            narrative_features=feature_sections["narrative"],
            discourse_features=feature_sections["discourse"],
            ideology_predictions={},
        )

        # ---------------- AGGREGATION ----------------
        aggregation_output = self.aggregation_pipeline.run(
            profile,
            text=text,
            analysis_modules=analysis_modules,
        )

        scores = aggregation_output.get("raw_scores")
        if not scores:
            scores = self.score_calculator.compute_scores(profile)

        # ---------------- 🔥 FULL SYSTEM PREDICTION ----------------
        prediction_output = self._run_prediction(text)

        raw_pred = prediction_output.get("raw_output", {})

        # ---------------- FINAL REPORT ----------------
        # REC-1: the previous report extracted ``graph`` /
        # ``graph_explanation`` / ``drift`` / ``monitoring`` from
        # ``raw_pred``, but ``PredictionService.predict`` never produces
        # those keys — they were always ``None``. The graph features
        # already live under ``graph_features``/``entity_graph`` above;
        # drift and monitoring are surfaced by their own services.
        report = {

            "text": text,

            # Feature blocks
            "bias_features": feature_sections["bias"],
            "emotion_features": feature_sections["emotion"],
            "narrative_features": feature_sections["narrative"],
            "discourse_features": feature_sections["discourse"],

            # Graph
            "graph_features": {
                **graph_features,
                **graph_metrics,
            },
            "entity_graph": entity_graph,

            # Analysis
            "analysis_modules": analysis_modules,
            "profile": profile,

            # Scores
            "scores": scores,
            "aggregation": aggregation_output,

            # Model outputs (basic prediction blob)
            "label": prediction_output.get("label"),
            "confidence": prediction_output.get("confidence"),
            "fake_probability": prediction_output.get("fake_probability"),
            "prediction_raw": raw_pred,
        }

        logger.info("Article analysis complete")

        return report