"""
Package: src.inference
Description:
    Public API for the TruthLens inference subsystem.

    Exposes the core classes used for running model inference,
    caching results, logging events, formatting outputs, and
    generating structured analysis reports.
"""

from src.inference.inference_cache import InferenceCache, InferenceCacheConfig
from src.inference.inference_engine import (
    InferenceConfig as EngineInferenceConfig,
    InferenceEngine,
)
from src.inference.inference_logger import InferenceLogEntry, InferenceLogger
from src.inference.feature_preparer import FeaturePreparer, FeaturePreparationConfig
from src.inference.model_loader import ModelArtifacts, ModelLoader
from src.inference.inference_pipeline import PredictionPipeline, PredictionPipelineConfig
from src.inference.report_generator import ReportConfig, ReportGenerator
from src.inference.result_formatter import ResultFormatter

__all__ = [
    "InferenceCache",
    "InferenceCacheConfig",
    "EngineInferenceConfig",
    "InferenceEngine",
    "InferenceLogEntry",
    "InferenceLogger",
    "FeaturePreparer",
    "FeaturePreparationConfig",
    "ModelArtifacts",
    "ModelLoader",
    "PredictionPipeline",
    "PredictionPipelineConfig",
    "ReportConfig",
    "ReportGenerator",
    "ResultFormatter",
]
