"""Aggregation layer — see `aggregation_pipeline.AggregationPipeline`.

UNUSED-AG: previously empty. Re-exporting the public surface so
downstream callers can `from src.aggregation import AggregationPipeline`
without needing to know which sub-module each symbol lives in.
"""

from .aggregation_config import (
    AggregationConfig,
    AttributionConfig,
    CalibrationConfig,
    DriftConfig,
    FusionConfig,
    MonitoringConfig,
    NormalizationConfig,
    RiskConfig,
    UncertaintyConfig,
    WeightConfig,
    WEIGHT_GROUPS,
    TASK_TO_GROUP,
    SCALAR_WEIGHT_KEYS,
    load_aggregation_config,
)
from .aggregation_metrics import AggregationMetrics
from .aggregation_pipeline import AggregationPipeline
from .aggregation_validator import AggregationValidator
from .feature_mapper import FeatureMapper, TaskSignal
from .risk_assessment import (
    RiskConfig as RuntimeRiskConfig,
    assess_risk_levels,
    assess_truthlens_risks,
    assess_batch,
    from_pydantic_config as risk_from_pydantic_config,
)
from .score_explainer import ScoreExplainer
from .truthlens_score_calculator import TruthLensScoreCalculator
from .weight_manager import DEFAULT_WEIGHTS, WeightManager

__all__ = [
    "AggregationPipeline",
    "AggregationConfig",
    "AggregationMetrics",
    "AggregationValidator",
    "AttributionConfig",
    "CalibrationConfig",
    "DriftConfig",
    "FusionConfig",
    "MonitoringConfig",
    "NormalizationConfig",
    "RiskConfig",
    "RuntimeRiskConfig",
    "UncertaintyConfig",
    "WeightConfig",
    "WEIGHT_GROUPS",
    "TASK_TO_GROUP",
    "SCALAR_WEIGHT_KEYS",
    "FeatureMapper",
    "TaskSignal",
    "ScoreExplainer",
    "TruthLensScoreCalculator",
    "WeightManager",
    "DEFAULT_WEIGHTS",
    "load_aggregation_config",
    "assess_risk_levels",
    "assess_truthlens_risks",
    "assess_batch",
    "risk_from_pydantic_config",
]
