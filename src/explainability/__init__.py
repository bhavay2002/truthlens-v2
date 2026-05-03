"""
src.explainability
==================
Explainability sub-package for TruthLens AI.

Public API
----------
``ExplainabilityOrchestrator`` / ``ExplainabilityConfig``
    Owners of the full explainability lifecycle (SHAP, LIME, bias,
    emotion, attention rollout, propaganda, aggregation, consistency).

``run_explainability_pipeline`` / ``explain_prediction_full`` / ``explain_fast``
    Convenience entry points that wrap the orchestrator and return a
    fully-validated ``ExplainabilityResult``.

``ExplainabilityResult`` / ``AggregatedExplanation`` /
``ExplanationOutput`` / ``TokenImportance``
    Pydantic schemas for the canonical explainability output. Defined
    in ``src.explainability.common_schema`` (CRIT-6 / CRIT-7).
"""

from src.explainability.common_schema import (
    AggregatedExplanation,
    ExplainabilityResult,
    ExplanationOutput,
    TokenImportance,
)
from src.explainability.orchestrator import (
    ExplainabilityConfig,
    ExplainabilityOrchestrator,
    get_default_orchestrator,
)
from src.explainability.explainability_pipeline import (
    explain_fast,
    explain_prediction_full,
    run_explainability_pipeline,
)

__all__ = [
    "ExplainabilityConfig",
    "ExplainabilityOrchestrator",
    "get_default_orchestrator",
    "run_explainability_pipeline",
    "explain_prediction_full",
    "explain_fast",
    "ExplainabilityResult",
    "AggregatedExplanation",
    "ExplanationOutput",
    "TokenImportance",
]
