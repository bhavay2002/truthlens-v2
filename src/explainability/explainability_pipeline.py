"""src/explainability/explainability_pipeline.py

Unified explainability pipeline.

Audit fixes
-----------
* **CRIT-6 / CRIT-7**: ``ExplainabilityResult`` is no longer redefined
  here. The single source of truth lives in
  ``src.explainability.common_schema`` and is re-exported below for
  backward compatibility. The legacy ``model_explainer.py`` shim has
  been removed.
* **PERF-6**: ``run_explainability_pipeline`` now uses
  ``get_default_orchestrator`` instead of instantiating a fresh
  orchestrator per article.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from src.explainability.common_schema import ExplainabilityResult  # CRIT-6/7
from src.explainability.orchestrator import (
    ExplainabilityConfig,
    get_default_orchestrator,
)

logger = logging.getLogger(__name__)

PredictionFn = Callable[[str], Dict[str, Any]]

__all__ = [
    "ExplainabilityResult",
    "ExplainabilityConfig",
    "run_explainability_pipeline",
    "explain_prediction_full",
    "explain_fast",
]


# =========================================================
# CORE PIPELINE
# =========================================================

def run_explainability_pipeline(
    text: str,
    predict_fn: PredictionFn,
    *,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    tokens: Optional[list[str]] = None,
    attentions: Optional[Any] = None,
    config: Optional[ExplainabilityConfig] = None,
) -> ExplainabilityResult:

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be non-empty")

    config = config or ExplainabilityConfig()
    orchestrator = get_default_orchestrator(config)

    logger.info("Running explainability pipeline")

    prediction = predict_fn(text)

    explanation = orchestrator.explain(
        text=text,
        predict_fn=predict_fn,
        model=model,
        tokenizer=tokenizer,
        tokens=tokens,
        attentions=attentions,
    )

    return ExplainabilityResult(
        prediction=prediction,
        shap_explanation=explanation.get("shap_explanation"),
        lime_explanation=explanation.get("lime_explanation"),
        attention_explanation=explanation.get("attention_explanation"),
        propaganda_explanation=explanation.get("propaganda_explanation"),
        bias_explanation=explanation.get("bias_explanation"),
        emotion_explanation=explanation.get("emotion_explanation"),
        aggregated_explanation=explanation.get("aggregated_explanation"),
        consistency_metrics=explanation.get("consistency_metrics"),
        explanation_metrics=explanation.get("explanation_metrics"),
        monitoring=explanation.get("monitoring"),
        explanation_quality_score=explanation.get("explanation_quality_score"),
        module_failures=list(explanation.get("module_failures") or []),
        metadata=explanation.get("metadata"),
    )


# =========================================================
# BACKWARD COMPAT WRAPPERS
# =========================================================

def explain_prediction_full(
    text: str,
    predict_fn: PredictionFn,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    use_lime: bool = True,
    use_shap: bool = True,
) -> ExplainabilityResult:

    config = ExplainabilityConfig(
        enabled=True,
        use_lime=use_lime,
        use_shap=use_shap,
        use_bias_emotion=True,
        use_attention_rollout=False,
        use_aggregation=True,
        use_consistency=True,
        use_explanation_metrics=True,
        cache_enabled=False,
    )

    return run_explainability_pipeline(
        text=text,
        predict_fn=predict_fn,
        model=model,
        tokenizer=tokenizer,
        config=config,
    )


def explain_fast(
    text: str,
    predict_fn: PredictionFn,
) -> ExplainabilityResult:

    config = ExplainabilityConfig(
        enabled=True,
        use_lime=True,
        use_shap=False,
        use_bias_emotion=False,
        use_attention_rollout=False,
        use_aggregation=False,
        use_consistency=False,
        use_explanation_metrics=False,
        cache_enabled=False,
    )

    return run_explainability_pipeline(
        text=text,
        predict_fn=predict_fn,
        config=config,
    )
