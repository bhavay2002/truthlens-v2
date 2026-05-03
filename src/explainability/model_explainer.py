from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional

from src.explainability.orchestrator import (
    ExplainabilityConfig,
    get_default_orchestrator,
)

logger = logging.getLogger(__name__)

PredictionFn = Callable[[str], Dict[str, Any]]

PIPELINE_VERSION = "v2"


# =========================================================
# FULL EXPLAIN (COMPAT + UPGRADED)
# =========================================================

def explain_prediction_full(
    text: str,
    predict_fn: PredictionFn,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    use_lime: bool = True,
    use_shap: bool = True,
) -> Dict[str, Any]:

    start_time = time.time()

    config = ExplainabilityConfig(
        enabled=True,
        use_lime=use_lime,
        use_shap=use_shap,
        use_bias_emotion=True,
        use_attention_rollout=True,      # 🔥 upgraded
        use_propaganda_explainer=False,
        use_aggregation=True,            # 🔥 upgraded
        use_consistency=True,            # 🔥 upgraded
        cache_enabled=True,              # 🔥 upgraded
    )

    orchestrator = get_default_orchestrator(config)

    prediction = predict_fn(text)

    try:
        result = orchestrator.explain(
            text=text,
            predict_fn=predict_fn,
            model=model,
            tokenizer=tokenizer,
        )
    except Exception as e:
        logger.exception("Explainability failed")
        result = {"error": str(e)}

    latency = (time.time() - start_time) * 1000

    return {
        # -------------------------
        # core outputs
        # -------------------------
        "prediction": prediction,
        "bias_explanation": result.get("bias_explanation"),
        "emotion_explanation": result.get("emotion_explanation"),
        "shap_explanation": result.get("shap_explanation"),
        "lime_explanation": result.get("lime_explanation"),

        # -------------------------
        # 🔥 NEW unified outputs
        # -------------------------
        "aggregated_explanation": result.get("aggregated_explanation"),
        "metrics": result.get("explanation_metrics"),
        "consistency": result.get("consistency_metrics"),

        # -------------------------
        # 🔥 metadata
        # -------------------------
        "metadata": {
            "pipeline_version": PIPELINE_VERSION,
            "latency_ms": latency,
            "modules_enabled": {
                "lime": use_lime,
                "shap": use_shap,
                "aggregation": True,
                "consistency": True,
                "attention": True,
            },
        },
    }


# =========================================================
# FAST EXPLAIN (LOW LATENCY)
# =========================================================

def explain_fast(
    text: str,
    predict_fn: PredictionFn,
) -> Dict[str, Any]:

    start_time = time.time()

    config = ExplainabilityConfig(
        enabled=True,
        use_lime=True,
        use_shap=False,
        use_bias_emotion=False,
        use_attention_rollout=False,
        use_propaganda_explainer=False,
        use_aggregation=False,
        use_consistency=False,
        cache_enabled=True,  # 🔥 keep cache
    )

    orchestrator = get_default_orchestrator(config)

    try:
        result = orchestrator.explain_fast(
            text=text,
            predict_fn=predict_fn,
        )
    except Exception as e:
        logger.exception("Fast explain failed")
        result = {"error": str(e)}

    latency = (time.time() - start_time) * 1000

    return {
        "result": result,
        "metadata": {
            "pipeline_version": PIPELINE_VERSION,
            "latency_ms": latency,
            "mode": "fast",
        },
    }