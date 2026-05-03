from __future__ import annotations

from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
import math
import numpy as np

EPS = 1e-12


# =========================================================
# BASE TOKEN UNIT
# =========================================================

class TokenImportance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    token: str
    # CRIT-5: drop ge/le bounds — explainers may legitimately produce signed
    # attributions or values outside [0,1]. Only finiteness is enforced.
    importance: float

    @field_validator("token")
    @classmethod
    def validate_token(cls, v):
        if not v.strip():
            raise ValueError("token must be non-empty")
        return v

    @field_validator("importance")
    @classmethod
    def validate_importance(cls, v):
        if not math.isfinite(v):
            raise ValueError("importance must be finite")
        return float(v)


# =========================================================
# METHOD OUTPUT
# =========================================================

class ExplanationOutput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    method: Literal[
        "shap", "lime", "attention",
        "integrated_gradients", "propaganda", "custom"
    ]

    tokens: List[str]
    importance: List[float]
    structured: List[TokenImportance]

    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    entropy: Optional[float] = None
    raw: Optional[Any] = None

    # CRIT-9: distinguish faithful (model-derived) explanations from
    # heuristic / lexicon-only signals. Defaults to True; heuristic
    # explainers (e.g. propaganda) must explicitly set it False so the
    # aggregator can gate inclusion by policy.
    faithful: bool = True

    # -------------------------
    # VALIDATION (CRIT-5: pass-through, no re-normalization)
    # -------------------------

    @field_validator("importance")
    @classmethod
    def validate_importance_finite(cls, v):
        """CRIT-5: pass-through validator. Only checks finiteness — does NOT
        re-normalize. Callers are responsible for calibration so that
        ``importance[i]`` and ``structured[i].importance`` remain in sync.
        """
        if not v:
            return v
        arr = np.asarray(v, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError("importance must be finite")
        return [float(x) for x in arr]

    @field_validator("structured")
    @classmethod
    def validate_structured(cls, v, info):
        tokens = info.data.get("tokens", [])
        importance = info.data.get("importance", [])

        if len(tokens) != len(importance):
            raise ValueError("tokens and importance must align")

        if len(v) != len(tokens):
            raise ValueError("structured must align with tokens")

        # CRIT-5: enforce that the flat ``importance`` list and the
        # per-token ``structured`` list agree. Drift between the two
        # representations was the root cause of cross-module disagreements.
        for s, i in zip(v, importance):
            if not math.isfinite(s.importance):
                raise ValueError("structured importance must be finite")
            if abs(s.importance - float(i)) > 1e-6:
                raise ValueError(
                    "structured importance must equal flat importance "
                    f"(got {s.importance!r} vs {i!r})"
                )

        return v


# =========================================================
# AGGREGATED OUTPUT
# =========================================================

class AggregatedExplanation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tokens: List[str]
    final_token_importance: List[float]

    structured: List[TokenImportance]

    method_weights: Dict[str, float]

    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    agreement_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # CRIT-11 plumbing: the aggregator may carry the original text and
    # per-token character offsets so downstream faithfulness metrics can
    # ablate at the input-text level rather than re-tokenising a joined
    # string. Optional for backward compatibility.
    text: Optional[str] = None
    offsets: Optional[List[List[int]]] = None

    @field_validator("final_token_importance")
    @classmethod
    def validate_scores(cls, v):
        return v

    @field_validator("structured")
    @classmethod
    def validate_structured(cls, v, info):
        tokens = info.data.get("tokens", [])

        if len(v) != len(tokens):
            raise ValueError("structured must align with tokens")

        return v


# =========================================================
# METRICS
# =========================================================

class ConsistencyMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shap_vs_lime: Optional[float] = None
    shap_vs_attention: Optional[float] = None
    ig_vs_lime: Optional[float] = None
    ig_vs_attention: Optional[float] = None
    shap_vs_ig: Optional[float] = None

    overall_consistency: Optional[float] = None


class ExplanationMetricsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    faithfulness: float
    comprehensiveness: float
    sufficiency: float
    deletion_score: float
    insertion_score: float

    overall_score: Optional[float] = None


# =========================================================
# FINAL OUTPUT  (CRIT-6 / CRIT-7: single source of truth)
# =========================================================

class ExplainabilityResult(BaseModel):
    """Canonical explainability result.

    CRIT-6 / CRIT-7: this is the *single* ExplainabilityResult schema.
    ``src.explainability.explainability_pipeline.ExplainabilityResult``
    re-exports this class. ``model_explainer.py`` has been removed.
    """

    # ``extra="ignore"`` so callers can pass extra orchestrator fields
    # without exploding while still validating the documented contract.
    model_config = ConfigDict(extra="ignore")

    prediction: Dict[str, Any]

    shap_explanation: Optional[Any] = None
    lime_explanation: Optional[Any] = None
    attention_explanation: Optional[Any] = None
    propaganda_explanation: Optional[Any] = None

    # heuristic / model-side explainers
    bias_explanation: Optional[Any] = None
    emotion_explanation: Optional[Any] = None

    aggregated_explanation: Optional[Any] = None

    consistency_metrics: Optional[Dict[str, float]] = None
    explanation_metrics: Optional[Dict[str, Any]] = None

    monitoring: Optional[Dict[str, Any]] = None

    explanation_quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # FAITH-6: surface which sub-explainers failed so consumers can react.
    module_failures: List[str] = Field(default_factory=list)

    metadata: Optional[Dict[str, Any]] = None
