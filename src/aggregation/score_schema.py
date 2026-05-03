from __future__ import annotations

from typing import Dict, Any, Optional, Literal, List
import math
from pydantic import BaseModel, field_validator, ConfigDict


_ALLOWED_LEVELS = {"LOW", "MEDIUM", "HIGH"}


# =========================================================
# BASE TASK SCORE (UPGRADED)
# =========================================================

class TaskScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    score: float
    confidence: Optional[float] = None

    # 🔥 NEW
    probabilities: Optional[List[float]] = None
    entropy: Optional[float] = None

    @field_validator("score", "confidence", "entropy")
    @classmethod
    def validate_numeric(cls, v):
        if v is None:
            return v

        if isinstance(v, bool):
            raise TypeError("Must be numeric")

        fv = float(v)

        if not math.isfinite(fv):
            raise ValueError("Must be finite")

        return fv

    @field_validator("score", "confidence")
    @classmethod
    def validate_range(cls, v):
        if v is None:
            return v
        if not (0.0 <= v <= 1.0):
            raise ValueError("Must be in [0,1]")
        return v


# =========================================================
# SCORES (UPGRADED)
# =========================================================

class TruthLensScoreModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tasks: Dict[str, TaskScore]

    manipulation_risk: float
    credibility_score: float
    final_score: float

    # 🔥 NEW
    uncertainty_summary: Optional[Dict[str, float]] = None

    # Aggregation Engine v2 — neural + hybrid outputs
    neural_credibility_score: Optional[float] = None
    hybrid_alpha: Optional[float] = None

    @field_validator("tasks")
    @classmethod
    def validate_tasks(cls, v):
        return v if v else {}

    @field_validator("neural_credibility_score", "hybrid_alpha")
    @classmethod
    def validate_neural(cls, v):
        if v is None:
            return v
        import math
        fv = float(v)
        if not math.isfinite(fv):
            raise ValueError("Must be finite")
        return fv


# =========================================================
# RISK MODEL (UPGRADED)
# =========================================================

class RiskValue(BaseModel):
    level: Literal["LOW", "MEDIUM", "HIGH"]
    score: Optional[float] = None  # 🔥 continuous score


class TruthLensRiskModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    manipulation_risk: Optional[RiskValue] = None
    credibility_level: Optional[RiskValue] = None
    overall_truthlens_rating: Optional[RiskValue] = None


# =========================================================
# EXPLAINABILITY (UPGRADED)
# =========================================================

class TokenAttribution(BaseModel):
    token: str
    importance: float
    contribution: float
    direction: Literal["positive", "negative"]


class ExplanationSection(BaseModel):
    method: Literal["integrated_gradients", "shap", "attention"]

    top_features: List[str]
    attributions: List[TokenAttribution]

    # 🔥 NEW
    section_score: Optional[float] = None


class ExplanationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sections: Dict[str, ExplanationSection] = {}


# =========================================================
# METADATA (NEW 🔥)
# =========================================================

class SystemMetadata(BaseModel):
    device: Optional[str] = None
    latency_ms: Optional[float] = None
    request_id: Optional[str] = None

    calibration_method: Optional[str] = None
    normalization_method: Optional[str] = None


# =========================================================
# FINAL OUTPUT (UPGRADED)
# =========================================================

class TruthLensAggregationOutputModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str

    # 🔥 VERSIONING
    model_version: str
    aggregation_version: Optional[str] = None

    # -------------------------
    # CORE OUTPUTS
    # -------------------------
    scores: TruthLensScoreModel
    raw_scores: Dict[str, float]

    risks: TruthLensRiskModel
    explanations: ExplanationModel

    # -------------------------
    # SYSTEM INFO
    # -------------------------
    metadata: Optional[SystemMetadata] = None

    # -------------------------
    # EXTENSIONS
    # -------------------------
    analysis_modules: Dict[str, Any]

    @field_validator("analysis_modules")
    @classmethod
    def validate_modules(cls, v):
        if not isinstance(v, dict):
            raise TypeError("Must be dict")
        return v