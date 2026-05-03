from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Literal

import yaml
from pydantic import BaseModel, Field, ConfigDict, field_validator


logger = logging.getLogger(__name__)


# =========================================================
# CFG-AG-1: single source of truth for weight grouping.
#
# Both `weight_manager.py` and `truthlens_score_calculator.py` used to
# define their own private copies of these mappings. Any drift between
# the two definitions silently broke per-group renormalisation. They
# now live here and are re-exported from `weight_manager` for back-
# compat with existing imports.
# =========================================================

WEIGHT_GROUPS: Dict[str, tuple] = {
    "manipulation": (
        "bias",
        "emotion",
        "narrative",
        "analysis_influence_manipulation",
    ),
    "credibility": (
        "discourse",
        "graph",
        "analysis_influence_credibility",
        # TST-AG-WGT: `credibility_bias_penalty` participates in the
        # credibility group normalisation so that the four credibility
        # weights sum to 1.0, matching the test contract in
        # `test_weight_manager.py::test_grouped_normalization_sums_each_group_to_one`.
        # The penalty value stays in [0, 1] after normalisation (each
        # group member is ≤ 1 when the sum is 1), so the calculator's
        # `np.clip` on the penalty term remains a no-op in practice.
        "credibility_bias_penalty",
    ),
    "final": (
        "final_credibility",
        "final_manipulation",
        "final_ideology",
    ),
}


TASK_TO_GROUP: Dict[str, str] = {
    "bias":            "manipulation",
    "emotion":         "manipulation",
    "narrative":       "manipulation",
    "narrative_frame": "manipulation",
    "propaganda":      "manipulation",
    "discourse":       "credibility",
    "graph":           "credibility",
    "argument":        "credibility",
    "analysis":        "credibility",
    "ideology":        "final",
}


# Scalar multipliers that are NOT renormalised inside a weight group.
# credibility_bias_penalty was moved into WEIGHT_GROUPS["credibility"]
# (TST-AG-WGT), so this tuple is now empty. Kept for back-compat with
# any import of the symbol.
SCALAR_WEIGHT_KEYS: tuple = ()


# =========================================================
# NORMALIZATION CONFIG
# =========================================================

class NormalizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["minmax", "zscore", "robust", "quantile"] = "minmax"
    feature_range: tuple[float, float] = (0.0, 1.0)
    clip: bool = True

    per_feature: Optional[Dict[str, str]] = None

    @field_validator("feature_range")
    @classmethod
    def validate_range(cls, v):
        if len(v) != 2 or v[0] >= v[1]:
            raise ValueError("feature_range must be (min < max)")
        return v


# =========================================================
# CALIBRATION CONFIG (🔥 NEW)
# =========================================================

class CalibrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["temperature", "isotonic", "sigmoid", "none"] = "temperature"
    n_bins: int = 15
    enabled: bool = True


# =========================================================
# UNCERTAINTY CONFIG (🔥 NEW)
# =========================================================

class UncertaintyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enable_entropy: bool = True
    track_percentiles: bool = True

    p95_threshold: float = 0.8
    p99_threshold: float = 0.95

    @field_validator("p95_threshold", "p99_threshold")
    @classmethod
    def validate_range(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("Threshold must be in [0,1]")
        return v


# =========================================================
# WEIGHT CONFIG (UPGRADED)
# =========================================================

class WeightConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weights: Dict[str, float] = Field(default_factory=dict)
    version: str = "v2"

    allow_dynamic_adjustment: bool = True

    # 🔥 adaptive weighting
    use_confidence: bool = True
    use_entropy: bool = True
    use_explainability: bool = True

    smoothing: float = 0.1


# =========================================================
# RISK CONFIG (UPGRADED)
# =========================================================

class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    low_threshold: float = 0.3
    medium_threshold: float = 0.6

    uncertainty_penalty: float = 0.2

    @field_validator("medium_threshold")
    @classmethod
    def validate_thresholds(cls, v, info):
        low = info.data.get("low_threshold", 0.3)
        if v <= low:
            raise ValueError("medium_threshold must be greater than low_threshold")
        return v


# =========================================================
# ATTRIBUTION CONFIG (UPGRADED)
# =========================================================

class AttributionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal[
        "integrated_gradients",
        "shap",
        "attention"
    ] = "integrated_gradients"

    top_k: int = 5
    normalize: bool = True

    use_confidence_weighting: bool = True
    use_entropy_weighting: bool = True

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v):
        if v <= 0:
            raise ValueError("top_k must be positive")
        return v


# =========================================================
# FUSION CONFIG (WGT-AG-2)
#
# Previously the score calculator hardcoded a 0.1 cap for the graph
# cross-signal and a 0.5 blend for the explanation alignment. Both
# values are surfaced here so they can be tuned from config.yaml
# without code changes.
# =========================================================

class FusionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    graph_influence_cap: float = 0.1
    explanation_blend: float = 0.5

    @field_validator("graph_influence_cap", "explanation_blend")
    @classmethod
    def _in_unit(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("Must be in [0, 1]")
        return v


# =========================================================
# NEURAL AGGREGATOR CONFIG  (Aggregation Engine v2)
# =========================================================

class NeuralAggregatorConfig(BaseModel):
    """Configuration for the learned NeuralAggregator (spec §4).

    The neural path is **opt-in** (``enabled=False`` by default) so
    existing deployments remain unchanged until a trained checkpoint
    is available. When disabled, ``AggregationPipeline`` falls back
    to the rule-based ``TruthLensScoreCalculator`` transparently.

    Parameters
    ----------
    enabled:
        Activate the neural aggregator. Requires a valid
        ``checkpoint_path`` pointing to a file written by
        ``NeuralAggregator.save``.
    architecture:
        ``"mlp"`` — spec §4.1 (fast baseline).
        ``"attention"`` — spec §4.2 (recommended; provides learned
        per-feature importance as explanation output).
    hidden_dim:
        Width of the hidden layers inside the aggregator.
    dropout:
        Dropout probability applied in the aggregator trunk.
    checkpoint_path:
        Path to a ``.pt`` checkpoint. ``None`` means no checkpoint is
        loaded (useful for constructing an untrained module in tests
        or during training setup).
    alpha:
        Base neural blending weight for ``HybridScorer``.
        ``final = α * neural + (1−α) * rule``.
    dynamic_alpha:
        Use confidence-based dynamic α instead of the fixed ``alpha``.
    alpha_min:
        Minimum α in dynamic mode (fallback toward rule-based scoring
        when model confidence is low).
    alpha_max:
        Maximum α in dynamic mode.
    fallback_on_error:
        If ``True`` (default), runtime errors in the neural forward
        pass are caught and the result falls back to rule-only scoring
        rather than crashing the pipeline.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False

    architecture: Literal["mlp", "attention"] = "attention"
    hidden_dim: int = 256
    dropout: float = 0.2

    checkpoint_path: Optional[str] = None

    # Hybrid blending (§7)
    alpha: float = 0.7
    dynamic_alpha: bool = True
    alpha_min: float = 0.2
    alpha_max: float = 0.9

    fallback_on_error: bool = True

    @field_validator("dropout", "alpha", "alpha_min", "alpha_max")
    @classmethod
    def _in_unit(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("Must be in [0, 1]")
        return v

    @field_validator("hidden_dim")
    @classmethod
    def _positive(cls, v):
        if v < 1:
            raise ValueError("hidden_dim must be >= 1")
        return v


# =========================================================
# DRIFT CONFIG (🔥 NEW)
# =========================================================

class DriftConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True

    method: Literal["kl", "js", "psi"] = "js"
    threshold: float = 0.1


# =========================================================
# MONITORING CONFIG (🔥 NEW)
# =========================================================

class MonitoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True

    track_latency: bool = True
    track_confidence: bool = True
    track_entropy: bool = True


# =========================================================
# ROOT CONFIG (UPGRADED)
# =========================================================

class AggregationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalization: NormalizationConfig = NormalizationConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    uncertainty: UncertaintyConfig = UncertaintyConfig()
    weights: WeightConfig = WeightConfig()
    risk: RiskConfig = RiskConfig()
    attribution: AttributionConfig = AttributionConfig()

    drift: DriftConfig = DriftConfig()
    monitoring: MonitoringConfig = MonitoringConfig()

    # WGT-AG-2: fusion constants (graph cap, explanation blend)
    fusion: FusionConfig = FusionConfig()

    # Aggregation Engine v2 — learned + hybrid scoring (spec §4–7)
    neural: NeuralAggregatorConfig = NeuralAggregatorConfig()

    # CRIT-AG-12: explicit task -> task_type map. When empty the
    # pipeline lazy-loads from `config/config.yaml`. Set this here
    # only if you want to override or pre-populate without touching
    # the global app config.
    task_types: Dict[str, str] = Field(default_factory=dict)

    # PERF-AG-5: optional thread-pool fan-out for `run_batch`. Workers
    # > 1 only helps now that the per-article pipeline is stateless
    # (CRIT-AG-5/6 removed the per-article `fit_transform` mutations).
    batch_max_workers: int = 1

    # pipeline behavior
    strict_mode: bool = False
    enable_logging: bool = True
    enable_explanations: bool = True
    enable_risk: bool = True

    # versioning
    config_version: str = "v2"

    # CFG-3 (v13/v14 audit): single source of truth for the
    # model-version label that ends up on every aggregated result.
    # Previously hard-coded as the literal "truthlens-v2" inside
    # ``AggregationPipeline.run`` (line 359), which silently drifted
    # from the model the predictor actually loaded.
    model_version: str = "truthlens-v2"

    @field_validator("batch_max_workers")
    @classmethod
    def _positive_workers(cls, v):
        if v < 1:
            raise ValueError("batch_max_workers must be >= 1")
        return v


# =========================================================
# LOADER
# =========================================================

def load_aggregation_config(
    config_path: Optional[str | Path] = None,
    *,
    override: Optional[Dict[str, Any]] = None,
) -> AggregationConfig:

    config_data: Dict[str, Any] = {}

    if config_path:
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")

        try:
            with path.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except Exception as e:
            logger.exception("Failed to load config")
            raise RuntimeError("Config loading failed") from e

        # CFG-AG-6: when pointed at the global app config (which has an
        # `aggregation:` block alongside `tasks:`, `model:`, ...) pull
        # just the aggregation sub-tree. Standalone aggregation YAML
        # files (no `aggregation:` key) are loaded as-is.
        if isinstance(raw, dict) and isinstance(raw.get("aggregation"), dict):
            config_data = raw["aggregation"]
        else:
            config_data = raw

    if override:
        config_data.update(override)

    config = AggregationConfig(**config_data)

    logger.info(
        "[AggregationConfig] Loaded | version=%s",
        config.config_version,
    )

    return config


# =========================================================
# EXPORT
# =========================================================

def default_config_dict() -> Dict[str, Any]:
    return AggregationConfig().model_dump()