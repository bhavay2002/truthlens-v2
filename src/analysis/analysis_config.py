# src/analysis/analysis_config.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.analysis.spacy_config import SpacyConfig


# =========================================================
# GLOBAL FLAGS
# =========================================================

@dataclass
class GlobalConfig:
    """
    Global runtime controls.
    """

    enable_logging: bool = True
    enable_debug: bool = False
    enable_timing: bool = True
    enable_validation: bool = True

    strict_mode: bool = False  # raise errors on missing features
    fail_fast: bool = False    # stop pipeline on analyzer failure

    max_text_length: int = 100_000
    truncate_text: bool = True


# =========================================================
# ANALYZER CONFIG
# =========================================================

@dataclass
class AnalyzerConfig:
    """
    Per-analyzer configuration.
    """

    enabled: bool = True
    order: int = 0
    weight: float = 1.0

    # future: thresholding / calibration
    threshold: Optional[float] = None

    # optional override dependencies
    requires: List[str] = field(default_factory=list)


# =========================================================
# FEATURE CONTROL
# =========================================================

@dataclass
class FeatureControlConfig:
    """
    Controls feature normalization & validation.
    """

    normalize_features: bool = True
    clip_features: bool = True
    clip_range: tuple[float, float] = (0.0, 1.0)

    validate_schema: bool = True
    enforce_schema: bool = False  # strict schema match

    track_feature_completeness: bool = True


# =========================================================
# PIPELINE CONFIG
# =========================================================

@dataclass
class PipelineConfig:
    """
    Controls pipeline execution.
    """

    batch_size: int = 32
    n_process: int = 1

    use_spacy_pipe: bool = True

    cache_docs: bool = False
    reuse_context: bool = False

    enable_parallel_analyzers: bool = False  # future


# =========================================================
# EXPERIMENT CONFIG (VERY IMPORTANT)
# =========================================================

@dataclass
class ExperimentConfig:
    """
    Used for research experiments and ablations.
    """

    experiment_name: str = "default"
    version: str = "v1"

    seed: int = 42

    enable_ablation: bool = False
    disabled_analyzers: List[str] = field(default_factory=list)

    log_features: bool = False
    log_outputs: bool = False


# =========================================================
# MAIN CONFIG
# =========================================================

@dataclass
class AnalysisConfig:
    """
    Master configuration object.
    """

    global_config: GlobalConfig = field(default_factory=GlobalConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    features: FeatureControlConfig = field(default_factory=FeatureControlConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    spacy: SpacyConfig = field(default_factory=SpacyConfig)

    analyzers: Dict[str, AnalyzerConfig] = field(default_factory=dict)

    # -----------------------------------------------------

    def is_enabled(self, name: str) -> bool:
        cfg = self.analyzers.get(name)
        return cfg.enabled if cfg else True

    # -----------------------------------------------------

    def get_order(self, name: str, default: int = 0) -> int:
        cfg = self.analyzers.get(name)
        return cfg.order if cfg else default

    # -----------------------------------------------------

    def get_requires(self, name: str) -> List[str]:
        cfg = self.analyzers.get(name)
        return cfg.requires if cfg else []

    # -----------------------------------------------------

    def apply_ablation(self):
        """
        Disable analyzers based on experiment config.
        """
        if not self.experiment.enable_ablation:
            return

        for name in self.experiment.disabled_analyzers:
            if name in self.analyzers:
                self.analyzers[name].enabled = False


# =========================================================
# DEFAULT CONFIG BUILDER
# =========================================================

def build_default_config() -> AnalysisConfig:

    config = AnalysisConfig()

    # -----------------------------------------------------
    # Analyzer defaults
    # -----------------------------------------------------

    # NOTE: keys MUST match the names used by `build_default_registry`
    # (see src/analysis/analysis_registry.py). Previously this dict used
    # short aliases ("omission", "conflict", "propagation", ...) that did
    # not exist in the registry — so per-analyzer enabled=False / ordering
    # / ablation flags silently no-op'd. Keep these aligned.
    default_orders = {
        "rhetorical": 1,
        "argument": 2,
        "context": 3,
        "discourse": 4,
        "emotion": 5,
        "framing": 6,
        "information": 7,
        "information_omission": 8,
        "ideology": 9,
        "narrative_role": 10,
        "narrative_conflict": 11,
        "narrative_propagation": 12,
        "narrative_temporal": 13,
        "source": 14,
    }

    for name, order in default_orders.items():
        config.analyzers[name] = AnalyzerConfig(
            enabled=True,
            order=order,
        )

    return config


# =========================================================
# CONFIG ↔ REGISTRY VALIDATION (CRIT-A3)
# =========================================================

def validate_config_against_registry(
    config: "AnalysisConfig",
    registry_names: List[str],
) -> None:
    """Raise if `config.analyzers` references unknown analyzer names.

    Called at pipeline construction time so misspelled aliases (the bug
    behind CRIT-A3) fail loudly instead of silently no-op'ing ablation /
    ordering / enabled flags.
    """
    registry_set = set(registry_names)
    unknown = [n for n in config.analyzers.keys() if n not in registry_set]
    if unknown:
        raise RuntimeError(
            "AnalysisConfig.analyzers references unknown analyzer names "
            f"(not in registry): {unknown}. Registry contains: "
            f"{sorted(registry_set)}"
        )


# =========================================================
# GLOBAL SINGLETON
# =========================================================

ANALYSIS_CONFIG: AnalysisConfig = build_default_config()