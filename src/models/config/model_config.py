from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml


# =========================================================
# Encoder Configuration
# =========================================================

@dataclass
class EncoderConfig:
    model_name: str = "roberta-base"
    pooling: str = "cls"
    device: Optional[str] = None

    gradient_checkpointing: bool = False
    enable_fused_attention: bool = True

    use_amp: bool = True
    amp_dtype: str = "float16"

    use_compile: bool = False
    compile_mode: str = "default"

    freeze_layers: int = 0
    enable_adapters: bool = False
    adapter_type: str = "lora"
    adapter_dim: int = 16


# =========================================================
# Head Configuration
# =========================================================

@dataclass
class HeadConfig:
    input_dim: int
    output_dim: int
    dropout: float = 0.1
    use_layernorm: bool = False


@dataclass
class RegressionConfig:
    enabled: bool = False
    output_dim: int = 1
    hidden_dim: Optional[int] = None
    activation: str = "gelu"
    dropout: float = 0.1


# =========================================================
# Task Configuration
# =========================================================

@dataclass
class TaskConfig:
    name: str
    num_labels: int
    task_type: str = "multi_class"
    regression: Optional[RegressionConfig] = None
    use_label_smoothing: bool = False
    # A3.1: per-task loss weight on the canonical YAML-driven config
    # surface. Previously per-task weights only existed on the
    # convenience ``MultiTaskTruthLensConfig`` (one float per known
    # task), which (a) couldn't express a weight for tasks added by
    # the YAML pipeline and (b) silently defaulted everything to 1.0.
    # The convenience config still carries its named ``*_weight``
    # fields for back-compat, but ``TaskConfig.loss_weight`` is now
    # the authoritative source of truth.
    loss_weight: float = 1.0


# =========================================================
# Training Configuration
# =========================================================

@dataclass
class TrainingConfig:
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 32
    num_epochs: int = 3

    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    use_scheduler: bool = True
    scheduler_type: str = "linear"
    warmup_steps: int = 0

    early_stopping: bool = True
    early_stopping_patience: int = 3

    seed: int = 42

    # MIN-EPOCH-EARLY-STOPPING / SPIKE-LR-RELAX / EXPLOSION-WATCHDOG:
    # declared here so the strict ``TrainingConfig(**raw.get("training",
    # {}))`` call inside ``ModelConfigLoader.load_multitask_config``
    # doesn't crash on the new YAML keys. Defaults preserve the legacy
    # behaviour. NOTE: this dataclass uses ``num_epochs`` while the
    # YAML key is ``epochs``; the multitask training factory bypasses
    # this loader entirely (see ``MT-FACTORY-NOLEGACY-CFG`` in
    # ``create_multitask_trainer_fn.py``), so the mismatch is
    # tolerated for back-compat with the few remaining callers
    # (``model_factory``, ``encoder_factory``, inference
    # ``model_loader``) that only need encoder / task metadata from
    # this struct, not training hyperparameters.
    min_epochs: int = 1
    spike_lr_scale: float = 0.5
    spike_warn_threshold: float = 100.0
    # EXPLOSION-WATCHDOG-RESPONSE: declared so the strict
    # ``TrainingConfig(**raw.get("training", {}))`` call inside
    # ``ModelConfigLoader.load_multitask_config`` doesn't crash on the
    # new YAML key. POST-V6: default relaxed 0.7 → 0.85 (see
    # ``TrainingStepConfig.spike_decay_factor`` for rationale).
    spike_decay_factor: float = 0.85
    # EXPLOSION-WATCHDOG-SKIP: declared defensively for the same
    # reason as the fields above. Mirrors
    # ``TrainingStepConfig.spike_skip_threshold``; default 150.0
    # places the skip threshold 50% above the warn threshold (100.0).
    spike_skip_threshold: float = 150.0


# =========================================================
# Uncertainty Configuration
# =========================================================

@dataclass
class UncertaintyConfig:
    enable_mc_dropout: bool = False
    mc_samples: int = 10

    enable_deep_ensemble: bool = False
    ensemble_size: int = 3


# =========================================================
# Regularization Configuration
# =========================================================

@dataclass
class RegularizationConfig:
    label_smoothing: float = 0.0
    dropout: float = 0.1

    use_mixup: bool = False
    mixup_alpha: float = 0.2

    use_adversarial_training: bool = False
    adv_epsilon: float = 1e-5


# =========================================================
# Monitoring Configuration
# =========================================================

@dataclass
class MonitoringConfig:
    enable_drift_detection: bool = True
    drift_threshold: float = 0.1

    enable_uncertainty_monitoring: bool = True
    enable_confidence_tracking: bool = True


# =========================================================
# MultiTask Model Configuration
# =========================================================

@dataclass
class MultiTaskModelConfig:

    encoder: EncoderConfig
    tasks: Dict[str, TaskConfig]

    training: TrainingConfig = field(default_factory=TrainingConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    dropout: float = 0.1
    shared_encoder: bool = True
    reduce_intermediate_allocation: bool = True

    metadata: Dict[str, Any] = field(default_factory=dict)


# =========================================================
# Config Loader
# =========================================================

class ModelConfigLoader:

    @staticmethod
    def load_yaml(config_path: str | Path) -> Dict[str, Any]:

        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _load_tasks(raw_tasks: Dict[str, Any]) -> Dict[str, TaskConfig]:

        tasks_cfg = {}

        for name, task_data in raw_tasks.items():

            regression_cfg = None

            if isinstance(task_data.get("regression"), dict):
                regression_cfg = RegressionConfig(**task_data["regression"])

            tasks_cfg[name] = TaskConfig(
                name=name,
                num_labels=task_data["num_labels"],
                task_type=task_data.get("task_type", "multi_class"),
                regression=regression_cfg,
                use_label_smoothing=task_data.get("use_label_smoothing", False),
                # A3.1: pick up the optional ``loss_weight`` from YAML.
                loss_weight=float(task_data.get("loss_weight", 1.0)),
            )

        return tasks_cfg

    @staticmethod
    def load_multitask_config(config_path: str | Path) -> MultiTaskModelConfig:

        raw = ModelConfigLoader.load_yaml(config_path)

        encoder_cfg = EncoderConfig(**raw.get("encoder", {}))

        tasks_cfg = ModelConfigLoader._load_tasks(raw["tasks"])

        training_cfg = TrainingConfig(**raw.get("training", {}))
        uncertainty_cfg = UncertaintyConfig(**raw.get("uncertainty", {}))
        regularization_cfg = RegularizationConfig(**raw.get("regularization", {}))
        monitoring_cfg = MonitoringConfig(**raw.get("monitoring", {}))

        return MultiTaskModelConfig(
            encoder=encoder_cfg,
            tasks=tasks_cfg,
            training=training_cfg,
            uncertainty=uncertainty_cfg,
            regularization=regularization_cfg,
            monitoring=monitoring_cfg,
            dropout=raw.get("dropout", 0.1),
            shared_encoder=raw.get("shared_encoder", True),
            reduce_intermediate_allocation=raw.get(
                "reduce_intermediate_allocation", True
            ),
            metadata=raw.get("metadata", {}),
        )