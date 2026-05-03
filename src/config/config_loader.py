from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

logger = logging.getLogger(__name__)


# =========================================================
# SECTIONS (TYPED)
# =========================================================

@dataclass(frozen=True)
class ProjectConfig:
    name: str
    seed: int


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    log_every: int
    eval_every: int
    checkpoint_every: int
    gradient_accumulation_steps: int
    early_stopping_patience: int
    # MIN-EPOCH-EARLY-STOPPING / GRAD-CLIP-1 / SPIKE-LR-RELAX / MIN-DELTA:
    # these knobs were added to ``config/config.yaml`` to control the
    # new training-stability behaviour (see the ``training:`` block
    # there for the full rationale). They have to be declared here too
    # because ``load_config`` below does ``TrainingConfig(**raw["training"])``
    # — i.e. a strict kwargs-init that crashes on any YAML key it
    # doesn't know about. Defaults reproduce the legacy behaviour
    # (no min-epoch floor, 1.0 grad-clip cap, aggressive 0.5 spike
    # halving, zero min-delta) so this struct stays back-compat with
    # older YAMLs that don't set them.
    min_epochs: int = 1
    max_grad_norm: float = 1.0
    spike_lr_scale: float = 0.5
    early_stopping_min_delta: float = 0.0
    # EXPLOSION-WATCHDOG: pre-clip gradient L2 norm above which
    # ``TrainingStep.run`` emits a ``Gradient spike detected`` warning.
    # Mirrors ``TrainingStepConfig.spike_warn_threshold`` and is read
    # from ``training.spike_warn_threshold`` in ``config.yaml`` via
    # ``_resolve_spike_warn_threshold`` in
    # ``src/training/create_multitask_trainer_fn.py``. Default 100.0
    # matches the audit-driven default; set to 0.0 in YAML to disable.
    spike_warn_threshold: float = 100.0
    # EXPLOSION-WATCHDOG-RESPONSE: factor applied to optimiser LR when
    # the watchdog above fires (separate from the legacy
    # ``spike_lr_scale`` which is wired into the REDUCE_LR pathway).
    # Mirrors ``TrainingStepConfig.spike_decay_factor``; read from
    # ``training.spike_decay_factor`` via ``_resolve_spike_decay_factor``
    # in ``src/training/create_multitask_trainer_fn.py``. POST-V6:
    # default relaxed 0.7 → 0.85 — the previous 0.7 compounded too
    # aggressively on consecutive spikes and stalled learning past
    # epoch 4. Set to 1.0 in YAML to keep the warning-only behaviour
    # without LR decay. Declared here defensively because
    # ``load_config`` does ``TrainingConfig(**raw["training"])``.
    spike_decay_factor: float = 0.85
    # EXPLOSION-WATCHDOG-SKIP: pre-clip gradient L2 norm above which
    # ``TrainingStep.run`` discards the optimiser step entirely.
    # Mirrors ``TrainingStepConfig.spike_skip_threshold``. Default
    # 150.0 places the skip threshold 50% above the warn threshold
    # (100.0); set to 0.0 in YAML to disable. Declared here
    # defensively because ``load_config`` does
    # ``TrainingConfig(**raw["training"])``.
    spike_skip_threshold: float = 150.0


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    lr: float
    weight_decay: float


@dataclass(frozen=True)
class SchedulerConfig:
    name: str
    step_mode: str  # step | epoch | metric
    warmup_steps: int


@dataclass(frozen=True)
class PrecisionConfig:
    use_amp: bool
    amp_dtype: str  # bf16 | fp16
    allow_tf32: bool
    # AMP-INIT-SCALE-FIX: optional ``GradScaler(init_scale=...)``
    # override (torch default is 2**16 = 65536). Declared here
    # defensively because ``load_config`` does
    # ``PrecisionConfig(**raw["precision"])`` and would otherwise crash
    # on the new YAML key. Mirrors
    # ``TrainingStepConfig.grad_scaler_init_scale``; ``None`` = keep
    # torch default. No-op under bf16 (scaler constructed disabled).
    grad_scaler_init_scale: Optional[float] = None


@dataclass(frozen=True)
class ModelConfig:
    encoder: str
    hidden_dim: int
    dropout: float
    gradient_checkpointing: bool = False
    flash_attention: bool = True
    torch_compile: bool = False
    compile_mode: str = "default"


@dataclass(frozen=True)
class LossConfig:
    ignore_index: int
    validate_loss: bool
    # NORMALIZER-ALPHA-DAMP: EMA decay forwarded to ``EMALossNormalizer``
    # via ``LossEngineConfig`` (see ``loss.normalizer_alpha`` in
    # ``config/config.yaml`` for the rationale). Declared here so the
    # strict ``LossConfig(**raw["loss"])`` kwargs-init in ``load_config``
    # below doesn't crash on the YAML key. Default ``None`` keeps the
    # ``EMALossNormalizer`` library default (0.1) when the YAML field
    # is absent.
    normalizer_alpha: Optional[float] = None


@dataclass(frozen=True)
class DataConfig:
    batch_size: int
    num_workers: int
    pin_memory: bool
    shuffle: bool


@dataclass(frozen=True)
class DistributedConfig:
    use_ddp: bool
    backend: str
    find_unused_parameters: bool


@dataclass(frozen=True)
class MonitoringConfig:
    spike_threshold: float
    ema_alpha: float
    health_threshold: float
    grad_monitor_interval: int


@dataclass(frozen=True)
class CheckpointConfig:
    dir: str
    max_checkpoints: int
    monitor_metric: str
    mode: str  # min | max


@dataclass(frozen=True)
class EvaluationConfig:
    device: Optional[str]


@dataclass(frozen=True)
class TrackingConfig:
    backend: str
    project_name: str
    run_name: Optional[str]
    tags: Dict[str, str]


# =========================================================
# ROOT CONFIG
# =========================================================

@dataclass(frozen=True)
class Config:
    project: ProjectConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    precision: PrecisionConfig
    model: ModelConfig
    tasks: Dict[str, str]
    task_weights: Dict[str, float]
    loss: LossConfig
    data: DataConfig
    distributed: DistributedConfig
    monitoring: MonitoringConfig
    checkpoint: CheckpointConfig
    evaluation: EvaluationConfig
    tracking: TrackingConfig


# =========================================================
# VALIDATION
# =========================================================

def _validate(cfg: Dict[str, Any]) -> None:
    required_top = [
        "project", "training", "optimizer", "scheduler", "precision",
        "model", "tasks", "task_weights", "loss", "data",
        "distributed", "monitoring", "checkpoint", "evaluation", "tracking"
    ]
    missing = [k for k in required_top if k not in cfg]
    if missing:
        raise ValueError(f"Missing top-level config keys: {missing}")

    # simple invariants
    if cfg["checkpoint"]["mode"] not in {"min", "max"}:
        raise ValueError("checkpoint.mode must be 'min' or 'max'")

    if cfg["scheduler"]["step_mode"] not in {"step", "epoch", "metric"}:
        raise ValueError("scheduler.step_mode must be step|epoch|metric")

    if cfg["precision"]["amp_dtype"] not in {"bf16", "fp16"}:
        raise ValueError("precision.amp_dtype must be bf16|fp16")

    # tasks vs weights alignment
    tasks = set(cfg["tasks"].keys())
    weights = set(cfg["task_weights"].keys())
    if tasks != weights:
        raise ValueError(f"tasks and task_weights mismatch: {tasks ^ weights}")


# =========================================================
# LOADER
# =========================================================

def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    _validate(raw)

    cfg = Config(
        project=ProjectConfig(**raw["project"]),
        training=TrainingConfig(**raw["training"]),
        optimizer=OptimizerConfig(**raw["optimizer"]),
        scheduler=SchedulerConfig(**raw["scheduler"]),
        precision=PrecisionConfig(**raw["precision"]),
        model=ModelConfig(**raw["model"]),
        tasks=dict(raw["tasks"]),
        task_weights=dict(raw["task_weights"]),
        loss=LossConfig(**raw["loss"]),
        data=DataConfig(**raw["data"]),
        distributed=DistributedConfig(**raw["distributed"]),
        monitoring=MonitoringConfig(**raw["monitoring"]),
        checkpoint=CheckpointConfig(**raw["checkpoint"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        tracking=TrackingConfig(**raw["tracking"]),
    )

    logger.info("Config loaded: %s | model=%s | epochs=%d",
                cfg.project.name, cfg.model.encoder, cfg.training.epochs)

    return cfg