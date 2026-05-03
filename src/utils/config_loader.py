"""
File: config_loader.py
Location: src/utils/

Production-grade multi-task configuration loader for TruthLens.

Supports:
- N-task architecture
- Per-task datasets
- Task-aware model config
- Sampling strategies
- Encoder control
- Strong validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache
from copy import deepcopy

import yaml

logger = logging.getLogger(__name__)


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


# =========================================================
# CONSTANTS
# =========================================================

VALID_TASK_TYPES = {"binary", "multiclass", "multilabel"}


# =========================================================
# DATACLASSES
# =========================================================

@dataclass(slots=True)
class TaskDatasetConfig:
    train_path: Path
    validation_path: Optional[Path]
    test_path: Optional[Path]
    text_column: str = "text"
    label_column: str = "label"


@dataclass(slots=True)
class TaskConfig:
    name: str
    task_type: str
    num_labels: int
    dataset: TaskDatasetConfig


@dataclass(slots=True)
class EncoderConfig:
    name: str
    pretrained_model: Optional[str]
    hidden_size: int
    freeze_epochs: int = 0
    layerwise_lr_decay: Optional[float] = None


@dataclass(slots=True)
class ModelConfig:
    encoder: EncoderConfig


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    gradient_accumulation_steps: int = 1
    device: str = "auto"
    max_grad_norm: float = 1.0
    fp16: bool = False


@dataclass(slots=True)
class SamplingConfig:
    strategy: str = "uniform"
    temperature: float = 1.0


@dataclass(slots=True)
class ExperimentConfig:
    seed: int
    output_dir: Path
    experiment_name: str


@dataclass(slots=True)
class AppConfig:
    model: ModelConfig
    tasks: Dict[str, TaskConfig]
    training: TrainingConfig
    sampling: SamplingConfig
    experiment: ExperimentConfig


# =========================================================
# PATH RESOLUTION
# =========================================================

def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


# =========================================================
# YAML LOADER
# =========================================================

@lru_cache(maxsize=4)
def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    path = _resolve_path(config_path or DEFAULT_CONFIG_PATH)

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    logger.info("Loading config from %s", path)

    try:
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise RuntimeError("Invalid YAML") from e

    return deepcopy(config)


# =========================================================
# VALIDATION
# =========================================================

def _validate_required(config: Dict[str, Any], keys: list[str]):
    missing = [k for k in keys if k not in config]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")


def _validate_task(name: str, cfg: Dict[str, Any]):
    if "type" not in cfg:
        raise ValueError(f"{name}: missing type")

    if cfg["type"] not in VALID_TASK_TYPES:
        raise ValueError(f"{name}: invalid task type")

    if "num_labels" not in cfg:
        raise ValueError(f"{name}: missing num_labels")

    if cfg["num_labels"] <= 0:
        raise ValueError(f"{name}: num_labels must be > 0")

    if "dataset" not in cfg:
        raise ValueError(f"{name}: missing dataset config")


# =========================================================
# COMPAT: normalize the lightweight YAML schema to the
# verbose one expected by the loader below.
# =========================================================

_DEFAULT_NUM_LABELS = {
    "binary": 2,
    "multiclass": 3,
    "multilabel": 2,
}


def _resolve_num_labels(name: str, task_type: str, provided: Optional[int]) -> int:
    """Return ``provided`` when set, else fall back to a per-type default
    and log a loud warning. Per-task class counts are dataset-specific
    (bias=2, ideology=3, propaganda=2, narrative=3, narrative_frame=5,
    emotion=11) so guessing from ``task_type`` alone is brittle —
    callers that hit this path almost always have a config bug.
    """
    if provided is not None:
        return int(provided)

    fallback = _DEFAULT_NUM_LABELS.get(task_type, 2)
    logger.warning(
        "Task '%s' missing explicit num_labels; falling back to %d "
        "for type='%s'. Add num_labels to config.yaml to silence this.",
        name, fallback, task_type,
    )
    return fallback


def _normalize_encoder(model_cfg: Any) -> Dict[str, Any]:
    """Accept either ``encoder: "name"`` or ``encoder: {name: ...}``."""
    if not isinstance(model_cfg, dict):
        return {}

    encoder = model_cfg.get("encoder", {})
    if isinstance(encoder, str):
        return {
            "name": encoder,
            "hidden_size": model_cfg.get("hidden_dim", model_cfg.get("hidden_size", 768)),
        }
    if isinstance(encoder, dict):
        return encoder
    return {}


def _normalize_task(name: str, cfg: Any) -> Dict[str, Any]:
    """Accept either ``task: "multiclass"`` or full task dict."""
    if isinstance(cfg, str):
        task_type = cfg
        return {
            "type": task_type,
            "num_labels": _resolve_num_labels(name, task_type, None),
            "dataset": {
                "train_path": f"data/{name}/train.csv",
                "validation_path": None,
                "test_path": None,
            },
        }
    if isinstance(cfg, dict):
        out = dict(cfg)
        out["num_labels"] = _resolve_num_labels(
            name, out.get("type", ""), out.get("num_labels"),
        )
        out.setdefault(
            "dataset",
            {
                "train_path": f"data/{name}/train.csv",
                "validation_path": None,
                "test_path": None,
            },
        )
        return out
    raise ValueError(f"{name}: unsupported task config form ({type(cfg).__name__})")


# =========================================================
# MAIN LOADER
# =========================================================

def load_app_config(config_path: str | Path | None = None) -> AppConfig:
    config = load_config(config_path)

    _validate_required(config, ["model", "tasks", "training"])

    # ---------------- MODEL ----------------
    encoder_cfg = _normalize_encoder(config["model"])

    model = ModelConfig(
        encoder=EncoderConfig(
            name=encoder_cfg.get("name", "roberta-base"),
            pretrained_model=encoder_cfg.get("pretrained_model"),
            hidden_size=encoder_cfg.get("hidden_size", 768),
            freeze_epochs=encoder_cfg.get("freeze_epochs", 0),
            layerwise_lr_decay=encoder_cfg.get("layerwise_lr_decay"),
        )
    )

    # ---------------- TASKS ----------------
    tasks: Dict[str, TaskConfig] = {}

    for name, raw_cfg in config["tasks"].items():
        cfg = _normalize_task(name, raw_cfg)
        _validate_task(name, cfg)

        ds = cfg["dataset"]

        dataset = TaskDatasetConfig(
            train_path=_resolve_path(ds["train_path"]),
            validation_path=_resolve_path(ds["validation_path"]) if ds.get("validation_path") else None,
            test_path=_resolve_path(ds["test_path"]) if ds.get("test_path") else None,
            text_column=ds.get("text_column", "text"),
            label_column=ds.get("label_column", "label"),
        )

        tasks[name] = TaskConfig(
            name=name,
            task_type=cfg["type"],
            num_labels=cfg["num_labels"],
            dataset=dataset,
        )

    # ---------------- TRAINING ----------------
    t = config["training"]
    optim_cfg = config.get("optimizer", {}) if isinstance(config.get("optimizer", {}), dict) else {}
    data_cfg = config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}

    training = TrainingConfig(
        batch_size=t.get("batch_size", data_cfg.get("batch_size", 16)),
        epochs=t.get("epochs", 1),
        learning_rate=t.get("learning_rate", optim_cfg.get("lr", 3e-5)),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 1),
        device=t.get("device", "auto"),
        max_grad_norm=t.get("max_grad_norm", 1.0),
        fp16=t.get("fp16", False),
    )

    # ---------------- SAMPLING ----------------
    s = config.get("task_sampling", {})

    sampling = SamplingConfig(
        strategy=s.get("strategy", "uniform"),
        temperature=s.get("temperature", 1.0),
    )

    # ---------------- EXPERIMENT ----------------
    e = config.get("experiment", {})

    experiment = ExperimentConfig(
        seed=e.get("seed", 42),
        output_dir=_resolve_path(e.get("output_dir", "outputs")),
        experiment_name=e.get("experiment_name", "truthlens"),
    )

    logger.info("Loaded config | tasks=%d", len(tasks))

    return AppConfig(
        model=model,
        tasks=tasks,
        training=training,
        sampling=sampling,
        experiment=experiment,
    )