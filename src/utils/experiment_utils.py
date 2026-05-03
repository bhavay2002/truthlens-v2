
#src\utils\experiment_utils.py

from __future__ import annotations

import logging
import json
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict

import torch

from src.utils.json_utils import append_json
from src.utils.time_utils import timestamp

logger = logging.getLogger(__name__)


# =========================================================
# SYSTEM METADATA
# =========================================================

def _get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _get_device_info() -> Dict[str, Any]:
    if torch.cuda.is_available():
        return {
            "device": "cuda",
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
        }
    return {"device": "cpu"}


# =========================================================
# CONFIG SANITIZATION (CRITICAL)
# =========================================================

def _sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert config into JSON-serializable format.
    """

    def convert(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        if hasattr(obj, "__dict__"):
            return convert(vars(obj))
        return obj

    return convert(config)


# =========================================================
# EXPERIMENT RECORD
# =========================================================

@dataclass(slots=True)
class ExperimentRecord:
    experiment_id: str
    timestamp: str

    model: str
    tasks: Dict[str, Any]

    parameters: Dict[str, Any]
    metrics: Dict[str, Any]

    runtime_seconds: Optional[float]

    #  CRITICAL
    config_snapshot: Dict[str, Any]

    git_commit: str
    hostname: str
    device_info: Dict[str, Any]


# =========================================================
# ID GENERATION
# =========================================================

def generate_experiment_id(prefix: str = "exp") -> str:
    return f"{prefix}_{timestamp()}"


# =========================================================
# EXPERIMENT DIRECTORY
# =========================================================

def create_experiment_dir(base_dir: str | Path, exp_id: str) -> Path:
    path = Path(base_dir) / exp_id
    path.mkdir(parents=True, exist_ok=True)
    return path


# =========================================================
# RECORD BUILDER (FINAL VERSION)
# =========================================================

def create_experiment_record(
    model_name: str,
    tasks: Dict[str, Any],
    parameters: Dict[str, Any],
    metrics: Dict[str, Any],
    dataset: Optional[str] = None,
    runtime: Optional[float] = None,
    config_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    #  ENFORCE REPRODUCIBILITY
    if config_snapshot is None:
        raise ValueError(
            "config_snapshot is required for reproducibility"
        )

    ts = timestamp()
    exp_id = f"exp_{ts}"

    sanitized_config = _sanitize_config(config_snapshot)

    record = ExperimentRecord(
        experiment_id=exp_id,
        timestamp=ts,
        model=model_name,
        tasks=tasks,
        parameters=parameters,
        metrics=metrics,
        runtime_seconds=runtime,
        config_snapshot=sanitized_config,
        git_commit=_get_git_commit(),
        hostname=socket.gethostname(),
        device_info=_get_device_info(),
    )

    return asdict(record)


# =========================================================
# LOGGING (PERSISTENT)
# =========================================================

_REQUIRED_KEYS = {
    "experiment_id",
    "timestamp",
    "model",
    "tasks",
    "parameters",
    "metrics",
    "runtime_seconds",
    "config_snapshot",
    "git_commit",
}


def log_experiment(
    record: Dict[str, Any],
    base_dir: str | Path = "reports/experiments",
) -> Path:

    missing = _REQUIRED_KEYS - set(record.keys())
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    exp_id = record["experiment_id"]

    exp_dir = create_experiment_dir(base_dir, exp_id)

    # ---------------- SAVE FULL RECORD ----------------
    full_path = exp_dir / "experiment.json"

    with full_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    # ---------------- APPEND SUMMARY ----------------
    summary_path = Path(base_dir) / "experiments_summary.json"

    append_json(
        {
            "experiment_id": exp_id,
            "model": record["model"],
            "timestamp": record["timestamp"],
        },
        summary_path,
    )

    logger.info("Experiment saved: %s", exp_id)

    return full_path


# =========================================================
# STEP LOGGING
# =========================================================

def log_training_step(
    exp_dir: Path,
    step: int,
    task: str,
    loss: float,
):

    step_file = exp_dir / "training_steps.json"

    append_json(
        {
            "step": step,
            "task": task,
            "loss": loss,
        },
        step_file,
    )


# =========================================================
# METRIC LOGGING
# =========================================================

def log_metrics(
    exp_dir: Path,
    metrics: Dict[str, Any],
    split: str = "validation",
):

    file_path = exp_dir / f"{split}_metrics.json"

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)