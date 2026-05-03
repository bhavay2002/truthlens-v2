from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import mlflow
except ImportError:
    mlflow = None

from src.utils.device_utils import is_primary_process

logger = logging.getLogger(__name__)


# =========================================================
# SAFETY
# =========================================================

def _ensure_mlflow():
    if mlflow is None:
        raise RuntimeError("MLflow not installed")


# =========================================================
# FLATTEN
# =========================================================

def flatten_dict(d: Dict[str, Any], parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# =========================================================
# RUN CONTEXT
# =========================================================

class MLflowRun:

    def __init__(
        self,
        experiment_name: str = "truthlens",
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or {}

    def __enter__(self):
        _ensure_mlflow()

        if not is_primary_process():
            return self

        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run(run_name=self.run_name)

        if self.tags:
            mlflow.set_tags(self.tags)

        logger.info(f"[MLFLOW] Run started: {self.run_name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        if not is_primary_process():
            return

        status = "FAILED" if exc else "FINISHED"
        mlflow.end_run(status=status)

        logger.info(f"[MLFLOW] Run ended: {status}")


# =========================================================
# MULTI-TASK METRIC LOGGING (FIXED)
# =========================================================

def log_task_metrics(
    task: str,
    metrics: Dict[str, Any],
    step: Optional[int] = None,
):
    _ensure_mlflow()

    if not is_primary_process():
        return

    flat = flatten_dict(metrics)

    for key, value in flat.items():

        if not isinstance(value, (int, float)):
            continue

        name = f"{task}.{key}"

        try:
            mlflow.log_metric(name, float(value), step=step)
        except Exception:
            logger.warning(f"[MLFLOW] Failed metric: {name}")


# =========================================================
# GLOBAL METRICS
# =========================================================

def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None):
    _ensure_mlflow()

    if not is_primary_process():
        return

    flat = flatten_dict(metrics)

    for key, value in flat.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, float(value), step=step)


# =========================================================
# PARAMS
# =========================================================

def log_params(params: Dict[str, Any], prefix=""):
    _ensure_mlflow()

    if not is_primary_process():
        return

    flat = flatten_dict(params)

    for key, value in flat.items():
        name = f"{prefix}{key}" if prefix else key

        try:
            mlflow.log_param(name, value)
        except Exception:
            logger.warning(f"[MLFLOW] Failed param: {name}")


# =========================================================
# DATASET INFO
# =========================================================

def log_dataset_info(name: str, size=None, version=None, hash=None):
    _ensure_mlflow()

    if not is_primary_process():
        return

    mlflow.log_param("dataset.name", name)
    if size:
        mlflow.log_param("dataset.size", size)
    if version:
        mlflow.log_param("dataset.version", version)
    if hash:
        mlflow.log_param("dataset.hash", hash)


# =========================================================
# ARTIFACTS (STRUCTURED)
# =========================================================

def log_artifact(path: str | Path, artifact_path: str):
    _ensure_mlflow()

    if not is_primary_process():
        return

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    mlflow.log_artifact(str(path), artifact_path=artifact_path)


# =========================================================
# EVALUATION LOGGING
# =========================================================

def log_evaluation_report(report: Dict[str, Any]) -> None:
    """Serialize an evaluation report to a temp JSON and log it as an artifact."""
    _ensure_mlflow()

    if not is_primary_process():
        return

    import json
    import tempfile

    from src.evaluation.report_writer import _make_serializable

    safe_report = _make_serializable(report)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="eval_", delete=False
    ) as fh:
        tmp_path = Path(fh.name)
        json.dump(safe_report, fh, indent=2)

    try:
        mlflow.log_artifact(str(tmp_path), artifact_path="evaluation")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            logger.debug("Could not remove temp eval file: %s", tmp_path)


# =========================================================
# MODEL LOGGING (FIXED 🔥)
# =========================================================

def log_model(model, tokenizer=None, config=None, name="model"):
    _ensure_mlflow()

    if not is_primary_process():
        return

    try:
        import mlflow.pytorch

        mlflow.pytorch.log_model(model, name)

        # Save tokenizer
        if tokenizer:
            tok_path = Path("tokenizer")
            tokenizer.save_pretrained(tok_path)
            mlflow.log_artifacts(str(tok_path), artifact_path="tokenizer")

        # Save config
        if config:
            cfg_path = Path("config.json")
            import json
            with cfg_path.open("w") as f:
                json.dump(config, f, indent=2)

            mlflow.log_artifact(str(cfg_path), artifact_path="config")

        logger.info("[MLFLOW] Model + tokenizer + config logged")

    except Exception as e:
        logger.warning(f"[MLFLOW] Model logging failed: {e}")


# =========================================================
# SYSTEM INFO
# =========================================================

def log_system_info():
    _ensure_mlflow()

    if not is_primary_process():
        return

    import platform
    import sys

    mlflow.log_param("system.python", sys.version)
    mlflow.log_param("system.platform", platform.platform())