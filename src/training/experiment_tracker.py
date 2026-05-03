from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class ExperimentTrackerConfig:
    backend: str = "none"
    project_name: str = "truthlens"
    run_name: Optional[str] = None
    tracking_uri: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    group: Optional[str] = None   # ✅ NEW (CV / tuning grouping)


# =========================================================
# TRACKER
# =========================================================

class ExperimentTracker:

    def __init__(self, config: Optional[ExperimentTrackerConfig] = None):

        self.config = config or ExperimentTrackerConfig()
        self.backend = self.config.backend.lower()

        self._step = 0
        self._start_time = time.time()

        self._init_backend()

        logger.info("ExperimentTracker initialized | backend=%s", self.backend)

    # =====================================================
    # DISTRIBUTED SAFETY
    # =====================================================

    def _is_main(self):
        try:
            import torch.distributed as dist
            return not dist.is_initialized() or dist.get_rank() == 0
        except Exception:
            return True

    def _safe(self, fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as e:
            logger.warning("Tracker error: %s", e)

    # =====================================================
    # INIT
    # =====================================================

    def _init_backend(self):

        if self.backend == "mlflow":
            import mlflow

            if self.config.tracking_uri:
                mlflow.set_tracking_uri(self.config.tracking_uri)

            mlflow.set_experiment(self.config.project_name)

            mlflow.start_run(run_name=self.config.run_name)

            if self.config.tags:
                mlflow.set_tags(self.config.tags)

        elif self.backend == "wandb":
            import wandb

            wandb.init(
                project=self.config.project_name,
                name=self.config.run_name,
                tags=list(self.config.tags.keys()),
                group=self.config.group,  # ✅ NEW
                config={},
            )

        elif self.backend == "none":
            return

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    # =====================================================
    # METRIC HELPERS
    # =====================================================

    def _flatten(self, metrics: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
        """
        Flatten nested dict:
        {"a": {"b": 1}} → {"a/b": 1}
        """
        flat = {}

        for k, v in metrics.items():
            name = f"{prefix}/{k}" if prefix else k

            if isinstance(v, dict):
                flat.update(self._flatten(v, name))
            else:
                try:
                    flat[name] = float(v)
                except Exception:
                    continue

        return flat

    # =====================================================
    # LOGGING
    # =====================================================

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        *,
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ):

        if not self._is_main():
            return

        # N-HIGH-3: Previously this UNCONDITIONALLY did ``self._step = step + 1``,
        # which meant any caller that explicitly passed ``step=`` (Trainer
        # passes ``self.global_step`` for both train and eval) was silently
        # OVERWRITING the internal step counter — so any later
        # ``log_metrics(...)`` without an explicit step started counting
        # from the caller's last value (often a huge gap or even a step
        # that went BACKWARDS, which MLflow rejects). Only auto-advance the
        # internal counter when the caller did NOT supply a step.
        if step is None:
            step = self._step
            self._step += 1

        metrics = self._flatten(metrics, prefix)

        # -------------------------
        # ADD SYSTEM METRICS
        # -------------------------

        metrics["time/elapsed"] = time.time() - self._start_time

        # -------------------------
        # BACKENDS
        # -------------------------

        if self.backend == "mlflow":
            import mlflow
            for k, v in metrics.items():
                self._safe(mlflow.log_metric, k, v, step=step)

        elif self.backend == "wandb":
            import wandb
            self._safe(wandb.log, metrics, step=step)

    def log_params(self, params: Dict[str, Any]):

        if not self._is_main():
            return

        if self.backend == "mlflow":
            import mlflow
            self._safe(mlflow.log_params, params)

        elif self.backend == "wandb":
            import wandb
            self._safe(wandb.config.update, params, allow_val_change=True)

    def log_config(self, config: Dict[str, Any]):
        """
        Log full experiment config (important for reproducibility)
        """
        self.log_params(config)

    def log_artifact(self, path: str):

        if not self._is_main():
            return

        if self.backend == "mlflow":
            import mlflow
            self._safe(mlflow.log_artifact, path)

        elif self.backend == "wandb":
            import wandb
            self._safe(wandb.save, path)

    # =====================================================
    # ADVANCED
    # =====================================================

    def watch_model(self, model):
        if self.backend == "wandb":
            import wandb
            self._safe(wandb.watch, model)

    def log_lr(self, lr: float):
        self.log_metrics({"lr": lr})

    def log_throughput(self, value: float):
        self.log_metrics({"throughput": value})

    # =====================================================
    # FINALIZE
    # =====================================================

    def finish(self):

        if not self._is_main():
            return

        if self.backend == "mlflow":
            import mlflow
            self._safe(mlflow.end_run)

        elif self.backend == "wandb":
            import wandb
            self._safe(wandb.finish)

    # =====================================================
    # CONTEXT
    # =====================================================

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.finish()