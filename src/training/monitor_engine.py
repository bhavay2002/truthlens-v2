# src/models/training/monitor_engine.py

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch

# LOSS-1: Single source of truth for spike detection. The codebase previously
# shipped TWO ``SpikeDetector`` classes — one here (pure ratio) and one in
# ``instrumentation`` (bias-corrected EMA + ratio + z-score). Both wired to
# REDUCE_LR actions and could fire in the same step on conflicting policies.
# Re-export the stricter, bias-corrected version and let MonitoringEngine
# delegate to it, so there is one detector and one policy across the layer.
from src.training.instrumentation import SpikeDetector  # noqa: F401  (re-exported)

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class MonitoringConfig:
    spike_threshold: float = 3.0
    ema_alpha: float = 0.1
    health_threshold: float = 0.3

    enable_grad_monitor: bool = True
    grad_monitor_interval: int = 100

    enable_throughput: bool = True
    throughput_ema_alpha: float = 0.2  # NEW

    anomaly_on_nan: bool = True


# =========================================================
# HELPERS
# =========================================================

class EMA:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


class HealthScore:
    def compute(
        self,
        loss: float,
        ema: float,
        grad_norm: Optional[float] = None,
    ) -> float:

        if ema is None:
            return 1.0

        stability = 1.0 - min(abs(loss - ema) / (ema + 1e-9), 1.0)

        grad_penalty = 0.0
        if grad_norm is not None and grad_norm > 10:
            grad_penalty = min((grad_norm - 10) / 50, 1.0)

        return max(0.0, stability - grad_penalty)


# =========================================================
# ACTION ENUM
# =========================================================

class MonitorAction:
    NONE = "none"
    REDUCE_LR = "reduce_lr"
    SPIKE = "spike"
    NAN = "nan_detected"


# =========================================================
# ENGINE
# =========================================================

class MonitoringEngine:

    def __init__(self, config: Optional[MonitoringConfig] = None):

        self.config = config or MonitoringConfig()

        self.loss_ema = EMA(self.config.ema_alpha)
        self.throughput_ema = EMA(self.config.throughput_ema_alpha)

        self.spike_detector = SpikeDetector(threshold=self.config.spike_threshold)
        self.health = HealthScore()

        self.step = 0
        self._last_time = time.time()

        logger.info("MonitoringEngine initialized")

    # =====================================================
    # MAIN UPDATE
    # =====================================================

    def update(
        self,
        outputs: Dict[str, Any],
        *,
        model: Optional[torch.nn.Module] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:

        self.step += 1

        loss = self._extract_loss(outputs)

        # -------------------------
        # NAN GUARD
        # -------------------------

        if self.config.anomaly_on_nan and not torch.isfinite(torch.tensor([loss])):
            return self._build_metrics(
                loss, None, None, 0.0, MonitorAction.NAN
            )

        ema = self.loss_ema.update(loss)

        # LOSS-1: unified detector exposes ``detect(loss, ema, var=None)``.
        # With var=None it falls back to pure-ratio behaviour, matching the
        # previous local detector exactly while sharing one implementation.
        spike = self.spike_detector.detect(loss, ema)

        grad_norm = None

        if (
            self.config.enable_grad_monitor
            and model is not None
            and self.step % self.config.grad_monitor_interval == 0
        ):
            grad_norm = self._compute_grad_norm(model)

        health_score = self.health.compute(loss, ema, grad_norm)

        throughput = self._compute_throughput(batch_size)

        # -------------------------
        # ACTION POLICY (IMPROVED)
        # -------------------------

        if spike:
            action = MonitorAction.SPIKE
        elif health_score < self.config.health_threshold:
            action = MonitorAction.REDUCE_LR
        else:
            action = MonitorAction.NONE

        return self._build_metrics(
            loss,
            ema,
            grad_norm,
            health_score,
            action,
            spike=spike,
            throughput=throughput,
        )

    # =====================================================
    # BUILD METRICS
    # =====================================================

    def _build_metrics(
        self,
        loss,
        ema,
        grad_norm,
        health,
        action,
        *,
        spike=False,
        throughput=None,
    ):

        return {
            "monitor/loss": loss,
            "monitor/ema_loss": ema,
            "monitor/spike": spike,
            "monitor/grad_norm": grad_norm,
            "monitor/health": health,
            "monitor/action": action,
            "monitor/throughput": throughput,
        }

    # =====================================================
    # LOSS
    # =====================================================

    def _extract_loss(self, outputs: Dict[str, Any]) -> float:

        # N-CRIT-1: Previously this raised ``RuntimeError`` on a non-finite
        # loss, which crashed the run BEFORE the ``anomaly_on_nan`` policy
        # downstream had a chance to translate the NaN into a soft signal
        # (action=NAN).  That made the ``anomaly_on_nan`` flag a no-op —
        # any non-finite loss would crash regardless.  Return ``nan`` here
        # and let the caller's ``torch.isfinite`` guard apply the policy.
        loss = outputs.get("raw_loss")
        if loss is None:
            loss = outputs.get("loss")

        if isinstance(loss, torch.Tensor):
            if not torch.isfinite(loss):
                return float("nan")
            return float(loss.detach().item())

        if isinstance(loss, (int, float)):
            return float(loss)

        raise RuntimeError("Loss not found in outputs")

    # =====================================================
    # GRADIENTS (SAFE)
    # =====================================================

    def _compute_grad_norm(self, model: torch.nn.Module) -> float:
        # N-LOW-8: route through the canonical implementation instead of
        # maintaining yet another L2 reduction loop. See
        # ``training_setup._compute_grad_norm`` for the consolidation.
        from src.training.training_utils import compute_grad_norm
        return compute_grad_norm(model)

    # =====================================================
    # THROUGHPUT (SMOOTHED)
    # =====================================================

    def _compute_throughput(self, batch_size: Optional[int]) -> Optional[float]:

        if not self.config.enable_throughput or batch_size is None:
            return None

        now = time.time()
        duration = now - self._last_time
        self._last_time = now

        if duration <= 0:
            return None

        raw_tp = batch_size / duration
        return self.throughput_ema.update(raw_tp)