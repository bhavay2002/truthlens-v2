#src\training\instrumentation.py

from __future__ import annotations
 
import math
import os
import statistics
import time
from collections import deque, defaultdict, Counter
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# CORE SAFETY UTILS
# =========================================================

def _to_float(x: Any) -> float:
    if torch.is_tensor(x):
        return float(x.detach().item())
    return float(x)


def _is_finite(x: float) -> bool:
    return math.isfinite(x)


# =========================================================
# LOSS TRACKING (BIAS-CORRECTED EMA)
# =========================================================

class LossTracker:
    def __init__(self, tasks: Iterable[str], alpha: float = 0.1, eps: float = 1e-8):
        self.alpha = alpha
        self.eps = eps
        self.ema = {t: 0.0 for t in tasks}
        self.steps = {t: 0 for t in tasks}
        # N-CRIT-2: per-task non-finite counter — surfaced for AutoDebugEngine.
        self._nan_counter: Dict[str, int] = {}

    def update(self, losses: Dict[str, Any]) -> Dict[str, float]:
        out = {}

        for t, v in losses.items():
            val = _to_float(v)

            # N-CRIT-2: Previously this raised ``RuntimeError`` on a single
            # non-finite per-task loss, which crashed the WHOLE training
            # step (including every other task that was healthy) and bypassed
            # the ``skip_nan_loss`` policy in TrainingStep.  The contract is
            # to track losses, not to police them; surface the failure via
            # an internal counter so AutoDebugEngine still classifies the
            # event as ``nan_loss`` downstream, then skip the EMA update for
            # this task so the EMA isn't poisoned by NaN/Inf.
            if not _is_finite(val):
                self._nan_counter[t] = self._nan_counter.get(t, 0) + 1
                # Carry forward the previous bias-corrected value so the
                # consumer sees a stable signal instead of a missing key.
                prev = self.ema.get(t, 0.0)
                steps = self.steps.get(t, 0)
                bias = 1 - (1 - self.alpha) ** steps if steps > 0 else self.eps
                out[t] = prev / (bias + self.eps)
                continue

            self.steps[t] = self.steps.get(t, 0) + 1

            prev = self.ema.get(t, 0.0)
            new = self.alpha * val + (1 - self.alpha) * prev
            self.ema[t] = new

            bias = 1 - (1 - self.alpha) ** self.steps[t]
            out[t] = new / (bias + self.eps)

        return out


# =========================================================
# LOSS STATS (VARIANCE)
# =========================================================

class LossStats:
    def __init__(self, window: int = 50):
        self.window = window
        self.history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    def update(self, losses: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        out = {}

        for t, v in losses.items():
            val = _to_float(v)
            h = self.history[t]
            h.append(val)

            # N-MED-1: previously this allocated a fresh ``torch.tensor`` per
            # task per step (and on CUDA, that's a host→device sync) just to
            # compute a sample variance over a Python deque of ~50 floats.
            # ``statistics`` is implemented in C and stays on the host —
            # measurably faster and free of accidental GPU traffic.
            mean = statistics.fmean(h)
            var = statistics.variance(h) if len(h) > 1 else 0.0

            out[t] = {"mean": mean, "var": var}

        return out


# =========================================================
# GRAD TRACKING
# =========================================================

class GradTracker:
    def __init__(self, window: int = 50):
        self.history = deque(maxlen=window)

    def update(self, model: nn.Module) -> Dict[str, float]:
        total = 0.0
        count = 0

        for p in model.parameters():
            if p.grad is None:
                continue

            g = p.grad.detach()
            n = g.norm().item()

            total += n * n
            count += 1

        total_norm = math.sqrt(total)

        record = {
            "total_norm": total_norm,
            "n_params": count,
        }

        self.history.append(record)
        return record


# =========================================================
# GRADNORM (UPGRADED)
# =========================================================

class GradNorm:
    def __init__(self, tasks: Iterable[str], alpha: float = 0.5):
        self.tasks = list(tasks)
        self.alpha = alpha
        self.initial_losses: Optional[Dict[str, float]] = None

    def compute(
        self,
        losses: Dict[str, torch.Tensor],
        shared_params: Iterable[nn.Parameter],
    ) -> Dict[str, float]:

        grads = {}
        params = [p for p in shared_params if p.requires_grad]

        for t, loss in losses.items():
            g = torch.autograd.grad(
                loss,
                params,
                retain_graph=True,
                allow_unused=True,
            )

            norm = 0.0
            for gg in g:
                if gg is not None:
                    norm += gg.norm().item() ** 2

            grads[t] = math.sqrt(norm)

        loss_vals = {t: _to_float(l) for t, l in losses.items()}

        if self.initial_losses is None:
            self.initial_losses = {t: max(v, 1e-6) for t, v in loss_vals.items()}

        ratios = {
            t: loss_vals[t] / self.initial_losses[t]
            for t in self.tasks
        }

        avg = sum(ratios.values()) / len(ratios)

        target = {
            t: avg * (ratios[t] ** self.alpha)
            for t in self.tasks
        }

        weights = {
            t: target[t] / (grads[t] + 1e-6)
            for t in self.tasks
        }

        # normalize
        s = sum(weights.values())
        scale = len(weights) / (s + 1e-6)

        return {t: w * scale for t, w in weights.items()}


# =========================================================
# SPIKE DETECTOR
# =========================================================

class SpikeDetector:
    def __init__(self, threshold: float = 2.5):
        self.threshold = threshold

    def detect(self, loss: float, ema: float, var: Optional[float] = None) -> bool:
        if not _is_finite(loss):
            return True

        ratio = loss / (ema + 1e-8) if ema > 0 else 0.0

        z = 0.0
        if var and var > 0:
            z = (loss - ema) / (math.sqrt(var) + 1e-8)

        return ratio > self.threshold or z > self.threshold


# =========================================================
# ANOMALY CLASSIFIER
# =========================================================

class AnomalyClassifier:
    def classify(
        self,
        loss: float,
        ema: float,
        grad_norm: float,
        logits: Optional[torch.Tensor] = None,
    ) -> str:

        if not _is_finite(loss):
            return "nan_loss"

        if grad_norm > 1000:
            return "exploding_gradients"

        if grad_norm < 1e-7:
            return "vanishing_gradients"

        if logits is not None and logits.numel() > 1:
            if logits.std().item() < 1e-4:
                return "logit_collapse"

        if ema > 0 and loss / (ema + 1e-8) > 2.5:
            return "loss_spike"

        return "normal"


# =========================================================
# FAILURE MEMORY
# =========================================================

class FailureMemory:
    def __init__(self, max_size: int = 500):
        self.data = defaultdict(list)
        self.max_size = max_size

    def store(self, ftype: str, signals: Dict[str, Any]):
        bucket = self.data[ftype]
        bucket.append({"time": time.time(), "signals": signals})

        if len(bucket) > self.max_size:
            bucket.pop(0)

    def stats(self):
        return {k: len(v) for k, v in self.data.items()}


# =========================================================
# AUTO DEBUG ENGINE (FINAL UPGRADE)
# =========================================================

class AutoDebugEngine:
    def __init__(
        self,
        tasks: Iterable[str],
        use_gradnorm: bool = False,
    ):

        self.loss_tracker = LossTracker(tasks)
        self.loss_stats = LossStats()
        self.grad_tracker = GradTracker()
        self.spike_detector = SpikeDetector()
        self.classifier = AnomalyClassifier()
        self.memory = FailureMemory()

        self.gradnorm = GradNorm(tasks) if use_gradnorm else None

        # NEW: throughput + trend tracking
        self._throughput_hist = deque(maxlen=50)
        self._failure_counter = Counter()

    # =====================================================
    # MAIN STEP
    # =====================================================

    def step(
        self,
        *,
        losses: Dict[str, torch.Tensor],
        total_loss: torch.Tensor,
        model: nn.Module,
        shared_params: Optional[Iterable[nn.Parameter]] = None,
        logits: Optional[torch.Tensor] = None,
        throughput: Optional[float] = None,
        cached_grad_norm: Optional[float] = None,
    ) -> Dict[str, Any]:

        # -------------------------
        # LOSS TRACKING
        # -------------------------

        ema_losses = self.loss_tracker.update(losses)
        stats = self.loss_stats.update(losses)

        # -------------------------
        # GRAD TRACKING
        #
        # REC-3: ``GradTracker.update`` iterates every parameter and
        # computes the L2 total norm — which is the EXACT same work that
        # the trainer just did via ``clip_grad_norm_``. Worse, by the
        # time this runs the trainer has already called
        # ``optimizer.zero_grad(set_to_none=True)``, so ``p.grad`` is
        # ``None`` for every parameter and ``GradTracker`` would record
        # ``total_norm=0`` (invisible silent-correctness bug — the
        # anomaly classifier would see "vanishing_gradients" on every
        # step!). When the trainer passes the cached value, append it
        # directly to the GradTracker history without re-iterating.
        # -------------------------

        if cached_grad_norm is not None:
            grad_stats = {
                "total_norm": float(cached_grad_norm),
                "n_params": -1,  # unknown (not recomputed)
            }
            self.grad_tracker.history.append(grad_stats)
        else:
            grad_stats = self.grad_tracker.update(model)

        # -------------------------
        # GLOBAL LOSS
        # -------------------------

        loss_val = _to_float(total_loss)
        ema_val = sum(ema_losses.values()) / max(len(ema_losses), 1)
        var_val = sum(v["var"] for v in stats.values()) / max(len(stats), 1)

        # -------------------------
        # SPIKE DETECTION
        # -------------------------

        spike = self.spike_detector.detect(loss_val, ema_val, var_val)

        # -------------------------
        # FAILURE CLASSIFICATION
        # -------------------------

        failure = self.classifier.classify(
            loss_val,
            ema_val,
            grad_stats["total_norm"],
            logits,
        )

        severity = self._severity_score(failure, grad_stats["total_norm"])

        if failure != "normal":
            self.memory.store(
                failure,
                {
                    "loss": loss_val,
                    "grad_norm": grad_stats["total_norm"],
                },
            )
            self._failure_counter[failure] += 1

        # -------------------------
        # THROUGHPUT TRACKING
        # -------------------------

        if throughput is not None:
            self._throughput_hist.append(throughput)

        throughput_trend = self._compute_throughput_trend()

        # -------------------------
        # GRADNORM (OPTIONAL)
        # -------------------------

        gradnorm_weights = None

        if self.gradnorm and shared_params is not None:
            gradnorm_weights = self.gradnorm.compute(losses, shared_params)

        # -------------------------
        # DECISION SIGNALS (NEW )
        # -------------------------

        action = self._decide_action(
            failure,
            spike,
            severity,
            throughput_trend,
        )

        # -------------------------
        # OUTPUT
        # -------------------------

        return {
            "debug/loss": loss_val,
            "debug/ema_loss": ema_val,
            "debug/grad_norm": grad_stats["total_norm"],
            "debug/spike": spike,
            "debug/failure": failure,
            "debug/severity": severity,
            "debug/action": action,
            "debug/throughput_trend": throughput_trend,
            "debug/gradnorm_weights": gradnorm_weights,
        }

    # =====================================================
    # SEVERITY
    # =====================================================

    def _severity_score(self, failure: str, grad_norm: float) -> float:

        if failure == "nan_loss":
            return 1.0

        if failure == "exploding_gradients":
            return min(1.0, grad_norm / 1000)

        if failure == "vanishing_gradients":
            return 0.6

        if failure == "logit_collapse":
            return 0.7

        if failure == "loss_spike":
            return 0.5

        return 0.0

    # =====================================================
    # THROUGHPUT TREND
    # =====================================================

    def _compute_throughput_trend(self) -> float:

        if len(self._throughput_hist) < 5:
            return 0.0

        recent = list(self._throughput_hist)

        return (recent[-1] - recent[0]) / (abs(recent[0]) + 1e-6)

    # =====================================================
    # DECISION LOGIC (CRITICAL )
    # =====================================================

    def _decide_action(
        self,
        failure: str,
        spike: bool,
        severity: float,
        throughput_trend: float,
    ) -> str:

        if failure == "nan_loss":
            return "stop_training"

        if failure == "exploding_gradients":
            return "reduce_lr"

        if spike:
            return "reduce_lr"

        if severity > 0.8:
            return "intervene"

        if throughput_trend < -0.5:
            return "check_dataloader"

        return "none"