from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional, Any, Dict

import torch
import torch.nn as nn

from src.training.training_utils import (
    compute_grad_norm,
    get_current_lr,
    move_batch_to_device,
)

# ✅ NEW: observability
from src.monitoring.feature_logger import (
    log_feature_stats,
    log_feature_summary,
)

logger = logging.getLogger(__name__)


# =========================================================
# AMP COMPATIBILITY SHIM
# =========================================================

def get_amp_components(
    device: str,
    enabled: bool,
    *,
    dtype: Optional[torch.dtype] = None,
    scaler_enabled: Optional[bool] = None,
    scaler_init_scale: Optional[float] = None,
):
    """Return ``(scaler, autocast_factory)`` matched to the installed PyTorch.

    Newer torch (≥ 2.3) exposes the device-type-aware ``torch.amp`` API;
    older torch (≤ 2.2) only ships ``torch.cuda.amp``. This helper picks
    the right one at runtime so callers don't have to branch on
    ``torch.__version__``.

    ``dtype`` (fp16 / bf16) is bound into the returned autocast factory.
    ``scaler_enabled`` lets the caller disable the GradScaler independently
    of the autocast flag — needed because the dynamic-loss-scaling /
    overflow-recovery path is fp16-only and must stay off for bf16.

    AMP-INIT-SCALE-FIX: ``scaler_init_scale`` overrides the
    ``GradScaler(init_scale=...)`` value (torch default: 2**16 = 65536).
    Lowering this to e.g. 2**10 = 1024 reduces the count of "Gradient
    overflow detected, step skipped" warnings during the early-training
    warm-up window on hardware with aggressive fp16 ranges (notably
    H100). Pass ``None`` to keep the torch default.
    """
    s_enabled = enabled if scaler_enabled is None else scaler_enabled

    # AMP-INIT-SCALE-FIX: only forward ``init_scale`` when the caller
    # opted in. Avoids forcing 65536 onto callers that depended on the
    # torch default and keeps the bf16 ``s_enabled=False`` path free of
    # an irrelevant kwarg (the scaler is a no-op when disabled, but
    # ``init_scale`` is still validated by the constructor).
    scaler_kwargs: Dict[str, Any] = {"enabled": s_enabled}
    if scaler_init_scale is not None:
        scaler_kwargs["init_scale"] = float(scaler_init_scale)

    # Modern API (PyTorch ≥ 2.3)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(device, **scaler_kwargs)
        if dtype is not None:
            autocast = lambda: torch.amp.autocast(device, enabled=enabled, dtype=dtype)
        else:
            autocast = lambda: torch.amp.autocast(device, enabled=enabled)

    # Legacy API (PyTorch ≤ 2.2)
    else:
        from torch.cuda.amp import GradScaler, autocast as cuda_autocast
        scaler = GradScaler(**scaler_kwargs)
        if dtype is not None:
            autocast = lambda: cuda_autocast(enabled=enabled, dtype=dtype)
        else:
            autocast = lambda: cuda_autocast(enabled=enabled)

    return scaler, autocast


# =========================================================
# CONFIG
# =========================================================

@dataclass
class TrainingStepConfig:
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_mixed_precision: bool = True
    skip_nan_loss: bool = True

    # EXPLOSION-WATCHDOG: pre-clip gradient L2 norm above which ``run()``
    # emits a single ``Gradient spike detected`` warning per offending
    # step. Independent of ``max_grad_norm`` (which silently clips); the
    # warning is the *visibility* signal so post-convergence instability
    # ("174 → 206 → 245 → 290 → 382 → 453") shows up loud in the trainer
    # log instead of being hidden by the clipper. Set to ``0.0`` to
    # disable. Default 100.0 is calibrated against the run that produced
    # this code path: a healthy step in the post-fix-V4 regime sits
    # well below 100, so anything ≥ 100 indicates the loss surface is
    # locally too sharp for the current LR / weight mix and is worth
    # surfacing to the operator.
    spike_warn_threshold: float = 100.0

    # EXPLOSION-WATCHDOG-RESPONSE: factor applied to every optimiser
    # ``param_group['lr']`` (and the scheduler's ``base_lrs`` so the
    # cosine LambdaLR doesn't immediately overwrite the decay) the
    # FIRST time the watchdog fires on a given step. This is the
    # "spike RESPONSE" half of the watchdog — not just detection. The
    # factor is independent from ``spike_lr_scale`` above (which is
    # tied to the legacy instrumentation/monitor REDUCE_LR action and
    # is currently disabled by setting it to 1.0 in YAML). Set to 1.0
    # to disable the response and keep the warning-only behaviour.
    #
    # POST-CONVERGENCE-FIX-V6: relaxed 0.7 → 0.85. The 0.7 (= -30% per
    # spike) factor compounded too aggressively when the watchdog fired
    # on consecutive steps — three spikes in a row collapsed the LR by
    # ~66% in <100 steps, which then prevented the model from learning
    # out of the post-spike plateau and contributed to the "stops
    # improving at epoch 4" symptom. 0.85 (= -15%) is half as severe
    # per spike and lets the model continue learning between events
    # while still applying a meaningful brake. Routes through the same
    # ``_reduce_lr`` helper so per-group LR + scheduler base_lrs stay
    # in sync.
    spike_decay_factor: float = 0.85

    # EXPLOSION-WATCHDOG-SKIP: pre-clip gradient L2 norm above which
    # ``run()`` SKIPS the optimiser / scaler step entirely for the
    # current accumulation boundary. Distinct from
    # ``spike_warn_threshold`` (warning + LR decay only) and from
    # ``max_grad_norm`` (silent clip). Rationale: some spikes are
    # numerically unrecoverable — the per-parameter direction encoded
    # in the gradient is dominated by a few outlier components, so
    # even a clipped step still pushes the parameters in a corrupted
    # direction. Skipping the step entirely (while preserving LR-decay
    # bookkeeping for the *next* step via the warn block above) is
    # cleaner than relying on LR reduction to attenuate the bad update.
    # Set to 0.0 to disable. Default 150.0 places the skip threshold
    # 50% above the warn threshold (100.0), so steps in the 100-150
    # band trigger warn + decay only, while steps above 150 are
    # discarded outright. Implementation calls ``self.scaler.update()``
    # (without ``scaler.step()``) under AMP to drain the unscale_'d
    # per-optimizer state, sets ``scaler_stepped_ok=False`` so the
    # scheduler is also held back this step, and relies on the
    # unconditional ``zero_grad`` at the end of ``run()`` to clear the
    # corrupt gradients before the next backward.
    spike_skip_threshold: float = 150.0

    # CFG-2: factor used by ``_reduce_lr`` when the spike / health detectors
    # (instrumentation engine OR monitor engine) raise ``REDUCE_LR``. Was
    # previously hardcoded as ``0.5`` in two separate sites; centralising
    # here makes it tunable from the config layer (and matches
    # ``LRSchedulerConfig.spike_lr_scale`` semantics).
    spike_lr_scale: float = 0.5

    # CFG-3: AMP autocast dtype.  Previously the autocast call hardcoded
    # the default fp16 path on CUDA; now ``"bf16"`` (better range, no
    # GradScaler needed on Ampere+) and ``"fp16"`` (legacy, requires
    # GradScaler) are both selectable via config.  Anything else falls
    # back to fp16 for backward-compat.  Mapped to ``torch.dtype`` at
    # autocast-call time so the config layer stays string-only.
    #
    # AMP-DTYPE-FIX: default flipped to "float16" to match the
    # inference-time AMP path and keep the project on a single half-
    # precision default. The config layer still accepts explicit bf16
    # for callers that need it.
    amp_dtype: str = "float16"

    # AMP-INIT-SCALE-FIX: ``GradScaler(init_scale=...)`` override.
    # Default ``None`` keeps the torch default (2**16 = 65536) so
    # callers that don't set it preserve the legacy behaviour. Set to
    # 1024.0 (= 2**10) on H100 / fp16 stacks to reduce the
    # "Gradient overflow detected, step skipped" warning rate during
    # the early-training warm-up window. No-op under bf16 (the scaler
    # is constructed with ``enabled=False`` further down). Wired into
    # ``get_amp_components(scaler_init_scale=...)`` at construction
    # time so the kwarg is only forwarded when the caller opts in.
    grad_scaler_init_scale: Optional[float] = None

    # N-MED-2: feature-logging cadence.  Previously hardcoded to 50
    # inside ``TrainingStep.run`` (decoupled from ``log_every_steps``).
    # Set to 0 to disable feature logging entirely.
    feature_log_every_steps: int = 50

    def __post_init__(self) -> None:
        # EDGE-8: ``loss / gradient_accumulation_steps`` would raise
        # ZeroDivisionError on a config typo.  Validate at construction so
        # the failure surfaces at config-load time, not 200 batches in.
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                "gradient_accumulation_steps must be >= 1 "
                f"(got {self.gradient_accumulation_steps})"
            )
        if self.max_grad_norm is not None and self.max_grad_norm < 0:
            raise ValueError(
                f"max_grad_norm must be >= 0 (got {self.max_grad_norm})"
            )
        if self.amp_dtype not in {"fp16", "bf16", "float16", "bfloat16"}:
            raise ValueError(
                f"amp_dtype must be one of fp16/bf16 (got {self.amp_dtype!r})"
            )


# =========================================================
# ACTION ENUM
# =========================================================

class TrainAction:
    NONE = "none"
    REDUCE_LR = "reduce_lr"
    STOP = "stop_training"
    CHECK_DATALOADER = "check_dataloader"


# =========================================================
# CORE ENGINE
# =========================================================

class TrainingStep:

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        loss_engine,
        monitor,
        tracker=None,
        task_scheduler=None,
        instrumentation=None,
        config: TrainingStepConfig = TrainingStepConfig(),
        device: Optional[str] = None,
    ):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_engine = loss_engine
        self.monitor = monitor
        self.tracker = tracker
        self.task_scheduler = task_scheduler
        self.instrumentation = instrumentation
        self.config = config

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # GPU-1: the model is moved to its final device ONCE in
        # ``create_trainer_fn`` (BEFORE ``build_optimizer``), so the
        # optimizer holds parameters that already live on the correct
        # device. The previous ``self.model.to(self.device)`` here was the
        # SECOND of three moves (Trainer.__init__ also did one, and
        # DistributedEngine.wrap_model does a third) — and crucially it
        # happened AFTER the optimizer was constructed, leaving the
        # optimizer with stale parameter references on the original device.
        # That's the classic "expected all tensors to be on the same
        # device" failure at first ``optimizer.step()``. Validate that the
        # model is already on the expected device and surface a clear
        # error if not, instead of silently re-moving it.
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = self.device

        # GPU-1b: ``torch.device("cuda") != torch.device("cuda:0")`` returns
        # True even though both refer to the same physical device — the
        # equality check is index-strict. After ``model.to("cuda")`` PyTorch
        # always resolves params to the indexed form (``cuda:0``), so the
        # naive ``!=`` comparison fired a false-positive "stale parameter
        # refs" warning on EVERY single trainer construction and triggered
        # an unnecessary in-place re-move. Compare on type + index, treating
        # ``index is None`` as "any index of this type".
        def _same_device(a: torch.device, b: torch.device) -> bool:
            if a.type != b.type:
                return False
            if a.index is None or b.index is None:
                return True
            return a.index == b.index

        if not _same_device(model_device, self.device):
            logger.warning(
                "GPU-1: TrainingStep received model on %s but expected %s; "
                "falling back to in-place move (optimizer may hold stale "
                "parameter refs — prefer moving the model BEFORE building "
                "the optimizer in create_trainer_fn).",
                model_device,
                self.device,
            )
            self.model.to(self.device)

        # GPU-3: the loss module wraps ``nn.CrossEntropyLoss(weight=...)`` /
        # ``nn.BCEWithLogitsLoss(pos_weight=...)`` whose ``weight`` /
        # ``pos_weight`` are registered as buffers on the loss module —
        # they only move device when ``.to(device)`` is called on the parent
        # module. The loss-balancing pipeline (``training.loss_balancer``
        # → ``compute_class_weights`` / ``compute_pos_weight``) constructs
        # those tensors on CPU and nobody downstream ever moves them. With
        # the model on CUDA and the loss buffers on CPU, the very first
        # forward pass crashes with the classic "Expected all tensors to be
        # on the same device, but found cuda:0 and cpu". Move the loss
        # module here — the same place we validate the model device — so
        # the buffers track the model. Safe in the no-buffer case (it's a
        # no-op when there are no class/pos weights).
        loss_module = getattr(self.loss_engine, "loss_module", None)
        if isinstance(loss_module, nn.Module):
            loss_module.to(self.device)

        self.use_amp = config.use_mixed_precision and self.device.type == "cuda"
        # CFG-3: resolve amp_dtype string → torch.dtype once.  bf16 does
        # not need a GradScaler (the dynamic-loss-scaling overflow-recovery
        # path is fp16-only) so we still construct the scaler but disable
        # it when bf16 is selected.
        self._amp_dtype = (
            torch.bfloat16
            if config.amp_dtype in ("bf16", "bfloat16")
            else torch.float16
        )
        # GPU/TORCH FIX: route GradScaler + autocast through ``get_amp_components``
        # so the right API (torch.amp ≥ 2.3 vs legacy torch.cuda.amp ≤ 2.2) is
        # picked at runtime instead of crashing on the missing attribute.
        scaler_enabled = self.use_amp and self._amp_dtype == torch.float16
        # AMP-INIT-SCALE-FIX: forward ``grad_scaler_init_scale`` so the
        # ``GradScaler(init_scale=...)`` value is tunable from YAML
        # (precision.grad_scaler_init_scale → factory →
        # TrainingStepConfig.grad_scaler_init_scale → here).
        self.scaler, self._autocast = get_amp_components(
            self.device.type,
            enabled=self.use_amp,
            dtype=self._amp_dtype,
            scaler_enabled=scaler_enabled,
            scaler_init_scale=config.grad_scaler_init_scale,
        )

        self._last_time = time.time()

        logger.info("TrainingStep initialized | AMP=%s", self.use_amp)

    # =====================================================
    # FEATURE HELPER (NEW)
    # =====================================================

    def _tensor_to_feature_dict(self, batch: Dict[str, Any], max_items: int = 50):
        """
        Convert tensor batch into small numeric feature dict for logging.

        PERF-5: Original implementation called ``float(flat[i])`` inside a
        Python loop, which forces *one host-device sync per element* (up to
        ``max_items × num_keys`` syncs per logging step). The new
        implementation slices on-device first and then performs **a single**
        ``.cpu().tolist()`` per tensor — at most one sync per key.
        """
        feature_dict: Dict[str, float] = {}

        for k, v in batch.items():
            if not isinstance(v, torch.Tensor):
                continue
            if not v.dtype.is_floating_point:
                continue

            flat = v.detach().flatten()[:max_items].cpu().tolist()
            feature_dict.update({f"{k}_{i}": float(x) for i, x in enumerate(flat)})

        return feature_dict

    # =====================================================
    # RUN STEP
    # =====================================================

    def run(
        self,
        batch: Dict[str, Any],
        step: int,
        *,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        MT-3: ``dry_run=True`` validates forward + loss + backward without
        mutating any persistent training state — specifically:
          * ``task_scheduler.next_task`` is NOT called (round-robin index
            and adaptive EMA stay frozen)
          * ``optimizer.step`` / ``scaler.step`` / ``scaler.update`` /
            ``scheduler.step`` are skipped (no parameter updates, no LR
            decay tick, no AMP loss-scale advance)
          * ``loss_engine.on_after_backward`` / ``on_step_end`` are skipped
            (balancer counters stay frozen)
          * ``monitor.update`` and ``instrumentation.step`` are skipped
            (EMAs / failure memory / spike detector stay frozen)
        Gradients ARE computed (so the backward path is exercised) and
        then immediately zeroed so the next real step starts clean. This
        is the contract the sanity check needs to be both safety-checking
        AND reproducibility-preserving.
        """

        # EDGE-CASE (section 9): the rest of the run loop unpacks ``batch``
        # via ``model(**batch)`` and indexes ``batch["labels"]`` inside
        # ``LossEngine.compute``. A list/tuple batch (e.g. a default
        # ``DataLoader`` collate that doesn't return a dict) would crash
        # at ``model(**batch)`` with a ``TypeError`` whose message points
        # at the model — masking the real cause (a custom dataset that
        # forgot to return a dict). Surface a clear error here at the
        # contract boundary so the dataset author sees the real issue.
        if not isinstance(batch, dict):
            raise TypeError(
                "TrainingStep expects ``batch`` to be a dict (got "
                f"{type(batch).__name__}). Datasets must return dicts; "
                "fix the dataset / collate_fn rather than reshaping here."
            )

        self.model.train()
        batch = self._move_batch(batch)

        # -------------------------
        # TASK SCHEDULING
        # -------------------------
        #
        # LOSS-2: We DELIBERATELY do not call ``_filter_batch`` here.
        # The original code filtered the labels dict to a single task per
        # step, which (a) wasted the joint-encoder forward pass — the model
        # is multi-task and produces logits for every head regardless — and
        # (b) starved the adaptive task scheduler of all but one task's
        # loss signal, collapsing it to round-robin behaviour in disguise.
        # MultiTaskLoss already masks per-task via its label dict, so the
        # full batch can flow through unchanged.
        #
        # MT-3: in dry-run we DO NOT call ``next_task`` because that
        # advances the round-robin index — the real first training step
        # would then start at index 1 instead of 0, silently desyncing
        # the task schedule from any reproducibility seed.

        task = None
        if self.task_scheduler and not dry_run:
            task = self.task_scheduler.next_task()

        # -------------------------
        # 🔍 FEATURE OBSERVABILITY (NEW)
        # -------------------------
        #
        # N-MED-2: cadence was previously hardcoded to 50, decoupled from
        # the trainer's ``log_every_steps``. That meant feature stats
        # could fire 10× more (or less) often than the train-loss log line,
        # and Optuna fast-trials with ``log_every_steps=1`` still paid the
        # 50-step feature-logging tax. Drive it from the configurable
        # cadence on the step config (``feature_log_every_steps``,
        # default 50 — same as old behaviour).
        feature_cadence = getattr(self.config, "feature_log_every_steps", 50)
        if feature_cadence > 0 and step % feature_cadence == 0:
            try:
                feature_dict = self._tensor_to_feature_dict(batch)

                if feature_dict:
                    log_feature_stats(
                        feature_dict,
                        task=task or "default",
                        step=step,
                    )

                    log_feature_summary(
                        feature_dict,
                        task=task or "default",
                        step=step,
                    )

            except Exception as e:
                logger.warning("Feature logging failed: %s", e)

        # -------------------------
        # FORWARD + LOSS
        # -------------------------

        # EDGE-CASE (section 9, NaN labels): ``LossEngine.compute`` /
        # ``MultiTaskLoss.forward`` raise ``RuntimeError`` on non-finite
        # aggregates. The previous implementation only honoured
        # ``skip_nan_loss`` for the FINAL ``torch.isfinite(total_loss)``
        # check below — meaning NaN labels (which propagate through
        # cross-entropy / BCE before the aggregate is built) escaped the
        # quarantine path and crashed the run despite ``skip_nan_loss=True``.
        # Wrap the whole forward+loss block so the same skip semantics
        # apply uniformly.
        try:
            # GPU/TORCH FIX: ``self._autocast`` was built once in __init__ via
            # ``get_amp_components``, which transparently picks the modern
            # ``torch.amp.autocast`` (PyTorch ≥ 2.3) or the legacy
            # ``torch.cuda.amp.autocast`` (PyTorch ≤ 2.2). CFG-3: the dtype
            # (fp16 vs bf16) was bound there from ``TrainingStepConfig.amp_dtype``.
            with self._autocast():

                # Strip non-tensor metadata that the data_processing
                # ``collate`` injects (currently ``task``) before
                # forward — the single-task model classes have strict
                # ``forward(input_ids, attention_mask, labels)``
                # signatures and reject unknown kwargs.
                model_batch = {
                    k: v for k, v in batch.items()
                    if k not in ("task",)
                }
                outputs = self.model(**model_batch)

                total_loss, task_losses = self.loss_engine.compute(
                    outputs,
                    batch,
                    shared_parameters=self.model.parameters(),
                )
                task_mask = batch.get("task_mask")
                if isinstance(task_mask, dict):
                    masked = {}
                    for t, loss in task_losses.items():
                        m = task_mask.get(t)
                        masked[t] = loss if m is None else loss * m.float().mean()
                    total_loss = sum(masked.values()) / max(1, len(masked))
        except RuntimeError as e:
            if self.config.skip_nan_loss:
                logger.warning(
                    "Skipping step due to RuntimeError in forward/loss "
                    "(likely NaN labels or non-finite logits): %s",
                    e,
                )
                self.optimizer.zero_grad(set_to_none=True)
                return {"loss": None, "skipped": True}
            raise

        # -------------------------
        # LOSS VALIDATION
        # -------------------------

        if not torch.isfinite(total_loss):

            if self.config.skip_nan_loss:
                logger.warning("Skipping step due to NaN loss")
                self.optimizer.zero_grad(set_to_none=True)
                return {"loss": None, "skipped": True}

            raise RuntimeError(f"Non-finite loss: {total_loss.item()}")

        # -------------------------
        # TASK SCHEDULER UPDATE
        # -------------------------
        #
        # MT-3: skip in dry-run so the adaptive EMA isn't poisoned by a
        # one-shot sanity loss before the first real training step.
        # MT-4: ``task_losses`` here is now the RAW per-task loss dict
        # (the second element of MultiTaskLoss.forward's return tuple),
        # which is exactly what the adaptive scheduler's softmax-of-EMA
        # expects — the previous weighted-and-normalized values would
        # have skewed the softmax across tasks with different weights.

        if self.task_scheduler and task_losses and not dry_run:
            self.task_scheduler.update_losses(
                {k: float(v.detach()) for k, v in task_losses.items()}
            )

        # -------------------------
        # BACKWARD
        # -------------------------

        loss = total_loss / self.config.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # -------------------------
        # OPTIMIZER STEP  (BUG-5: unscale BEFORE measuring grad_norm,
        # otherwise the AMP loss-scale (~6.5e4) is baked into every
        # logged grad_norm and instrumentation flags every step as
        # 'exploding gradients'. Also gated on the accumulation
        # boundary so partial micro-batches don't unscale prematurely.)
        # -------------------------

        should_step = (
            (step + 1) % self.config.gradient_accumulation_steps == 0
        )

        grad_norm: Optional[float] = None
        scaler_stepped_ok = True  # tracks whether the scaler actually stepped

        if should_step:

            # AMP-FIX: ``self.scaler.unscale_(optimizer)`` flips the scaler's
            # per-optimizer ``_per_optimizer_states`` to "unscaled" and that
            # flag is ONLY reset by ``self.scaler.update()``. In the
            # ``dry_run=True`` path below we deliberately skip
            # ``scaler.step`` / ``scaler.update`` (sanity check must not
            # mutate persistent training state — see MT-3 above), so calling
            # ``unscale_`` here would leave the optimizer permanently in the
            # "already unscaled" state. The very next REAL training step
            # would then hit ``RuntimeError: unscale_() has already been
            # called on this optimizer since the last update()`` at this
            # exact line. Gate ``unscale_`` on ``not dry_run`` so the
            # sanity-check leaves the scaler state pristine. The grad_norm
            # measured below in dry-run will be the AMP-scaled value, which
            # is acceptable: the sanity check only asserts the backward
            # pass works, not the absolute gradient magnitude.
            if self.use_amp and not dry_run:
                self.scaler.unscale_(self.optimizer)

            # REC-3: ``compute_grad_norm`` and ``clip_grad_norm_`` BOTH
            # iterate every parameter and compute the same total L2 norm
            # — and ``instrumentation.step`` calls ``GradTracker.update``
            # which does it a THIRD time (and after ``zero_grad`` clears
            # the gradients, so it would see zeros). Use ``clip_grad_norm_``
            # alone when clipping is enabled (it returns the pre-clip norm
            # — exactly what we want to log) and fall back to
            # ``compute_grad_norm`` only when clipping is disabled. The
            # resulting ``grad_norm`` is then forwarded to instrumentation
            # via ``cached_grad_norm`` so it doesn't redo the work.
            if self.config.max_grad_norm:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                )
            else:
                grad_norm = compute_grad_norm(self.model)

            # EXPLOSION-WATCHDOG: ``clip_grad_norm_`` returns the *pre-clip*
            # L2 norm, so even when clipping is on we still see the true
            # magnitude of the step the optimiser would otherwise have
            # taken. Surface a warning whenever that pre-clip norm crosses
            # ``spike_warn_threshold`` (default 100.0) so post-convergence
            # instability ("174 → 206 → 245 → 290 → 382 → 453") is loud in
            # the trainer log instead of being hidden by the silent clip.
            # Guarded on ``> 0`` so a zero/disabled threshold is a true
            # no-op, and on ``math.isfinite(grad_norm)`` so NaN / Inf
            # gradients don't double-fire here when the AMP overflow path
            # below already logs them.
            if (
                self.config.spike_warn_threshold > 0.0
                and math.isfinite(grad_norm)
                and grad_norm > self.config.spike_warn_threshold
            ):
                logger.warning(
                    "Gradient spike detected (grad_norm=%.1f > %.1f) at step=%d",
                    grad_norm,
                    self.config.spike_warn_threshold,
                    int(step),
                )
                # EXPLOSION-WATCHDOG-RESPONSE: not just detection — also
                # decay the LR by ``spike_decay_factor`` (default 0.7)
                # the same step the spike is observed. Routed through
                # ``_reduce_lr`` so per-group LR AND scheduler.base_lrs
                # are updated in lockstep (otherwise the cosine LambdaLR
                # would overwrite the per-group decay on its very next
                # ``step()`` call). ``_reduce_lr`` short-circuits on
                # ``factor >= 1.0``, so setting ``spike_decay_factor``
                # to 1.0 disables the response without disabling the
                # warning. Gated on ``not dry_run`` so the sanity check
                # leaves the optimiser pristine. Skipped under AMP fp16
                # overflow (scaler_stepped_ok=False set further below)
                # is N/A here because that flag is only computed inside
                # the ``not dry_run`` block beneath us.
                if not dry_run:
                    self._reduce_lr(factor=self.config.spike_decay_factor)

            # EXPLOSION-WATCHDOG-SKIP: pre-clip ``grad_norm`` above
            # ``spike_skip_threshold`` (default 150.0) → discard the
            # optimiser update entirely. Layered on top of (not in
            # place of) the warn + LR-decay block above: a step at
            # grad_norm=200 fires both — LR decays by
            # ``spike_decay_factor`` for the *next* step, AND this
            # step's corrupt update is dropped on the floor. Setting
            # ``spike_skip_threshold = 0.0`` disables the skip path
            # entirely (legacy behaviour). ``math.isfinite`` guard
            # avoids double-firing with the AMP fp16 overflow log
            # below (which handles non-finite gradients). ``not
            # dry_run`` guard preserves the sanity-check contract
            # ("must not mutate training state").
            spike_skip_step = (
                not dry_run
                and self.config.spike_skip_threshold > 0.0
                and math.isfinite(grad_norm)
                and grad_norm > self.config.spike_skip_threshold
            )
            if spike_skip_step:
                logger.warning(
                    "Extreme gradient detected (grad_norm=%.1f > %.1f) at "
                    "step=%d - SKIPPING optimizer step",
                    grad_norm,
                    self.config.spike_skip_threshold,
                    int(step),
                )
                # Mark the scheduler as held-back so the existing
                # ``if self.scheduler and scaler_stepped_ok:`` gate
                # below also skips advancing the LR schedule for
                # this step. Without this the cosine LambdaLR would
                # tick forward despite the optimiser not stepping —
                # i.e. the schedule would silently desync from the
                # optimiser-step count.
                scaler_stepped_ok = False

            # MT-3: in dry-run validate forward + loss + backward only.
            # Skip the optimizer / scaler / scheduler / balancer mutations
            # so the persistent training state is preserved. Gradients are
            # zeroed below so the first real step starts clean.
            if not dry_run:

                if spike_skip_step:
                    # EXPLOSION-WATCHDOG-SKIP: explicitly DO NOT call
                    # ``scaler.step(optimizer)`` / ``optimizer.step()``
                    # — that's the entire point of this branch. Under
                    # AMP we still need to drain the per-optimizer
                    # ``UNSCALED`` state set by the ``unscale_`` call
                    # earlier in the function, otherwise the *next*
                    # real step hits ``RuntimeError: unscale_() has
                    # already been called on this optimizer since the
                    # last update()``. ``scaler.update()`` resets that
                    # state and also lets the dynamic loss scale
                    # evolve normally (no overflow this step → scale
                    # may grow on its usual cadence).
                    if self.use_amp:
                        self.scaler.update()
                elif self.use_amp:
                    prev_scale = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    if self.scaler.get_scale() < prev_scale:
                        scaler_stepped_ok = False
                        logger.warning("Gradient overflow detected, step skipped")
                else:
                    self.optimizer.step()

                # BUG-6 (partial fix): only advance the scheduler when the
                # optimizer actually stepped, so AMP overflow doesn't drift
                # the LR schedule.
                if self.scheduler and scaler_stepped_ok:
                    try:
                        self.scheduler.step()
                    except TypeError:
                        self.scheduler.step(float(total_loss.detach()))

                self.loss_engine.on_after_backward()
                self.loss_engine.on_step_end()

            self.optimizer.zero_grad(set_to_none=True)

        else:
            # GRAD-LOG-EVERY-STEP: on accumulation micro-batches the
            # ``if should_step`` branch above is skipped and the
            # ``train/grad_norm`` log key would be ``None`` — i.e. for
            # ``grad_accum=4`` we'd be flying blind on 3 of every 4
            # backward passes. Measure the running L2 norm of the
            # currently-accumulated ``.grad`` tensors *without*
            # mutating them (no clip, no zero, no unscale) so the per-
            # step log line always carries a real number. Two caveats
            # worth knowing about the value:
            #   (a) it's the *partial* accumulated norm (after K of N
            #       micro-batches), not the full per-step norm — it
            #       monotonically grows toward the should_step value.
            #   (b) under AMP the ``.grad`` tensors here are still
            #       loss-scaled (we deliberately don't ``unscale_`` on
            #       non-step micro-batches because that flag is reset
            #       only by ``scaler.update`` which runs on step). We
            #       divide by the current scale to put the logged
            #       value on the same scale as the should_step
            #       grad_norm above, so the two are directly
            #       comparable in dashboards.
            partial = compute_grad_norm(self.model)
            if self.use_amp:
                scale = float(self.scaler.get_scale())
                if scale > 0.0:
                    partial = partial / scale
            grad_norm = float(partial)

        # -------------------------
        # THROUGHPUT
        # -------------------------

        now = time.time()
        duration = now - self._last_time
        self._last_time = now

        batch_size = self._infer_batch_size(batch)
        throughput = batch_size / duration if duration > 0 else None

        # -------------------------
        # MONITORING
        # -------------------------
        #
        # MT-3: dry-run skips the monitor entirely so its EMAs / spike
        # detector / health score don't carry sanity-check noise into
        # the first real training step.

        if dry_run:
            monitor_metrics: Dict[str, Any] = {}
        else:
            monitor_metrics = self.monitor.update(
                {"loss": float(total_loss.detach())},
                model=self.model,
                batch_size=batch_size,
            )

        # -------------------------
        # DEBUG ENGINE
        # -------------------------
        #
        # MT-3: skip in dry-run for the same reason as the monitor.
        # REC-3: when we already have ``grad_norm`` from clip_grad_norm_,
        # pass it through as ``cached_grad_norm`` so the instrumentation's
        # GradTracker doesn't re-iterate every parameter to recompute the
        # same value (and on should_step iterations would otherwise see
        # zeroed-out gradients after ``optimizer.zero_grad``).

        debug_info = {}

        if self.instrumentation and not dry_run:
            debug_info = self.instrumentation.step(
                losses=task_losses,
                total_loss=total_loss,
                model=self.model,
                shared_params=self.model.parameters(),
                logits=outputs.get("logits") if isinstance(outputs, dict) else None,
                throughput=throughput,
                cached_grad_norm=grad_norm,
            )

        # -------------------------
        # ACTION HANDLING
        # -------------------------

        action = debug_info.get("debug/action", TrainAction.NONE)

        # LOSS-1: Both the instrumentation engine and the monitor engine can
        # raise REDUCE_LR in the same step. The original code fired
        # ``_reduce_lr`` once per source, halving the LR TWICE on a spike
        # that both detectors caught (a 4× drop instead of the configured
        # 2×). De-duplicate per step.
        lr_reduced_this_step = False

        if action == TrainAction.STOP:
            raise RuntimeError("Training stopped by AutoDebugEngine")

        elif action == TrainAction.REDUCE_LR:
            self._reduce_lr()
            lr_reduced_this_step = True

        elif action == TrainAction.CHECK_DATALOADER:
            logger.warning("Potential dataloader bottleneck detected")

        if (
            not lr_reduced_this_step
            and monitor_metrics.get("monitor/action") == TrainAction.REDUCE_LR
        ):
            self._reduce_lr()

        # -------------------------
        # LOGGING
        # -------------------------

        log_data = {
            "train/loss": float(total_loss.detach()),
            "train/grad_norm": grad_norm,
            "train/lr": get_current_lr(self.optimizer),
            "train/throughput": throughput,
            **monitor_metrics,
            **debug_info,
        }

        # MT-3: dry-run does not pollute the experiment tracker with a
        # one-shot sanity row that would shift step-indexed plots by 1.
        if self.tracker and not dry_run:
            self.tracker.log_metrics(log_data, step=step)

        # -------------------------
        # OUTPUT
        # -------------------------

        return {
            "loss": loss.detach(),
            "raw_loss": total_loss.detach(),
            "task_losses": task_losses,
            "grad_norm": grad_norm,
            "throughput": throughput,
            "skipped": False,
            **monitor_metrics,
            **debug_info,
        }

    # =====================================================
    # UTILS
    # =====================================================

    def _move_batch(self, batch):
        # GPU-2: ``non_blocking=True`` is silently a no-op unless the
        # source tensor is in pinned host memory. The previous inline
        # comprehension passed ``non_blocking=True`` unconditionally,
        # advertising async H2D copies that never actually happened on
        # un-pinned tensors (e.g. CPU-only runs, or any DataLoader built
        # with ``pin_memory=False``). Delegate to the shared utility that
        # gates ``non_blocking`` on the per-tensor ``is_pinned()`` check.
        return move_batch_to_device(batch, self.device, non_blocking=True)

    def _infer_batch_size(self, batch):
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                return v.size(0)
        return 1

    # NOTE: ``_filter_batch`` was removed (LOSS-2). The model is multi-task
    # and produces logits for every head from a single forward pass; the
    # MultiTaskLoss orchestrator masks per-task via the labels dict, so
    # there is no value in pre-filtering the batch.

    def _reduce_lr(self, factor: Optional[float] = None):
        # CFG-2: factor is sourced from ``TrainingStepConfig.spike_lr_scale``
        # by default rather than the previous hardcoded ``0.5``. Callers may
        # still pass an explicit override.
        if factor is None:
            factor = float(self.config.spike_lr_scale)

        # SPIKE-LR-DISABLED: a factor >= 1.0 would either be a no-op
        # (= 1.0) or, worse, an LR *increase* (> 1.0). Short-circuit so
        # the YAML knob ``training.spike_lr_scale: 1.0`` cleanly
        # disables the entire reactive-decay loop — no mutation of the
        # optimiser groups, no scheduler base_lrs rewrite, no warning
        # spam in the logs. Real instability is now expected to be
        # damped by gradient clipping + cosine schedule rather than by
        # this reactive per-step reducer.
        if factor >= 1.0:
            return

        # BUG-6: a LambdaLR (and most functional schedulers) compute
        # ``g["lr"] = base_lr * lambda(step)`` on every ``scheduler.step()``.
        # Mutating only ``g["lr"]`` is therefore overwritten on the very
        # next scheduler step and the spike-recovery action becomes a
        # no-op. We must reduce the scheduler's ``base_lrs`` so the new
        # rate persists across subsequent scheduler steps.
        for g in self.optimizer.param_groups:
            g["lr"] *= factor

        if self.scheduler is not None and hasattr(self.scheduler, "base_lrs"):
            self.scheduler.base_lrs = [
                b * factor for b in self.scheduler.base_lrs
            ]

        logger.warning(
            "LR reduced (factor=%.3f) due to instability", float(factor),
        )