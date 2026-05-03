from __future__ import annotations

import logging
import math
from dataclasses import asdict
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from src.models.config.model_config import ModelConfigLoader

from src.training.training_setup import (
    TrainingSetupConfig,
    setup_runtime,
    optimize_model,
    run_sanity_check,
)

from .training_step import TrainingStep
from .evaluation_engine import EvaluationEngine
from ..models.checkpointing.checkpoint_manager import CheckpointEngine
from .distributed_engine import DistributedEngine
from .experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)


# =========================================================
# TRAINER (PRODUCTION-GRADE)
# =========================================================

class Trainer:

    def __init__(
        self,
        config_path: Optional[str] = None,
        model: torch.nn.Module = None,
        train_loader: DataLoader = None,
        val_loader: Optional[DataLoader] = None,
        training_step: TrainingStep = None,
        evaluator: EvaluationEngine = None,
        checkpoint: Optional[CheckpointEngine] = None,
        distributed: Optional[DistributedEngine] = None,
        tracker: Optional[ExperimentTracker] = None,
        monitor_metric: Optional[str] = None,
        maximize_metric: bool = False,
        params_override: Optional[Dict[str, Any]] = None,
        setup_config: Optional[TrainingSetupConfig] = None,
        log_every_steps: Optional[int] = None,
        checkpoint_every_steps: Optional[int] = None,
    ):

        # -------------------------------------------------
        # CONFIG
        #
        # N-LOW-4: ``config_path`` is now optional. The previous required
        # argument forced every caller (Optuna trials, unit tests,
        # ad-hoc smoke tests) to invent a YAML path even when they had
        # no need for the multitask config — and an empty string would
        # cascade into a confusing ``FileNotFoundError`` deep inside
        # ``ModelConfigLoader``. Skip the load when no path is supplied
        # and let downstream code detect ``self.cfg is None`` if it
        # actually needs config-derived defaults.
        # -------------------------------------------------
        if config_path:
            self.cfg = ModelConfigLoader.load_multitask_config(config_path)
        else:
            self.cfg = None

        self.model = model

        # EDGE-3: an empty train_loader (dataset shorter than batch_size
        # with ``drop_last=True``) used to silently no-op every epoch
        # and finish with ``global_step=0``, masking misconfigured data
        # pipelines.  Surface the misconfiguration loudly at construction.
        if train_loader is None:
            raise ValueError("Trainer requires a non-None train_loader")
        try:
            n_batches = len(train_loader)
        except TypeError:
            # iterable-style loaders don't support len(); skip the check
            n_batches = None
        if n_batches is not None and n_batches == 0:
            raise ValueError(
                "train_loader is empty (0 batches). Likely cause: "
                "dataset shorter than batch_size with drop_last=True, "
                "or a filter that excluded every row."
            )

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.training_step = training_step
        self.evaluator = evaluator
        self.checkpoint = checkpoint
        self.distributed = distributed
        self.tracker = tracker

        # N-MED-5: ``monitor_metric`` previously hardcoded its default to
        # ``"val_loss"`` regardless of task type — silently producing a
        # KeyError-shaped no-op (early stopping never triggers, best
        # checkpoint never saved) when the evaluator emitted, e.g.,
        # ``accuracy`` / ``micro_f1`` for classification heads. Resolve
        # the default lazily (``None`` → ``val_loss``) and warn loudly so
        # the user notices that they should set it explicitly per task.
        if monitor_metric is None:
            logger.warning(
                "N-MED-5: Trainer.monitor_metric defaulting to 'val_loss'. "
                "If your task is classification (multiclass / multilabel / "
                "binary), pass monitor_metric='accuracy' (or 'micro_f1') "
                "and maximize_metric=True so early stopping and best-"
                "checkpoint selection work."
            )
            monitor_metric = "val_loss"
        self.monitor_metric = monitor_metric
        self.maximize_metric = maximize_metric

        # -------------------------------------------------
        # TRAINING SETUP
        #
        # CFG-4: ``TrainingSetupConfig`` is ``frozen=True`` (immutability is
        # the right default — callers can't accidentally mutate runtime
        # precision flags mid-training). Previously the Trainer always
        # constructed a default instance, so callers could not disable e.g.
        # ``run_sanity_check`` for fast Optuna trials. Accept an explicit
        # override here as the documented escape hatch.
        # -------------------------------------------------
        self.setup_cfg = setup_config or TrainingSetupConfig()

        self.device = setup_runtime(self.setup_cfg)

        self.model = optimize_model(self.model, config=self.setup_cfg)

        # GPU-1: the model is moved to its final device EXACTLY ONCE in
        # ``create_trainer_fn`` BEFORE ``build_optimizer`` runs, so that
        # the optimizer captures parameter references already living on
        # the target device. The previous in-place ``self.model.to(self.device)``
        # here was the first of three redundant moves (TrainingStep also
        # did one, DistributedEngine.wrap_model another) and silently
        # broke optimizer state on AMP/CUDA when the model was created on
        # CPU. We now validate device match instead of re-moving — if the
        # caller forgot to move the model first, surface a loud warning
        # rather than papering over a stale-optimizer-state bug.
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = self.device

        # GPU-1b: ``torch.device("cuda") != torch.device("cuda:0")`` returns
        # True even though both refer to the same physical device — the
        # equality check is index-strict. ``setup_runtime`` returns the
        # un-indexed form (``cuda``) while ``model.to("cuda")`` resolves the
        # parameters to ``cuda:0``, so the naive ``!=`` comparison fired a
        # false-positive "stale parameter refs" warning AND triggered an
        # in-place re-move on EVERY trainer construction. That re-move
        # happens AFTER ``build_optimizer`` ran in ``create_trainer_fn``,
        # which is the textbook way to break optimizer state and produce
        # the "Expected all tensors to be on the same device, but found
        # cuda:0 and cpu" error on the first ``optimizer.step()`` (most
        # visible on the narrative multilabel head, where multiple loss
        # buffers + multi-class heads multiply the chance of a mixed-
        # device graph). Compare on type + index, treating ``index is
        # None`` as "any index of this type" — same fix already applied
        # in ``TrainingStep.__init__``.
        def _same_device(a: torch.device, b: torch.device) -> bool:
            if a.type != b.type:
                return False
            if a.index is None or b.index is None:
                return True
            return a.index == b.index

        if not _same_device(model_device, self.device):
            logger.warning(
                "GPU-1: Trainer received model on %s but expected %s; "
                "in-place moving (optimizer parameter refs may be stale). "
                "Move the model BEFORE build_optimizer in create_trainer_fn.",
                model_device,
                self.device,
            )
            self.model.to(self.device)

        # -------------------------------------------------
        # TRAINING PARAMS  (BUG-9: honour params["epochs"] from
        # Optuna / hyperparameter tuning if supplied; fall back to
        # the YAML config otherwise.)
        # -------------------------------------------------
        params_override = params_override or {}
        # N-LOW-4: when ``config_path`` was omitted there is no ``self.cfg``,
        # so fall back to sane defaults rather than dereferencing ``None``.
        cfg_epochs = (
            self.cfg.training.num_epochs if self.cfg is not None else 1
        )
        cfg_patience = (
            self.cfg.training.early_stopping_patience
            if self.cfg is not None
            else 3
        )
        self.epochs = int(params_override.get("epochs", cfg_epochs))
        self.early_patience = int(
            params_override.get("early_stopping_patience", cfg_patience)
        )
        # MIN-EPOCH-EARLY-STOPPING: ``min_epochs`` is the floor below
        # which the early-stopping break is suppressed even if the
        # patience counter has been exceeded. Defaults to 1 (no floor)
        # so callers that don't set it preserve the legacy behaviour.
        # Clamped to ``self.epochs`` to avoid the pathological
        # ``min_epochs > epochs`` configuration silently turning into
        # "early stopping disabled".
        self.min_epochs = max(
            1, int(params_override.get("min_epochs", 1))
        )
        if self.min_epochs > self.epochs:
            logger.warning(
                "Trainer: min_epochs=%d exceeds epochs=%d; clamping "
                "min_epochs to %d (early stopping is effectively "
                "disabled for this run).",
                self.min_epochs, self.epochs, self.epochs,
            )
            self.min_epochs = self.epochs

        # WEIGHTED-COMPOSITE-METRIC: per-task weights used to synthesise
        # the ``weighted_composite_score`` key injected into ``val_metrics``
        # at the end of each evaluation pass (see ``_inject_weighted_composite``
        # below). When empty / not provided the helper is a no-op and
        # only the raw evaluator keys (``val_loss``, ``{task}_score``)
        # are visible to early stopping & checkpointing.
        raw_weights = params_override.get("task_weights") or {}
        self.task_weights: Dict[str, float] = {
            str(k): float(v) for k, v in raw_weights.items()
        }

        # MONITOR-WEIGHTS: separate weights driving the
        # ``weighted_composite_score`` early-stopping metric, decoupled
        # from the LOSS ``task_weights`` above. The two answer different
        # questions: ``task_weights`` balances *gradient flow into the
        # shared encoder* (under-train heads get upweighted), while
        # ``monitor_task_weights`` answers "which heads should drive
        # the early-stopping decision?". Mixing them means a
        # rebalanced loss weight (e.g. ideology 0.7 → 1.1 because it's
        # lagging) silently shifts the early-stopping target.
        # Falls back to ``self.task_weights`` so callers that don't set
        # the key behave exactly as before. ``_inject_weighted_composite``
        # consumes ``self.monitor_task_weights`` (NOT ``self.task_weights``)
        # below.
        raw_monitor_weights = params_override.get("monitor_task_weights")
        if raw_monitor_weights is None:
            self.monitor_task_weights: Dict[str, float] = dict(self.task_weights)
        else:
            self.monitor_task_weights = {
                str(k): float(v) for k, v in raw_monitor_weights.items()
            }

        # MIN-DELTA: minimum *absolute* change in the monitored metric
        # required to count as an improvement. Without this, multi-task
        # validation noise oscillates ±0.001 around the running best
        # forever — every up-tick resets ``no_improve_epochs`` to 0,
        # every down-tick increments it by 1, so the patience counter
        # never reaches its limit and early stopping never fires.
        # Defaults to 0.0 (legacy behaviour) so single-task callers and
        # paths that don't set it are unaffected.
        self.min_delta = max(
            0.0, float(params_override.get("early_stopping_min_delta", 0.0))
        )

        self.global_step = 0
        self._epoch = 0
        self.best_metric = None
        self.no_improve_epochs = 0

        # -------------------------------------------------
        # LOGGING / CHECKPOINT CADENCE
        #
        # CFG-3: The previous implementation hardcoded 50 (log) and 500
        # (checkpoint) inside ``_train_epoch``. Both are now driven by
        # ctor args (with the same defaults) so:
        #   * Optuna / fast smoke tests can log every step
        #     (``log_every_steps=1``), and
        #   * long production runs can dial checkpoint cadence up
        #     (e.g. every 5000 steps) without code edits.
        # ``params_override`` also accepts the same keys so a YAML /
        # tuning config can drive both without touching the Trainer call
        # site.
        # -------------------------------------------------
        self.log_every_steps = int(
            log_every_steps
            if log_every_steps is not None
            else params_override.get("log_every_steps", 50)
        )
        self.checkpoint_every_steps = int(
            checkpoint_every_steps
            if checkpoint_every_steps is not None
            else params_override.get("checkpoint_every_steps", 500)
        )

        if self.log_every_steps <= 0:
            raise ValueError("log_every_steps must be > 0")
        if self.checkpoint_every_steps <= 0:
            raise ValueError("checkpoint_every_steps must be > 0")

        # -------------------------------------------------
        # DISTRIBUTED
        # -------------------------------------------------
        if self.distributed:
            self.distributed.initialize()
            self.model = self.distributed.wrap_model(self.model)

        # -------------------------------------------------
        # LOG CONFIG
        # -------------------------------------------------
        # N-LOW-4: ``asdict(self.cfg)`` would crash when the optional
        # config_path was not supplied; only log when we actually loaded one.
        if self.tracker and self._is_main() and self.cfg is not None:
            self.tracker.log_params(asdict(self.cfg))

        logger.info("Trainer initialized (PRODUCTION-GRADE)")

    # =====================================================
    # TRAIN ENTRY
    # =====================================================

    def train(self):

        # N-LOW-6: Wrap the whole training body in try/finally so the
        # tracker run is finalised and the distributed process group is
        # destroyed EVEN if training raises (sanity-check failure,
        # OOM during forward, KeyboardInterrupt, ...). Previously a
        # mid-training exception left:
        #   - the MLflow run hanging (so the next run picked up the
        #     orphaned active-run handle and silently logged into it),
        #   - the W&B process unfinalised (no upload, dropped artifacts),
        #   - the NCCL/GLOO process group alive (next run's
        #     init_process_group raised "already initialized").
        try:

            # 🔥 SANITY CHECK (CRITICAL)
            if self.setup_cfg.run_sanity_check:
                self._run_sanity_check()

            for epoch in range(self.epochs):

                self._epoch = epoch

                if self._is_main():
                    logger.info("Epoch %d/%d", epoch + 1, self.epochs)

                # DDP sampler sync
                if self.distributed and self.distributed.initialized:
                    if hasattr(self.train_loader.sampler, "set_epoch"):
                        self.train_loader.sampler.set_epoch(epoch)

                self._train_epoch()

                # -------------------------
                # VALIDATION
                # -------------------------
                if self.val_loader:

                    if self.distributed and self.distributed.initialized:
                        self.distributed.barrier()

                    val_metrics = self.evaluate()

                    # WEIGHTED-COMPOSITE-METRIC: enrich the evaluator
                    # output with a single task-balanced score before any
                    # downstream consumer (early stopping, checkpoint,
                    # tracker) reads it. Mutates ``val_metrics`` in-place
                    # so the new key flows through unchanged.
                    self._inject_weighted_composite(val_metrics)

                    metric_value = val_metrics.get(self.monitor_metric)

                    if metric_value is not None:
                        self._update_early_stopping(metric_value)

                    # LOGGING
                    if self.tracker and self._is_main():
                        self.tracker.log_metrics(val_metrics, step=self.global_step)

                    # CHECKPOINT
                    if self.checkpoint and self._is_main():
                        self._save_checkpoint(val_metrics)

                    # EARLY STOP
                    # MIN-EPOCH-EARLY-STOPPING: suppress the break until
                    # the trainer has completed at least ``min_epochs``
                    # epochs. ``epoch`` is 0-indexed in the surrounding
                    # ``for epoch in range(self.epochs)`` loop, so
                    # ``epoch + 1`` is the human-friendly "epochs
                    # completed" count and the comparison reads
                    # naturally ("only stop after we've finished epoch
                    # min_epochs or later").
                    if (
                        self.no_improve_epochs >= self.early_patience
                        and (epoch + 1) >= self.min_epochs
                    ):
                        if self._is_main():
                            logger.warning(
                                "Early stopping triggered "
                                "(epoch=%d, no_improve=%d, "
                                "patience=%d, min_epochs=%d)",
                                epoch + 1,
                                self.no_improve_epochs,
                                self.early_patience,
                                self.min_epochs,
                            )
                        break

        finally:
            # -------------------------------------------------
            # CLEANUP — runs even on exception (N-LOW-6)
            # -------------------------------------------------
            if self.tracker and self._is_main():
                try:
                    self.tracker.finish()
                except Exception:
                    logger.exception("Tracker finalisation failed")

            if self.distributed:
                try:
                    self.distributed.cleanup()
                except Exception:
                    logger.exception("Distributed cleanup failed")

    # =====================================================
    # TRAIN EPOCH
    # =====================================================

    def _train_epoch(self):

        for batch in self.train_loader:

            outputs = self.training_step.run(batch, self.global_step)

            self.global_step += 1

            # Skip failed step
            if outputs.get("skipped"):
                continue

            # -------------------------
            # LOGGING
            # -------------------------
            if self.global_step % self.log_every_steps == 0 and self._is_main():

                log_data = {
                    "train/loss": float(outputs.get("raw_loss", 0.0)),
                    "train/grad_norm": outputs.get("grad_norm"),
                    "train/throughput": outputs.get("throughput"),
                }

                logger.info("Step %d | %s", self.global_step, log_data)

                if self.tracker:
                    self.tracker.log_metrics(log_data, step=self.global_step)

            # -------------------------
            # CHECKPOINT  (BUG-4: persist optimizer/scheduler/scaler
            # state so resume restores momentum, LR-step counter and
            # AMP loss-scale.)
            # -------------------------
            if (
                self.checkpoint
                and self.global_step % self.checkpoint_every_steps == 0
                and self._is_main()
            ):
                self.checkpoint.save(
                    step=self.global_step,
                    epoch=self._epoch,
                    model=self._unwrap_model(),
                    optimizer=getattr(self.training_step, "optimizer", None),
                    scheduler=getattr(self.training_step, "scheduler", None),
                    scaler=getattr(self.training_step, "scaler", None),
                )

    # =====================================================
    # SANITY CHECK
    # =====================================================

    def _run_sanity_check(self):

        if not self.train_loader:
            return

        logger.info("Running sanity check...")

        batch = next(iter(self.train_loader))

        run_sanity_check(
            model=self._unwrap_model(),
            batch=batch,
            training_step=self.training_step,
            device=self.device,
        )

        logger.info("Sanity check passed")

    # =====================================================
    # EARLY STOPPING
    # =====================================================

    def _update_early_stopping(self, metric_value: float):

        improved = False

        # MIN-DELTA: an "improvement" must beat the previous best by at
        # least ``self.min_delta`` in the right direction. With
        # min_delta == 0 this reduces to the legacy strict-comparison
        # behaviour. With min_delta > 0 (recommended for noisy multitask
        # validation), tiny ±noise oscillations no longer reset the
        # patience counter, which is what allows early stopping to
        # actually fire on a real plateau.
        if self.best_metric is None:
            improved = True
        elif self.maximize_metric:
            improved = metric_value > self.best_metric + self.min_delta
        else:
            improved = metric_value < self.best_metric - self.min_delta

        if improved:
            self.best_metric = metric_value
            self.no_improve_epochs = 0
        else:
            self.no_improve_epochs += 1

    # =====================================================
    # CHECKPOINT LOGIC
    # =====================================================

    def _save_checkpoint(self, metrics: Dict[str, Any]):

        # BUG-4: persist optimizer / scheduler / scaler state so resume
        # restores momentum, LR-step counter, and AMP loss-scale.
        common_kwargs = dict(
            model=self._unwrap_model(),
            optimizer=getattr(self.training_step, "optimizer", None),
            scheduler=getattr(self.training_step, "scheduler", None),
            scaler=getattr(self.training_step, "scaler", None),
            epoch=self._epoch,
            metrics=metrics,
        )

        # epoch checkpoint
        self.checkpoint.save(
            step=self.global_step,
            **common_kwargs,
        )

        metric_value = metrics.get(self.monitor_metric)

        if metric_value is None:
            return

        # best checkpoint — uses CheckpointEngine's save_best mechanism
        if (
            self.best_metric is not None
            and metric_value == self.best_metric
        ):
            self.checkpoint.save(
                step=self.global_step,
                save_best=True,
                metric_name=self.monitor_metric,
                mode="max" if self.maximize_metric else "min",
                **common_kwargs,
            )

    # =====================================================
    # EVALUATION
    # =====================================================

    def evaluate(self) -> Dict[str, Any]:

        if not self.val_loader:
            return {}

        model = self._unwrap_model()
        results = self.evaluator.evaluate(model, self.val_loader)

        if self._is_main():
            logger.info("Validation: %s", results)

        return results

    # =====================================================
    # HELPERS
    # =====================================================

    def _unwrap_model(self):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.model.module
        return self.model

    # =====================================================
    # WEIGHTED-COMPOSITE-METRIC
    # =====================================================

    def _inject_weighted_composite(self, val_metrics: Dict[str, Any]) -> None:
        """Add ``weighted_composite_score`` to ``val_metrics`` in place.

        Computed as the weighted average of the per-task ``{task}_score``
        values emitted by ``EvaluationEngine``, using
        ``self.monitor_task_weights`` (which falls back to
        ``self.task_weights`` when the caller didn't supply a separate
        monitor mapping). The result is normalised by the sum of
        weights of the tasks that *actually* produced a score this run,
        so adding / removing a task from the eval set doesn't silently
        rescale the metric.

        No-ops when ``self.monitor_task_weights`` is empty (e.g.
        single-task training, or callers that didn't forward any
        weights), so the legacy behaviour is preserved for
        non-multitask paths.

        MONITOR-WEIGHTS: switched from ``self.task_weights`` (loss
        multiplier) to ``self.monitor_task_weights`` (early-stopping
        signal) so a rebalanced loss weight (e.g. ``ideology 0.7 →
        1.1`` because it's lagging) doesn't silently shift the
        early-stopping target.

        Why this matters: ``val_loss`` on a multitask run is dominated
        by the easy / large-dataset heads (``narrative``, ``propaganda``)
        and stays "improving" by tiny amounts long after the hard heads
        (``ideology``, ``narrative_frame``) have flatlined. A weighted
        score over per-task validation scores tracks the
        *task-balanced* signal that actually matches the training
        objective, so early stopping fires when the *whole system*
        stops improving — not when the loss curve looks busy.
        """
        if not self.monitor_task_weights:
            return

        total_weighted = 0.0
        total_weight = 0.0
        contributing: list[str] = []

        for task, weight in self.monitor_task_weights.items():
            key = f"{task}_score"
            score = val_metrics.get(key)
            # Skip tasks that didn't emit a score this eval pass (task
            # absent from val loader, metric init failure, etc.) AND
            # non-finite values (NaN/inf from a degenerate metric run)
            # rather than letting them poison the composite.
            if score is None:
                continue
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score_f):
                continue
            total_weighted += score_f * weight
            total_weight += weight
            contributing.append(task)

        if total_weight <= 0.0:
            # Nothing to compose — leave val_metrics untouched rather
            # than fabricate a 0.0 that would tank early stopping.
            return

        composite = total_weighted / total_weight
        val_metrics["weighted_composite_score"] = composite

        if self._is_main():
            logger.debug(
                "weighted_composite_score=%.4f over %d tasks: %s",
                composite, len(contributing), contributing,
            )

    def _is_main(self):
        return not self.distributed or self.distributed.is_main_process()

# =========================================================
# COMPAT: lightweight TrainerConfig dataclass
# =========================================================

from dataclasses import dataclass as _dataclass, field as _field
from typing import Any as _Any, Dict as _Dict, Optional as _Optional


@_dataclass
class TrainerConfig:
    config_path: str = ""
    monitor_metric: str = "val_loss"
    maximize_metric: bool = False

    # CFG-3: explicit cadence knobs (mirror the Trainer.__init__ kwargs).
    log_every_steps: int = 50
    checkpoint_every_steps: int = 500

    # CFG-4: explicit setup-config override (mirror Trainer.__init__).
    setup_config: _Optional["TrainingSetupConfig"] = None

    extras: _Dict[str, _Any] = _field(default_factory=dict)
