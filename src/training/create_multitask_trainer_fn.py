"""Production-grade multi-task trainer factory.

Counterpart to ``src.training.create_trainer_fn`` (single-task).

The single-task factory builds one Trainer per task and so trains six
independent encoders for the six TruthLens heads — wasting both compute
(every encoder forward pass is reused exactly once) and generalisation
(no shared representation across the heads). This factory wires up the
*intended* multi-task topology that ``MultiTaskTruthLensModel`` was
designed for:

    MultiTaskLoader  ──►  weighted task sampling
                                │
                                ▼
                    MultiTaskTruthLensModel   (single shared encoder
                                │              + per-task heads)
                                ▼
                          LossEngine          (weighted per-task loss
                                │              + EMA normalizer / coverage)
                                ▼
                          Optimizer / TrainingStep
                                │
                                ▼
                           Trainer.train()

Key contract differences vs ``create_trainer_fn``
-------------------------------------------------
* The model is instantiated **once** as a ``MultiTaskTruthLensModel``;
  every per-task head sits behind a single shared encoder. No per-task
  ``build_model`` calls — they would each instantiate a fresh encoder.

* Per-task DataLoaders are wrapped in ``MultiTaskLoader`` which:
  - yields single-task batches (one task per step, full batch from
    that task's per-task loader),
  - rewraps ``batch["labels"]`` from ``Tensor`` into ``{task: Tensor}``
    so it satisfies the ``MultiTaskLoss`` dict contract,
  - samples tasks by ``task_weights`` (training) / round-robin
    (validation).

* The ``LossEngine`` is built with the FULL ``task_types`` map — not a
  single-entry dict. This activates the multi-task code paths that
  ``LossEngine.__init__`` force-disables for the single-task case
  (EMA normalizer, coverage tracker, ``normalization="active"``,
  optional GradNorm/Uncertainty balancer).

* The ``TaskScheduler`` here is the *loss-driven* scheduler used by the
  training step for adaptive task EMAs / instrumentation — the
  *batch-level* task selection happens inside ``MultiTaskLoader``. Both
  honour ``settings.task_weights``.

Settings contract
-----------------
``settings`` is the attribute-style namespace produced by
``src.utils.settings.load_settings()`` (an ``AttrDict`` over
``config/config.yaml``). The factory reads:

  - ``settings.model``                    — passed through to
                                            ``MultiTaskTruthLensConfig``.
  - ``settings.training.epochs``
  - ``settings.training.lr`` (or ``settings.optimizer.lr``)
  - ``settings.training.weight_decay`` (or ``settings.optimizer.weight_decay``)
  - ``settings.training.use_amp`` (or ``settings.precision.use_amp``)
  - ``settings.training.max_grad_norm``
  - ``settings.training.gradient_accumulation_steps``
  - ``settings.training.batch_size`` (or ``settings.data.batch_size``)
  - ``settings.training.num_workers`` (or ``settings.data.num_workers``)
  - ``settings.training.device`` (default: ``cuda`` if available else ``cpu``)
  - ``settings.task_weights``             — per-task float dict.
  - ``settings.config_path``              — accepted but ignored
                                            (kept for back-compat with
                                            older callers). The factory
                                            no longer forwards it to
                                            ``Trainer``; see the
                                            ``MT-FACTORY-NOLEGACY-CFG``
                                            comment near the Trainer
                                            construction for rationale.

``data_bundle`` is the per-task DataFrame map:

    {
        "bias":       {"train": pd.DataFrame, "val": pd.DataFrame},
        "ideology":   {"train": ..., "val": ...},
        ...
    }

Exactly the layout that ``src.data_processing.dataset_factory.build_all_datasets``
already consumes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

import torch

from src.config.task_config import get_all_tasks, get_task_type
from src.data_processing.dataset_factory import build_dataset
from src.data_processing.dataloader_factory import build_dataloader, DataLoaderConfig
from src.data_processing.multitask_loader import MultiTaskLoader

from src.models.multitask.multitask_truthlens_model import (
    MultiTaskTruthLensModel,
    MultiTaskTruthLensConfig,
)
from src.models.optimization.optimizer_factory import build_optimizer

from src.training.evaluation_engine import EvaluationConfig, EvaluationEngine
from src.training.loss_engine import LossEngine, LossEngineConfig
from src.training.monitor_engine import MonitoringConfig, MonitoringEngine
from src.training.task_scheduler import TaskScheduler, TaskSchedulerConfig
from src.training.trainer import Trainer
from src.training.training_setup import TrainingSetupConfig
from src.training.training_step import TrainingStep, TrainingStepConfig

from src.utils.seed_utils import set_seed

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG HELPERS
# =========================================================

def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Attribute / dict / Mapping fallback accessor.

    Used to read from ``settings`` (AttrDict), nested dicts, or hybrid
    objects without a separate code path per shape.
    """
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _resolve_device(settings: Any) -> torch.device:
    explicit = _get(_get(settings, "training"), "device") or _get(settings, "device")
    if explicit and explicit not in ("auto", None):
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_lr(settings: Any) -> float:
    # MT-FACTORY: accept BOTH ``training.lr`` (spec layout) and
    # ``optimizer.lr`` (current ``config.yaml`` layout). Fail loudly
    # if neither is set rather than silently defaulting — LR is a
    # safety-critical knob and a typo silently picking up 0.0 / 1.0
    # would be catastrophic.
    training = _get(settings, "training")
    lr = _get(training, "lr")
    if lr is None:
        lr = _get(_get(settings, "optimizer"), "lr")
    if lr is None:
        raise ValueError(
            "settings is missing learning rate "
            "(expected settings.training.lr or settings.optimizer.lr)"
        )
    return float(lr)


def _resolve_weight_decay(settings: Any) -> float:
    training = _get(settings, "training")
    wd = _get(training, "weight_decay")
    if wd is None:
        wd = _get(_get(settings, "optimizer"), "weight_decay", 0.0)
    return float(wd)


def _resolve_use_amp(settings: Any) -> bool:
    training = _get(settings, "training")
    val = _get(training, "use_amp")
    if val is None:
        val = _get(_get(settings, "precision"), "use_amp", True)
    return bool(val)


def _resolve_batch_size(settings: Any) -> int:
    training = _get(settings, "training")
    bs = _get(training, "batch_size")
    if bs is None:
        bs = _get(_get(settings, "data"), "batch_size", 16)
    return int(bs)


def _resolve_num_workers(settings: Any) -> int:
    training = _get(settings, "training")
    nw = _get(training, "num_workers")
    if nw is None:
        nw = _get(_get(settings, "data"), "num_workers", -1)
    return int(nw)


def _resolve_grad_accum(settings: Any) -> int:
    training = _get(settings, "training")
    return int(_get(training, "gradient_accumulation_steps", 1))


def _resolve_max_grad_norm(settings: Any) -> float:
    training = _get(settings, "training")
    return float(_get(training, "max_grad_norm", 1.0))


def _resolve_epochs(settings: Any) -> int:
    training = _get(settings, "training")
    return int(_get(training, "epochs") or _get(training, "num_epochs") or 1)


def _resolve_early_stopping_patience(settings: Any) -> int:
    """Pull early-stopping patience from the YAML training section.

    MT-FACTORY-NOLEGACY-CFG: previously this knob was read by
    ``Trainer.__init__`` from ``self.cfg.training.early_stopping_patience``
    where ``self.cfg`` came from ``ModelConfigLoader.load_multitask_config``
    (a strict, *legacy* dataclass parser whose YAML schema diverged from
    the one this factory uses). Resolve it here from the SAME settings
    block every other knob is read from, and forward via
    ``params_override`` so behavior is preserved without going through
    the legacy loader.
    """
    training = _get(settings, "training")
    return int(_get(training, "early_stopping_patience") or 3)


def _resolve_min_epochs(settings: Any) -> int:
    """MIN-EPOCH-EARLY-STOPPING: minimum epochs to always train.

    Below ``min_epochs`` the trainer ignores the patience counter; above
    it, the standard early-stopping logic kicks in. Defaults to 1 (i.e.
    no minimum floor) when the YAML field is absent, preserving the
    legacy behaviour for callers that haven't opted in.
    """
    training = _get(settings, "training")
    return int(_get(training, "min_epochs") or 1)


def _resolve_spike_lr_scale(settings: Any) -> float:
    """SPIKE-LR-RELAX: factor used by the per-step LR auto-reducer.

    Read from ``training.spike_lr_scale`` (default 0.5 to preserve the
    legacy aggressive halving for callers that don't set it). Lower
    values are more aggressive; values close to 1.0 (e.g. 0.9) make
    the auto-reducer a gentle taper rather than a chainsaw. Setting it
    to exactly 1.0 fully disables the reducer (see SPIKE-LR-DISABLED
    short-circuit in ``TrainingStep._reduce_lr``).
    """
    training = _get(settings, "training")
    val = _get(training, "spike_lr_scale")
    return float(val) if val is not None else 0.5


def _resolve_spike_warn_threshold(settings: Any) -> float:
    """EXPLOSION-WATCHDOG: pre-clip grad-norm threshold for the
    ``Gradient spike detected`` warning emitted by ``TrainingStep.run``.

    Read from ``training.spike_warn_threshold`` (default 100.0 to match
    the audit-driven default in ``TrainingStepConfig``). Set to ``0.0``
    in YAML to disable the warning entirely; raise it to (say) 200.0
    if a particular run mix is intentionally noisier and the warning
    becomes spam.
    """
    training = _get(settings, "training")
    val = _get(training, "spike_warn_threshold")
    return float(val) if val is not None else 100.0


def _resolve_min_delta(settings: Any) -> float:
    """MIN-DELTA: floor for what counts as a validation improvement.

    Read from ``training.early_stopping_min_delta`` (default 0.0 to
    preserve the legacy strict-comparison behaviour). Multitask runs
    that monitor noisy composite metrics should set this to ~2× the
    observed per-epoch noise floor so the patience counter actually
    accumulates on a real plateau.
    """
    training = _get(settings, "training")
    val = _get(training, "early_stopping_min_delta")
    return float(val) if val is not None else 0.0


def _resolve_normalizer_alpha(settings: Any) -> Optional[float]:
    """NORMALIZER-ALPHA-DAMP: EMA decay for the loss normaliser.

    Read from ``loss.normalizer_alpha``. Returns ``None`` when the YAML
    field is absent so downstream code keeps the legacy
    ``EMALossNormalizer`` default (0.1) instead of accidentally
    overriding it.
    """
    loss_section = _get(settings, "loss")
    val = _get(loss_section, "normalizer_alpha") if loss_section is not None else None
    return float(val) if val is not None else None


def _resolve_task_weights(
    settings: Any,
    tasks: list,
) -> Dict[str, float]:
    raw = _get(settings, "task_weights")
    if raw is None:
        return {t: 1.0 for t in tasks}

    # AttrDict behaves like both a namespace and a mapping; normalise
    # to a plain dict so downstream code (LossEngineConfig, scheduler
    # configs) doesn't have to care.
    if isinstance(raw, Mapping):
        weights = {t: float(raw.get(t, 1.0)) for t in tasks}
    else:
        weights = {t: float(getattr(raw, t, 1.0)) for t in tasks}

    return weights


def _resolve_monitor_task_weights(
    settings: Any,
    tasks: list,
    fallback: Dict[str, float],
) -> Dict[str, float]:
    """MONITOR-WEIGHTS: weights used by the Trainer's
    ``weighted_composite_score`` early-stopping metric, decoupled from
    the per-task LOSS weights in ``settings.task_weights``.

    Decoupling matters because the loss weights are tuned to balance
    *gradient flow into the shared encoder* (under-train heads get
    upweighted), while the monitor weights answer "which heads should
    drive the early-stopping decision?" — typically the heads that
    represent the application objective rather than the heads that
    happen to need the most learning signal. Mixing the two means a
    rebalanced loss weight (e.g. ideology 0.7 → 1.1 because it's
    lagging) silently shifts the early-stopping target.

    Reads ``settings.monitor_task_weights`` (a YAML mapping in the
    same shape as ``task_weights``). Tasks not listed there get
    weight 0.0 and are excluded from the composite — this is the
    intended way to opt a head out of the early-stopping signal
    (e.g. ``narrative_frame`` is a research head that doesn't drive
    product decisions). When the YAML key is missing entirely, falls
    back to the loss ``task_weights`` so single-task / legacy callers
    behave exactly as before.
    """
    raw = _get(settings, "monitor_task_weights")
    if raw is None:
        return dict(fallback)

    if isinstance(raw, Mapping):
        weights = {t: float(raw.get(t, 0.0)) for t in tasks}
    else:
        weights = {t: float(getattr(raw, t, 0.0)) for t in tasks}

    # Drop zero-weight tasks so ``_inject_weighted_composite`` doesn't
    # iterate over them at all (it's already idempotent for skipped
    # tasks, but this keeps the debug log clean).
    return {t: w for t, w in weights.items() if w > 0.0}


def _resolve_spike_decay_factor(settings: Any) -> float:
    """EXPLOSION-WATCHDOG-RESPONSE: factor applied by the watchdog's
    ``_reduce_lr`` call when a gradient spike is observed.

    Read from ``training.spike_decay_factor``. Default 0.85 (post-V6
    audit recommendation, relaxed from the previous 0.7 because
    consecutive spikes were collapsing the LR by ~66% in <100 steps
    and stalling learning past epoch 4). Set to 1.0 in YAML to keep
    the warning-only behaviour without the LR decay; values >1.0 are
    short-circuited by ``_reduce_lr`` so the worst case is "no-op",
    not "LR runaway".
    """
    training = _get(settings, "training")
    val = _get(training, "spike_decay_factor")
    return float(val) if val is not None else 0.85


def _resolve_spike_skip_threshold(settings: Any) -> float:
    """EXPLOSION-WATCHDOG-SKIP: pre-clip gradient L2 norm above which
    ``TrainingStep.run`` discards the optimiser step entirely.

    Read from ``training.spike_skip_threshold`` (default 150.0,
    placing the skip threshold 50% above the warn threshold of
    100.0). Set to 0.0 in YAML to disable the skip path and keep the
    legacy "always step, just warn loud" behaviour. Steps in the
    100-150 band trigger warn + LR decay only; steps above 150 are
    discarded outright.
    """
    training = _get(settings, "training")
    val = _get(training, "spike_skip_threshold")
    return float(val) if val is not None else 150.0


def _resolve_grad_scaler_init_scale(settings: Any) -> Optional[float]:
    """AMP-INIT-SCALE-FIX: optional ``GradScaler(init_scale=...)`` value.

    Read from ``precision.grad_scaler_init_scale``. Returns ``None``
    when the YAML key is missing so the torch default (2**16 = 65536)
    is preserved for callers that don't set it. Set to 1024.0 (= 2**10)
    on H100 / fp16 stacks to suppress the early-warmup
    "Gradient overflow detected" warning storm.
    """
    precision = _get(settings, "precision")
    val = _get(precision, "grad_scaler_init_scale")
    return float(val) if val is not None else None


def _build_model_config(settings: Any) -> MultiTaskTruthLensConfig:
    """Map ``settings.model`` → ``MultiTaskTruthLensConfig``.

    Accepts either an already-built ``MultiTaskTruthLensConfig`` (passed
    through verbatim) or an attribute / dict-shaped namespace. Unknown
    keys are dropped with a single warning so a stale YAML field can't
    blow up training.
    """
    model_section = _get(settings, "model")

    if isinstance(model_section, MultiTaskTruthLensConfig):
        return model_section

    if model_section is None:
        return MultiTaskTruthLensConfig()

    if isinstance(model_section, Mapping):
        raw = dict(model_section)
    else:
        raw = {
            k: v
            for k, v in vars(model_section).items()
            if not k.startswith("_")
        }

    valid = set(MultiTaskTruthLensConfig.__dataclass_fields__)
    # Common config.yaml field name is ``encoder``; the dataclass field
    # is ``model_name``. Translate so the YAML doesn't need to use
    # internal names.
    if "model_name" not in raw and "encoder" in raw:
        encoder_val = raw.pop("encoder")
        raw["model_name"] = (
            encoder_val.get("name")
            if isinstance(encoder_val, Mapping)
            else getattr(encoder_val, "name", encoder_val)
            if not isinstance(encoder_val, str)
            else encoder_val
        )

    # MT-FACTORY: ``MultiTaskTruthLensConfig`` is *strict* by design —
    # the dataclass docstring rejects unknown kwargs to make YAML typos
    # surface as load-time errors. But ``config.yaml::model`` legitimately
    # carries a few RUNTIME knobs (compile / checkpointing / flash attn)
    # that belong on ``TrainingSetupConfig`` instead, plus a couple of
    # informational fields (``hidden_dim``) the model derives from the
    # pretrained encoder. Filter those out *silently* so the warning
    # below only fires for genuine typos. The runtime knobs are picked
    # up further down via ``TrainingSetupConfig(use_compile=…,
    # compile_mode=…, use_gradient_checkpointing=…)`` reading the SAME
    # ``settings.model`` block — so dropping them here does NOT silently
    # disable them. ``flash_attention`` is currently a no-op in
    # ``MultiTaskTruthLensModel`` (no attention-impl override wired);
    # silencing the warning avoids the false alarm.
    _RUNTIME_ONLY_KEYS = {
        "torch_compile",       # → TrainingSetupConfig.use_compile
        "compile_mode",        # → TrainingSetupConfig.compile_mode
        "gradient_checkpointing",  # → TrainingSetupConfig.use_gradient_checkpointing
        "flash_attention",     # currently no-op (no attention impl override)
        "hidden_dim",          # derived from the pretrained encoder
    }

    unknown = set(raw) - valid - _RUNTIME_ONLY_KEYS
    if unknown:
        logger.warning(
            "create_multitask_trainer_fn: dropping unknown settings.model "
            "fields %s (valid: %s)",
            sorted(unknown),
            sorted(valid),
        )
    kept = {k: raw[k] for k in raw if k in valid}
    return MultiTaskTruthLensConfig(**kept)


def _build_monitoring_config(settings: Any) -> MonitoringConfig:
    """Build a ``MonitoringConfig`` from the YAML ``monitoring:`` block.

    MONITORING-CFG-FIX: previously the factory passed
    ``_get(settings, "monitoring")`` (a raw AttrDict) directly to
    ``MonitoringEngine(...)``. The engine's first line is then
    ``self.config = config or MonitoringConfig()`` — the AttrDict is
    truthy so the dataclass default is bypassed, and the next line
    ``EMA(self.config.throughput_ema_alpha)`` raises
    ``AttributeError: 'AttrDict' has no attribute 'throughput_ema_alpha'``
    because the YAML doesn't define every dataclass field. Map the
    YAML fields explicitly and fall back to dataclass defaults for
    anything missing — the dataclass defaults are the contract, not
    the YAML.
    """
    section = _get(settings, "monitoring")
    if section is None:
        return MonitoringConfig()
    if isinstance(section, MonitoringConfig):
        return section

    valid = set(MonitoringConfig.__dataclass_fields__)
    if isinstance(section, Mapping):
        raw = {k: v for k, v in section.items() if k in valid}
    else:
        raw = {
            k: getattr(section, k)
            for k in valid
            if hasattr(section, k)
        }
    return MonitoringConfig(**raw)


# =========================================================
# FACTORY
# =========================================================

def create_multitask_trainer_fn(
    settings: Any,
    data_bundle: Mapping[str, Mapping[str, Any]],
    *,
    tokenizer: Any,
    enabled_tasks: Optional[list] = None,
    config_path: Optional[str] = None,
) -> Trainer:
    """Build a Trainer that jointly trains every TruthLens task.

    Parameters
    ----------
    settings:
        Attribute-accessible settings namespace (see module docstring
        for the field contract).
    data_bundle:
        ``{task_name: {"train": DataFrame, "val": DataFrame}}``. Every
        task in ``enabled_tasks`` must have BOTH splits present.
    tokenizer:
        HF tokenizer used by ``build_dataset`` to tokenise text. Required
        — the multi-task pipeline must use a single shared tokenizer
        (and thus a single shared vocabulary) since the encoder is
        shared.
    enabled_tasks:
        Optional whitelist; defaults to every task registered in
        ``src.config.task_config``. Order is preserved.
    config_path:
        Optional path passed through to ``Trainer`` (which uses it for
        ``ModelConfigLoader.load_multitask_config``). Falls back to
        ``settings.config_path``; empty string is acceptable when the
        downstream YAML loader isn't strictly required.

    Returns
    -------
    A fully-wired :class:`Trainer` ready for ``trainer.train()``.
    """

    # -----------------------------------------------------
    # 1. SEED + DEVICE
    # -----------------------------------------------------
    seed = int(_get(_get(settings, "project"), "seed", 42))
    set_seed(seed)

    device = _resolve_device(settings)

    # -----------------------------------------------------
    # 2. TASK REGISTRY
    # -----------------------------------------------------
    tasks = list(enabled_tasks) if enabled_tasks else get_all_tasks()
    if not tasks:
        raise ValueError("No tasks resolved — registry is empty.")

    missing = [t for t in tasks if t not in data_bundle]
    if missing:
        raise ValueError(
            f"data_bundle is missing required tasks: {missing}. "
            f"Provided: {list(data_bundle)}"
        )

    for t in tasks:
        for split in ("train", "val"):
            if split not in data_bundle[t]:
                raise ValueError(
                    f"data_bundle[{t!r}] is missing split={split!r} "
                    f"(found: {list(data_bundle[t])})"
                )

    task_types: Dict[str, str] = {t: get_task_type(t) for t in tasks}
    task_weights: Dict[str, float] = _resolve_task_weights(settings, tasks)
    # MONITOR-WEIGHTS: resolved here (not just at the params_override
    # site below) so the value is logged alongside ``task_weights``,
    # making mismatches obvious when reviewing a run from the head of
    # the trainer log. Falls back to ``task_weights`` when the YAML key
    # is missing — see ``_resolve_monitor_task_weights`` for rationale.
    monitor_task_weights: Dict[str, float] = _resolve_monitor_task_weights(
        settings, tasks, fallback=task_weights,
    )

    logger.info(
        "create_multitask_trainer_fn | tasks=%s | task_types=%s | "
        "loss_weights=%s | monitor_weights=%s",
        tasks, task_types, task_weights, monitor_task_weights,
    )

    # -----------------------------------------------------
    # 3. PER-TASK DATASETS + DATALOADERS
    # -----------------------------------------------------
    batch_size = _resolve_batch_size(settings)
    num_workers = _resolve_num_workers(settings)
    max_length = int(_get(_get(settings, "model"), "max_length", 512))

    loader_cfg = DataLoaderConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=bool(_get(_get(settings, "data"), "pin_memory", True)),
        use_sampler=bool(_get(_get(settings, "data"), "use_sampler", True)),
        drop_last=bool(_get(_get(settings, "data"), "drop_last", False)),
    )

    train_loaders = {}
    val_loaders = {}

    for task in tasks:
        train_df = data_bundle[task]["train"]
        val_df = data_bundle[task]["val"]

        train_ds = build_dataset(
            task=task,
            df=train_df,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        val_ds = build_dataset(
            task=task,
            df=val_df,
            tokenizer=tokenizer,
            max_length=max_length,
        )

        train_loaders[task] = build_dataloader(
            task=task,
            dataset=train_ds,
            df=train_df,
            split="train",
            config=loader_cfg,
            tokenizer=tokenizer,
        )
        val_loaders[task] = build_dataloader(
            task=task,
            dataset=val_ds,
            df=val_df,
            split="val",
            config=loader_cfg,
            tokenizer=tokenizer,
        )

    # -----------------------------------------------------
    # 4. MULTI-TASK LOADER (unified iterator)
    # -----------------------------------------------------
    train_loader = MultiTaskLoader(
        dataloaders=train_loaders,
        task_weights=task_weights,
        strategy="weighted",
        seed=seed,
    )
    # MT-FACTORY: validation uses round-robin so every task gets the
    # same number of evaluation batches per epoch — required for the
    # per-task metrics in EvaluationEngine to be comparable across
    # tasks (a weighted val loop would over-sample heavy tasks and
    # bias the validation report).
    val_loader = MultiTaskLoader(
        dataloaders=val_loaders,
        task_weights=None,
        strategy="round_robin",
        seed=seed,
    )

    # -----------------------------------------------------
    # 5. MODEL  (single shared encoder + multi-head)
    # -----------------------------------------------------
    model_cfg = _build_model_config(settings)
    if model_cfg.enabled_tasks is None:
        # MT-FACTORY: pin the head set to the resolved task list so the
        # model's task_logits keys match the loader's task selection.
        # Mismatches here would cause MultiTaskLoss to silently skip
        # tasks (no logits ⇒ no loss contribution).
        model_cfg.enabled_tasks = list(tasks)

    model = MultiTaskTruthLensModel(config=model_cfg)

    # GPU-1 (mirrors single-task factory): move BEFORE optimizer build
    # so the optimizer captures parameters that already live on the
    # target device. See create_trainer_fn for the full rationale.
    model = model.to(device)

    # -----------------------------------------------------
    # 6. LOSS ENGINE  (TRUE multi-task mode)
    # -----------------------------------------------------
    grad_accum = _resolve_grad_accum(settings)

    loss_engine = LossEngine(
        LossEngineConfig(
            task_types=task_types,
            task_weights=task_weights,
            # MT-FACTORY: ``"active"`` divides the summed weighted
            # losses by the number of heads that actually fired this
            # step. Because MultiTaskLoader emits single-task batches,
            # exactly one head fires per step and this resolves to a
            # divide-by-1 — equivalent to ``"sum"`` but advertises the
            # multi-task intent. ``"mean"`` (spec) would divide by the
            # FULL task count even for single-task batches and shrink
            # the gradient by 1/N.
            normalization="active",
            use_normalizer=True,
            use_coverage=True,
            gradient_accumulation_steps=grad_accum,
            # NORMALIZER-ALPHA-DAMP: forward ``loss.normalizer_alpha``
            # so a YAML edit takes effect without code changes.
            normalizer_alpha=_resolve_normalizer_alpha(settings),
        )
    )

    # -----------------------------------------------------
    # 7. TASK SCHEDULER  (loss-EMA tracker for the training step)
    # -----------------------------------------------------
    task_scheduler = TaskScheduler(
        tasks=tasks,
        config=TaskSchedulerConfig(
            strategy="weighted",
            task_weights=task_weights,
            seed=seed,
        ),
    )

    # -----------------------------------------------------
    # 8. OPTIMIZER  (no LR scheduler wired — left to the caller via
    #    Trainer.checkpoint hooks; can be added when settings provides
    #    explicit warmup / total step counts.)
    # -----------------------------------------------------
    optimizer = build_optimizer(
        model=model,
        lr=_resolve_lr(settings),
        weight_decay=_resolve_weight_decay(settings),
    )

    # -----------------------------------------------------
    # 9. MONITORING
    # -----------------------------------------------------
    monitor = MonitoringEngine(_build_monitoring_config(settings))

    # -----------------------------------------------------
    # 10. TRAINING STEP  (AMP + clipping + grad accum)
    # -----------------------------------------------------
    training_step = TrainingStep(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        loss_engine=loss_engine,
        monitor=monitor,
        tracker=None,
        task_scheduler=task_scheduler,
        instrumentation=None,
        config=TrainingStepConfig(
            gradient_accumulation_steps=grad_accum,
            max_grad_norm=_resolve_max_grad_norm(settings),
            use_mixed_precision=_resolve_use_amp(settings),
            # SPIKE-LR-RELAX: forwards ``training.spike_lr_scale`` so a
            # YAML edit (e.g. 0.9) takes effect without code changes.
            spike_lr_scale=_resolve_spike_lr_scale(settings),
            # EXPLOSION-WATCHDOG: forwards ``training.spike_warn_threshold``
            # so the per-step ``Gradient spike detected`` warning is
            # tunable from YAML. Default 100.0 matches
            # ``TrainingStepConfig.spike_warn_threshold``.
            spike_warn_threshold=_resolve_spike_warn_threshold(settings),
            # EXPLOSION-WATCHDOG-RESPONSE: forwards
            # ``training.spike_decay_factor`` so the LR-decay-on-spike
            # action (separate from the legacy ``spike_lr_scale`` /
            # REDUCE_LR pathway above) is tunable from YAML.
            # POST-CONVERGENCE-FIX-V6: default relaxed 0.7 → 0.85
            # because the previous 0.7 compounded too aggressively on
            # consecutive spikes and stalled learning past epoch 4.
            # Matches ``TrainingStepConfig.spike_decay_factor``; set
            # to 1.0 in YAML to keep the warning-only behaviour.
            spike_decay_factor=_resolve_spike_decay_factor(settings),
            # EXPLOSION-WATCHDOG-SKIP: forwards
            # ``training.spike_skip_threshold`` so the
            # discard-the-step-on-extreme-gradient action is tunable
            # from YAML. Default 150.0 matches
            # ``TrainingStepConfig.spike_skip_threshold``; set to 0.0
            # in YAML to disable. Layered ON TOP of the warn + decay
            # pathway above — a step at grad_norm=200 fires both, and
            # this step's update is dropped while the *next* step's
            # LR is decayed.
            spike_skip_threshold=_resolve_spike_skip_threshold(settings),
            # AMP-INIT-SCALE-FIX: forwards
            # ``precision.grad_scaler_init_scale`` so the
            # ``GradScaler(init_scale=...)`` value is tunable from YAML.
            # ``None`` (YAML key absent) preserves the torch default
            # (2**16); 1024.0 (= 2**10) is the audit-recommended value
            # for H100 / fp16 stacks. No-op under bf16 (the scaler is
            # constructed disabled).
            grad_scaler_init_scale=_resolve_grad_scaler_init_scale(settings),
        ),
        device=str(device),
    )

    # -----------------------------------------------------
    # 11. EVALUATION ENGINE
    # -----------------------------------------------------
    evaluator = EvaluationEngine(
        EvaluationConfig(
            task_types=task_types,
            device=str(device),
        )
    )

    # -----------------------------------------------------
    # 12. TRAINER
    # -----------------------------------------------------
    # Build TrainingSetupConfig from precision + model sections of settings
    _precision = _get(settings, "precision")
    _model_sec = _get(settings, "model")
    mt_setup_cfg = TrainingSetupConfig(
        use_amp=bool(_get(_precision, "use_amp", True)),
        amp_dtype=str(_get(_precision, "amp_dtype", "float16")),
        allow_tf32=bool(_get(_precision, "allow_tf32", True)),
        use_compile=bool(_get(_model_sec, "torch_compile", True)),
        compile_mode=str(_get(_model_sec, "compile_mode", "reduce-overhead")),
        use_gradient_checkpointing=bool(_get(_model_sec, "gradient_checkpointing", True)),
    )

    # MT-FACTORY-NOLEGACY-CFG: do NOT forward ``config_path`` to the
    # Trainer. ``Trainer.__init__`` uses it to call the *legacy*
    # ``ModelConfigLoader.load_multitask_config(config_path)``, whose
    # strict per-section dataclass parser expects a different YAML
    # schema than the one this factory reads from (e.g. legacy wants
    # ``training.num_epochs`` while ``config/config.yaml`` declares
    # ``training.epochs``; legacy ``MonitoringConfig`` has
    # ``enable_drift_detection`` while our YAML has ``spike_threshold``).
    # The legacy loader is still used by single-task callers
    # (``model_factory.py``, ``encoder_factory.py``,
    # ``inference/model_loader.py``) so we leave it untouched and
    # simply skip the load here. Trainer already guards every
    # ``self.cfg`` access with ``if self.cfg is not None`` (see N-LOW-4
    # comments in trainer.py); the two values it would have pulled out
    # — ``epochs`` and ``early_stopping_patience`` — are forwarded
    # explicitly via ``params_override`` from the SAME ``settings``
    # block this factory uses for every other knob, so behavior is
    # preserved.
    if config_path or _get(settings, "config_path"):
        logger.debug(
            "create_multitask_trainer_fn: ignoring config_path; epochs and "
            "early_stopping_patience are forwarded from settings.training "
            "directly via params_override."
        )

    trainer = Trainer(
        config_path=None,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_step=training_step,
        evaluator=evaluator,
        monitor_metric=_get(_get(settings, "checkpoint"), "monitor_metric", "val_loss"),
        maximize_metric=(
            _get(_get(settings, "checkpoint"), "mode", "min") == "max"
        ),
        params_override={
            "epochs": _resolve_epochs(settings),
            "early_stopping_patience": _resolve_early_stopping_patience(settings),
            # MIN-EPOCH-EARLY-STOPPING: trainer enforces "always train at
            # least this many epochs before any patience-based stop"
            "min_epochs": _resolve_min_epochs(settings),
            # MIN-DELTA: noise-floor threshold for "improvement"
            "early_stopping_min_delta": _resolve_min_delta(settings),
            # WEIGHTED-COMPOSITE-METRIC: forward the same per-task weights
            # already used for the loss multiplier and the batch sampler
            # so the Trainer can synthesise a single
            # ``weighted_composite_score`` validation metric (weighted
            # average of the per-task ``{task}_score`` values emitted by
            # the evaluator). Set ``checkpoint.monitor_metric`` to that
            # key with ``mode: max`` to drive early stopping by the
            # *task-balanced* signal instead of the easy-task-dominated
            # ``val_loss``.
            "task_weights": task_weights,
            # MONITOR-WEIGHTS: separate weights driving the
            # ``weighted_composite_score`` early-stopping metric.
            # Decoupled from ``task_weights`` (loss multiplier) so a
            # rebalanced loss weight doesn't silently shift the
            # early-stopping target. Trainer's
            # ``_inject_weighted_composite`` consumes this when present
            # and falls back to ``task_weights`` otherwise. See
            # ``_resolve_monitor_task_weights`` for the full rationale.
            "monitor_task_weights": monitor_task_weights,
        },
        setup_config=mt_setup_cfg,
    )

    logger.info(
        "MultiTaskTrainer ready | device=%s | tasks=%d | batch_size=%d | "
        "epochs=%d | grad_accum=%d | amp=%s",
        device, len(tasks), batch_size, _resolve_epochs(settings),
        grad_accum, training_step.use_amp,
    )

    return trainer


__all__ = ["create_multitask_trainer_fn"]
