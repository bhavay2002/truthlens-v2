from __future__ import annotations

import logging
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader

from src.models.registry.model_factory import build_model
from src.models.optimization.optimizer_factory import build_optimizer
from src.models.optimization.lr_scheduler import build_scheduler
from src.data_processing.dataloader_factory import build_dataloader, DataLoaderConfig
from src.data_processing.dataset_factory import build_dataset

from .training_setup import TrainingSetupConfig
from .training_step import TrainingStep, TrainingStepConfig
from .monitor_engine import MonitoringEngine
from .experiment_tracker import ExperimentTracker
from .task_scheduler import TaskScheduler
from .loss_engine import LossEngine, LossEngineConfig
from .loss_balancer import LossBalancerConfig, plan_for_dataframe
from .evaluation_engine import EvaluationEngine, EvaluationConfig
from .trainer import Trainer

from src.config.task_config import get_task_type, get_output_dim
from src.data_processing.data_contracts import get_contract
from src.utils.seed_utils import set_seed

logger = logging.getLogger(__name__)


# =========================================================
# HELPERS
# =========================================================

def _validate_params(params: Dict[str, Any]):

    required = ["lr", "batch_size", "tokenizer"]

    for k in required:
        if k not in params:
            raise ValueError(f"Missing required param: {k}")


def _resolve_device(params: Dict[str, Any]) -> str:
    if "device" in params:
        return params["device"]
    return "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# FACTORY
# =========================================================

def create_trainer_fn(
    *,
    task: str,
    train_df,
    val_df,
    params: Dict[str, Any],
):

    # =====================================================
    # VALIDATION + SEED
    # =====================================================

    _validate_params(params)

    seed = params.get("seed", 42)
    set_seed(seed)

    device = _resolve_device(params)

    # =====================================================
    # MODEL
    # =====================================================

    model = build_model(
        task=task,
        config=params,
    )

    # GPU-1: move the model to its target device EXACTLY ONCE, BEFORE
    # ``build_optimizer`` runs. This is the single hard requirement: the
    # optimizer captures parameter references by id, so any subsequent
    # ``model.to(...)`` swaps the underlying ``Tensor`` storage and
    # leaves the optimizer pointing at the old (CPU) parameters — which
    # is why ``optimizer.step()`` would raise the classic "expected all
    # tensors to be on the same device" error on the first step under
    # AMP/CUDA. Both ``Trainer.__init__`` and ``TrainingStep.__init__``
    # now only validate the device match (with a loud warning + fallback
    # in-place move) instead of silently re-moving.
    model = model.to(device)

    # =====================================================
    # DATALOADERS  (BUG-1: build_dataloader requires
    # dataset/split/config, not batch_size/shuffle directly)
    # =====================================================

    tokenizer = params["tokenizer"]
    max_length = int(params.get("max_length", 512))

    # =====================================================
    # LOSS-BALANCING PLAN (computed from train split only)
    #
    # This must happen BEFORE ``build_dataset`` so we can pass
    # ``valid_label_indices`` into both the train and val datasets.
    # The plan is the single source of truth for:
    #   * which multilabel columns survive (degenerate ones dropped),
    #   * the per-task class_weights / pos_weight tensors,
    #   * whether to swap CE → FocalLoss for severe imbalance.
    # If anything goes wrong here we fall back to "no plan", which
    # reproduces the old unweighted, full-width behaviour exactly.
    # =====================================================

    task_type = get_task_type(task)
    contract = get_contract(task)

    plan = None
    valid_label_indices: Optional[list] = None
    try:
        balancer_cfg = LossBalancerConfig(
            **(params.get("loss_balancer_config") or {})
        )
        plan = plan_for_dataframe(
            train_df,
            label_columns=list(contract.label_columns),
            task_type=task_type,
            num_classes=get_output_dim(task),
            config=balancer_cfg,
        )
        if task_type == "multilabel" and plan.valid_label_indices is not None:
            valid_label_indices = list(plan.valid_label_indices)
            if plan.dropped_label_indices:
                dropped_names = [
                    contract.label_columns[i]
                    for i in plan.dropped_label_indices
                    if 0 <= i < len(contract.label_columns)
                ]
                logger.warning(
                    "task=%s: dropping %d single-class multilabel column(s) "
                    "from training: %s",
                    task, len(plan.dropped_label_indices), dropped_names,
                )
        logger.info(
            "Loss-balancing plan | task=%s type=%s max_ratio=%.3f "
            "class_weights=%s pos_weight=%s focal=%s notes=%s",
            task, plan.task_type, plan.max_ratio,
            plan.class_weights is not None,
            plan.pos_weight is not None,
            plan.use_focal, plan.notes,
        )
    except Exception as exc:
        logger.warning(
            "Loss-balancing plan unavailable for task=%s (%s); "
            "falling back to unweighted loss / full-width labels.",
            task, exc,
        )

    train_dataset = build_dataset(
        task=task,
        df=train_df,
        tokenizer=tokenizer,
        max_length=max_length,
        valid_label_indices=valid_label_indices,
    )
    val_dataset = build_dataset(
        task=task,
        df=val_df,
        tokenizer=tokenizer,
        max_length=max_length,
        valid_label_indices=valid_label_indices,
    )

    loader_cfg = DataLoaderConfig(
        batch_size=int(params["batch_size"]),
        num_workers=int(params.get("num_workers", -1)),
        pin_memory=bool(params.get("pin_memory", True)),
        use_sampler=bool(params.get("use_sampler", True)),
        drop_last=bool(params.get("drop_last", False)),
    )

    train_loader: DataLoader = build_dataloader(
        task=task,
        dataset=train_dataset,
        df=train_df,
        split="train",
        config=loader_cfg,
        tokenizer=tokenizer,
    )
    val_loader: DataLoader = build_dataloader(
        task=task,
        dataset=val_dataset,
        df=val_df,
        split="val",
        config=loader_cfg,
        tokenizer=tokenizer,
    )

    # PERF-2: Under DDP every rank evaluated the FULL validation set and
    # then ``_sync_scalar`` averaged identical values across ranks — paying
    # an N× wall-time cost for an N-rank pool with zero accuracy gain.
    # Shard validation across ranks with a non-shuffling DistributedSampler
    # (``drop_last=False`` so the tail isn't silently truncated). The
    # streaming metrics now expose ``sync_distributed`` which SUM-reduces
    # raw (numerator, denominator) accumulators — the correct DDP merge for
    # variable-size shards.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        from torch.utils.data.distributed import DistributedSampler

        val_sampler = DistributedSampler(
            val_dataset,
            shuffle=False,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_loader.batch_size,
            sampler=val_sampler,
            collate_fn=val_loader.collate_fn,
            num_workers=val_loader.num_workers,
            pin_memory=val_loader.pin_memory,
            drop_last=False,
        )
        logger.info(
            "Sharded validation loader: world_size=%d (PERF-2)",
            torch.distributed.get_world_size(),
        )

    # =====================================================
    # OPTIMIZER + SCHEDULER  (BUG-7: pass real num_training_steps
    # so the LambdaLR doesn't decay to 0 at the default 1000.)
    # =====================================================

    optimizer = build_optimizer(
        model=model,
        lr=params["lr"],
        weight_decay=params.get("weight_decay", 0.0),
    )

    grad_accum = int(params.get("grad_accum", 1))
    epochs = int(params.get("epochs", params.get("num_epochs", 1)))
    steps_per_epoch = max(1, len(train_loader) // max(1, grad_accum))
    num_training_steps = int(
        params.get("num_training_steps", steps_per_epoch * epochs)
    )
    num_warmup_steps = int(
        params.get(
            "num_warmup_steps",
            max(1, int(0.1 * num_training_steps)),
        )
    )

    scheduler_cfg = {
        **params,
        "num_training_steps": num_training_steps,
        "num_warmup_steps": num_warmup_steps,
    }
    scheduler = build_scheduler(
        optimizer=optimizer,
        config=scheduler_cfg,
    )

    # =====================================================
    # LOSS ENGINE (DYNAMIC )
    # =====================================================

    # The loss-balancing ``plan`` was computed earlier (before
    # ``build_dataset``) so the same source of truth drives both
    # column dropping and loss-level balancing. Translate it into the
    # per-task maps that ``LossEngineConfig`` expects. ``plan`` may
    # be ``None`` if the planner crashed — in that case every map
    # stays empty and the engine falls back to plain unweighted loss.
    class_weights_map: Dict[str, Any] = {}
    pos_weights_map: Dict[str, Any] = {}
    use_focal_map: Dict[str, bool] = {}
    focal_gamma_map: Dict[str, float] = {}
    valid_idx_map: Dict[str, list] = {}
    if plan is not None:
        if plan.class_weights is not None:
            class_weights_map[task] = plan.class_weights
        if plan.pos_weight is not None:
            pos_weights_map[task] = plan.pos_weight
        if plan.use_focal:
            use_focal_map[task] = True
            focal_gamma_map[task] = plan.focal_gamma
    if valid_label_indices is not None:
        # Only meaningful for multilabel; it's harmless to thread for
        # other task types but the router only reads it on that path.
        valid_idx_map[task] = list(valid_label_indices)

    loss_engine = LossEngine(
        LossEngineConfig(
            task_types={task: task_type},
            # LOSS-3: pre-scale static weights by grad_accum so the
            # downstream loss/grad_accum division doesn't shrink them.
            gradient_accumulation_steps=grad_accum,
            class_weights=class_weights_map or None,
            pos_weights=pos_weights_map or None,
            use_focal=use_focal_map or None,
            focal_gamma=focal_gamma_map or None,
            valid_label_indices=valid_idx_map or None,
        )
    )

    # =====================================================
    # MONITORING
    # =====================================================

    monitor = MonitoringEngine(
        params.get("monitor_config")
    )

    # =====================================================
    # TASK SCHEDULER
    # =====================================================

    task_scheduler = TaskScheduler(
        tasks=[task],
        config=params.get("task_scheduler_config"),
    )

    # =====================================================
    # TRACKER
    # =====================================================

    tracker = ExperimentTracker(
        params.get("tracker_config")
    )

    # =====================================================
    # TRAINING STEP
    # =====================================================

    training_step = TrainingStep(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_engine=loss_engine,
        monitor=monitor,
        tracker=tracker,
        task_scheduler=task_scheduler,
        instrumentation=params.get("instrumentation"),
        config=TrainingStepConfig(
            gradient_accumulation_steps=params.get("grad_accum", 1),
            max_grad_norm=params.get("max_grad_norm", 1.0),
            use_mixed_precision=params.get("amp", True),
        ),
        device=device,
    )

    # =====================================================
    # EVALUATION  (BUG-3: EvaluationEngine requires a config)
    # =====================================================

    evaluator = EvaluationEngine(
        EvaluationConfig(
            task_types={task: task_type},
            device=device,
            # EVAL-MULTILABEL-SLICE: pass the same per-task surviving
            # column indices the LossEngine got, so the multilabel
            # evaluator slices the model's full-width logits down to
            # the dataset's reduced label width. Without this the val
            # loop crashes on the first batch with "shape of the mask
            # [B, K_kept] does not match tensor [B, C_full]" once the
            # loss-balancer drops any degenerate column. ``None`` keeps
            # the original full-width behaviour for tasks that don't
            # drop columns. ``valid_idx_map`` is built earlier in this
            # function alongside the LossEngineConfig wiring.
            valid_label_indices=valid_idx_map or None,
        )
    )

    # =====================================================
    # OPTIONAL COMPONENTS
    # =====================================================

    checkpoint = params.get("checkpoint")      # plug your CheckpointEngine
    distributed = params.get("distributed")    # plug DistributedEngine

    # =====================================================
    # TRAINER
    # =====================================================

    setup_cfg = TrainingSetupConfig(
        use_amp=bool(params.get("amp", True)),
        amp_dtype=str(params.get("amp_dtype", "float16")),
        allow_tf32=bool(params.get("allow_tf32", True)),
        use_compile=bool(params.get("use_compile", True)),
        compile_mode=str(params.get("compile_mode", "reduce-overhead")),
        use_gradient_checkpointing=bool(params.get("gradient_checkpointing", True)),
    )

    trainer = Trainer(
        config_path=params.get("config_path", ""),
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_step=training_step,
        evaluator=evaluator,
        checkpoint=checkpoint,
        distributed=distributed,
        tracker=tracker,
        monitor_metric=params.get("monitor_metric", "val_loss"),
        maximize_metric=params.get("maximize_metric", False),
        params_override=params,  # BUG-9: thread Optuna-suggested epochs through
        setup_config=setup_cfg,
    )

    logger.info(
        "Trainer created | task=%s | device=%s | batch_size=%s",
        task,
        device,
        params["batch_size"],
    )

    return trainer