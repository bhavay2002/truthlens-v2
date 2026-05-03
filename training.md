# `src/training/` — Production Documentation

**TruthLens AI · Training Subsystem · Full Reference**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Folder Architecture](#2-folder-architecture)
3. [End-to-End Training Pipeline](#3-end-to-end-training-pipeline)
4. [Training Loop Breakdown](#4-training-loop-breakdown)
5. [File-by-File Deep Dive](#5-file-by-file-deep-dive)
6. [Loss Functions & Optimization](#6-loss-functions--optimization)
7. [Hyperparameters & Configuration Reference](#7-hyperparameters--configuration-reference)
8. [Validation Strategy](#8-validation-strategy)
9. [Checkpointing & Logging](#9-checkpointing--logging)
10. [Config Integration](#10-config-integration)
11. [Optimization & Efficiency](#11-optimization--efficiency)
12. [Failure Handling & Debugging](#12-failure-handling--debugging)
13. [Extensibility Guide](#13-extensibility-guide)
14. [Common Pitfalls & Known Risks](#14-common-pitfalls--known-risks)
15. [Example Usage](#15-example-usage)
16. [Simple Explanation (For Non-Technical Reviewers)](#16-simple-explanation-for-non-technical-reviewers)

---

## 1. Overview

`src/training/` is the **complete training engine** for TruthLens AI — a multi-layer misinformation-detection system. It contains every component required to take a raw dataset and a model configuration, run a full supervised training loop, and produce a checkpointed, evaluated model.

### What It Does

- **Orchestrates** single-task and multi-task training over six detection heads (bias, ideology, propaganda, factuality, emotion, narrative frame) that share one transformer encoder.
- **Manages** the full lifecycle: setup → sanity check → training steps → evaluation → early stopping → checkpointing → cleanup.
- **Adapts** at the loss level: automatically detects class imbalance and switches between plain cross-entropy, class-weighted CE, and focal loss per task per training fold.
- **Instruments** every step: gradient norm tracking, loss spike detection, anomaly classification, throughput monitoring, and experiment tracking to MLflow or Weights & Biases.
- **Scales** to distributed training via PyTorch DistributedDataParallel (DDP) with correct NCCL/gloo fallback and distributed metric reduction.
- **Tunes** itself: Optuna-backed hyperparameter search with cross-validation, per-task direction detection, and multi-objective Pareto-front support.

### Relationship to the Rest of TruthLens

```
config/config.yaml
        │
        ▼
src/training/create_multitask_trainer_fn.py   ← primary entry point for production
src/training/create_trainer_fn.py             ← entry point for single-task / CV / tuning
        │
        ▼
src/training/trainer.py                       ← orchestrates the loop
        │
        ├── training_setup.py                 ← AMP · compile · gradient checkpointing
        ├── training_step.py                  ← forward · backward · clip · step
        ├── loss_engine.py                    ← multi-task loss aggregation
        ├── evaluation_engine.py              ← streaming eval metrics
        ├── monitor_engine.py                 ← real-time health scoring
        ├── experiment_tracker.py             ← MLflow / W&B / none
        ├── distributed_engine.py             ← DDP process group
        └── task_scheduler.py                 ← task sampling strategy
```

---

## 2. Folder Architecture

```
src/training/
├── __init__.py                    Public API surface; lazy Optuna import
├── trainer.py                     Main training orchestrator (Trainer class)
├── training_step.py               Single training step: forward+backward+clip+step
├── training_setup.py              Runtime setup: AMP, compile, grad-checkpoint
├── create_trainer_fn.py           Single-task Trainer factory
├── create_multitask_trainer_fn.py Multi-task Trainer factory (production entry)
├── loss_engine.py                 Multi-task loss aggregation + EMA normalization
├── loss_functions.py              FocalLoss, binary/multiclass/multilabel loss helpers
├── loss_balancer.py               Per-task class imbalance analysis + plan
├── evaluation_engine.py           Streaming validation metrics (Accuracy, F1, MSE)
├── experiment_tracker.py          MLflow / W&B / no-op experiment tracking
├── distributed_engine.py          DDP init, model wrap, sampler, all_reduce
├── instrumentation.py             AutoDebugEngine, LossTracker, GradNorm, SpikeDetector
├── monitor_engine.py              MonitoringEngine: per-step health scoring
├── cross_validation.py            K-Fold / StratifiedKFold CV runner
├── hyperparameter_tuning.py       Optuna-backed HPO (TPE sampler, MedianPruner)
├── task_scheduler.py              Task sampling: round-robin/random/weighted/adaptive
└── training_utils.py              Device helpers, grad norm, throughput, TrainingMetrics
```

### File Roles at a Glance

| File | Primary Responsibility |
|---|---|
| `trainer.py` | Outer epoch loop, early stopping, metric injection, cleanup |
| `training_step.py` | Per-batch forward/backward, AMP, gradient accumulation, spike watchdog |
| `training_setup.py` | One-time runtime optimizations (compile, TF32, anomaly detection, sanity check) |
| `create_trainer_fn.py` | Wires model + data + optimizer + loss + all engines into a single-task Trainer |
| `create_multitask_trainer_fn.py` | Same wiring for the shared-encoder multi-task topology |
| `loss_engine.py` | Weighted multi-task loss with EMA normalization and coverage tracking |
| `loss_functions.py` | FocalLoss, class weight computation, per-type loss dispatchers |
| `loss_balancer.py` | Inspects training labels, decides focal/weighted/unchanged per task |
| `evaluation_engine.py` | GPU-resident streaming metrics with correct DDP all_reduce |
| `experiment_tracker.py` | Unified MLflow/W&B/none facade with distributed-safe logging |
| `distributed_engine.py` | DDP process group management, model wrapping, NCCL→gloo fallback |
| `instrumentation.py` | AutoDebugEngine: EMA loss tracking, grad norm, spike/anomaly classification |
| `monitor_engine.py` | MonitoringEngine: step-level health score, EMA, action policy |
| `cross_validation.py` | K-Fold CV with per-fold seeding, stratification fallback, dashboard |
| `hyperparameter_tuning.py` | Optuna objective, TPE study, multi-objective Pareto front |
| `task_scheduler.py` | Round-robin / adaptive (loss-driven softmax) task selection |
| `training_utils.py` | `get_device`, `move_batch_to_device`, `compute_grad_norm`, `TrainingMetrics` |
| `__init__.py` | Re-exports stable API; Optuna symbols loaded lazily |

---

## 3. End-to-End Training Pipeline

### Single-Task Pipeline

```
Caller
  │
  ├─ create_trainer_fn(task, train_df, val_df, params)
  │       │
  │       ├─ build_model(task, config)          → model
  │       ├─ model.to(device)                   [GPU-1: ONCE, before optimizer]
  │       ├─ LossBalancer.plan_for_dataframe()  → LossBalancingPlan
  │       ├─ build_dataset() × 2                → train / val datasets
  │       ├─ build_dataloader() × 2             → train / val DataLoaders
  │       │    └─ DDP val sharding (PERF-2)
  │       ├─ build_optimizer(model, lr, wd)     → AdamW
  │       ├─ build_scheduler(optimizer, cfg)    → LambdaLR / cosine / etc.
  │       ├─ LossEngine(LossEngineConfig)        → weighted multi-task loss
  │       ├─ MonitoringEngine()                  → step-level health monitor
  │       ├─ TaskScheduler([task])               → trivially returns task
  │       ├─ ExperimentTracker(cfg)              → MLflow/W&B/none
  │       ├─ TrainingStep(model, optimizer, …)   → forward+backward engine
  │       ├─ EvaluationEngine(EvaluationConfig)  → streaming val metrics
  │       └─ Trainer(all_above)                  → returned to caller
  │
  └─ trainer.train()
         └─ (see Section 4)
```

### Multi-Task Pipeline

```
Caller
  │
  ├─ create_multitask_trainer_fn(settings, data_bundle, tokenizer)
  │       │
  │       ├─ get_all_tasks() → [bias, ideology, propaganda, ...]
  │       ├─ MultiTaskTruthLensModel(config)   single shared encoder
  │       ├─ model.to(device)
  │       ├─ build_dataset() × (tasks × 2)     per-task train/val
  │       ├─ MultiTaskLoader(per_task_loaders) yields {task, inputs, labels}
  │       ├─ LossEngine(task_types=all_tasks)   activates EMA normalizer
  │       ├─ TaskScheduler(tasks, adaptive)     loss-driven sampling
  │       ├─ TrainingStep, EvaluationEngine, ExperimentTracker, ...
  │       └─ Trainer(params_override with all tuning knobs)
  │
  └─ trainer.train()
```

### Data Flow Through One Training Step

```
DataLoader → batch (dict)
   │
   ├─ move_batch_to_device()         [training_utils: non_blocking iff pinned]
   ├─ model(**batch) → outputs       [forward pass under autocast]
   ├─ LossEngine.compute(outputs)    [multi-task weighted loss]
   │    ├─ TaskLossRouter.route()    [per-task: binary/multiclass/multilabel/regression]
   │    ├─ EMA normalization         [prevents one task dominating]
   │    └─ coverage check
   ├─ loss / grad_accum_steps        [LOSS-3: pre-scaled task weights]
   ├─ scaler.scale(loss).backward()  [AMP]
   ├─ accumulate N steps ...
   ├─ scaler.unscale_(optimizer)
   ├─ clip_grad_norm_(max_norm)      [spike watchdog runs here]
   ├─ scaler.step(optimizer)
   ├─ scaler.update()
   ├─ scheduler.step()
   └─ AutoDebugEngine.step()         [EMA, anomaly classify, action decide]
```

---

## 4. Training Loop Breakdown

### Trainer.train() — Outer Epoch Loop

```python
def train(self):
    setup_runtime(model, config)          # TF32, compile, grad-checkpoint
    run_sanity_check(model, loader)       # dry_run=True, MT-3

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            result = training_step.run(batch, global_step, epoch)
            global_step += 1

        metrics = evaluator.evaluate(model, val_loader)
        _inject_weighted_composite(metrics)  # WEIGHTED-COMPOSITE-METRIC

        if _should_stop(metrics, epoch):     # early stopping with min_epochs
            break

    # try/finally ensures tracker.finish() + distributed.cleanup() always run
```

### Early Stopping Logic

| Parameter | Default | Description |
|---|---|---|
| `patience` | 3 | Epochs without improvement before stopping |
| `min_delta` | 0.0 | Minimum improvement to reset patience counter |
| `min_epochs` | 1 | Always train at least this many epochs |
| `maximize_metric` | False | True for accuracy/F1; False for loss |
| `monitor_metric` | `"val_loss"` | Metric key to watch |

The `WEIGHTED-COMPOSITE-METRIC` key is injected into eval results when multiple tasks are active. It is a weighted average of per-task scores using `monitor_task_weights` (decoupled from loss `task_weights` — see Section 10).

### Cleanup Contract (try/finally)

```python
try:
    self._train_loop()
finally:
    if self.tracker:
        self.tracker.finish()       # always closes MLflow run / W&B run
    if self.distributed:
        self.distributed.cleanup()  # always destroys process group
```

This is annotated `N-LOW-6` in the source — previously cleanup only ran on success, leaving orphan tracking runs and zombie process groups after crashes.

---

## 5. File-by-File Deep Dive

### 5.1 `trainer.py` — Orchestrator

**Class:** `Trainer`

**Constructor parameters:**

| Parameter | Type | Purpose |
|---|---|---|
| `config_path` | str | Path to YAML for `ModelConfigLoader` |
| `model` | `nn.Module` | Pre-built, pre-moved model |
| `train_loader` | `DataLoader` | Training data |
| `val_loader` | `DataLoader` | Validation data |
| `training_step` | `TrainingStep` | Per-batch engine |
| `evaluator` | `EvaluationEngine` | Validation runner |
| `checkpoint` | optional | Checkpoint engine plug-in |
| `distributed` | optional | `DistributedEngine` plug-in |
| `tracker` | optional | `ExperimentTracker` |
| `monitor_metric` | str | Key to track for early stopping |
| `maximize_metric` | bool | Direction of metric |
| `params_override` | dict | Overrides YAML knobs (used by Optuna) |
| `setup_config` | `TrainingSetupConfig` | Runtime optimization flags |

**Key design decisions:**
- `params_override` is the mechanism by which Optuna trial parameters (`epochs`, `lr`, etc.) override config-file defaults without touching the YAML.
- `_inject_weighted_composite()` synthesizes the `WEIGHTED-COMPOSITE-METRIC` key from per-task scores so the monitor metric can be a single comparable scalar across all tasks.
- Cleanup is always in `try/finally` — this was a critical fix (`N-LOW-6`).

---

### 5.2 `training_step.py` — Per-Batch Engine

**Dataclass:** `TrainingStepConfig`

| Field | Default | Notes |
|---|---|---|
| `gradient_accumulation_steps` | 1 | Effective batch = batch_size × this |
| `max_grad_norm` | 1.0 | Gradient clipping ceiling |
| `spike_warn_threshold` | 100.0 | Pre-clip grad norm above which a warning fires |
| `spike_decay_factor` | 0.85 | LR multiplier applied on spike (post-audit relaxed from 0.7) |
| `spike_skip_threshold` | 150.0 | Pre-clip grad norm above which the entire step is skipped |
| `use_mixed_precision` | True | AMP enable |
| `amp_dtype` | `"float16"` | `"float16"` or `"bfloat16"` |

**Key methods:**
- `run(batch, global_step, epoch)` — main entry; returns `{"loss": float, "grad_norm": float, ...}`
- `_reduce_lr()` — multiplies all optimizer param-group LRs by `spike_decay_factor`; short-circuits if factor ≥ 1.0
- `get_amp_components(device, dtype)` — shim for PyTorch ≥2.3 (`torch.amp.autocast`) vs ≤2.2 (`torch.cuda.amp.autocast`) API

**Spike watchdog bands:**

```
grad_norm < spike_warn_threshold (100):   normal step, no warning
spike_warn_threshold ≤ grad_norm < spike_skip_threshold (150):  warn + reduce LR
grad_norm ≥ spike_skip_threshold (150):   skip optimizer step entirely
```

**Dry-run mode:** `dry_run=True` runs one batch without parameter updates (used for sanity check in `training_setup.run_sanity_check()`).

---

### 5.3 `training_setup.py` — Runtime Setup

**Dataclass:** `TrainingSetupConfig` (frozen)

| Field | Default | Notes |
|---|---|---|
| `use_amp` | True | AMP enable |
| `amp_dtype` | `"float16"` | |
| `allow_tf32` | True | cuDNN TF32 for matmul — significant A100 speedup |
| `use_compile` | True | `torch.compile` (annotated `COMPILE-RE-ENABLED`) |
| `compile_mode` | `"reduce-overhead"` | Triton kernel fusion mode |
| `use_gradient_checkpointing` | True | Recompute activations to save VRAM |
| `detect_anomaly` | False | `torch.autograd.set_detect_anomaly` — disable in production |

**Functions:**

| Function | What It Does |
|---|---|
| `setup_runtime(model, cfg)` | Applies TF32, compile, grad-checkpoint. MUST run before DDP wrap (N-LOW-3) |
| `optimize_model(model, cfg)` | Applies `torch.compile` if enabled |
| `run_sanity_check(model, loader, cfg)` | One dry_run batch to catch shape/device mismatches before epoch 1 (MT-3) |
| `get_autocast(device, dtype)` | Returns the correct `autocast` context manager for PyTorch version |
| `create_grad_scaler(cfg, device)` | Returns a `GradScaler`; no-op on CPU or bfloat16 |

**Critical ordering constraint:** `optimize_model()` (which calls `torch.compile`) MUST complete before `DistributedEngine.wrap_model()` (DDP wrap). Reversing this order corrupts DDP's bucket assignment. Annotated `N-LOW-3`.

---

### 5.4 `create_trainer_fn.py` — Single-Task Factory

The single-task factory assembles all engine objects from a flat `params` dict and returns a wired `Trainer`. It is the entry point used by:
- `cross_validate_task()` (one trainer per fold)
- `tune_task()` (one trainer per Optuna trial)
- Direct single-task training

**Key wiring decisions:**

1. **GPU-1 (Device Ordering):** `model.to(device)` runs exactly once, immediately after `build_model`, before `build_optimizer`. Any subsequent device probe raises loudly instead of silently re-moving.

2. **Loss-Balancing Plan:** Computed from train split labels before `build_dataset`. The same `valid_label_indices` list flows into:
   - `build_dataset` (drops degenerate columns from label tensors)
   - `LossEngineConfig` (slices logits in `TaskLossRouter._multilabel_loss`)
   - `EvaluationConfig` (slices logits in eval to match label width)

3. **BUG-7 (Scheduler Steps):** `num_training_steps` is computed as `steps_per_epoch × epochs` (not defaulted to 1000), so the LR schedule decays correctly.

4. **PERF-2 (DDP Validation Sharding):** If DDP is initialized, the val DataLoader is replaced with a `DistributedSampler`-backed version so each rank evaluates only its shard.

---

### 5.5 `create_multitask_trainer_fn.py` — Multi-Task Factory

The production entry point. Takes a `settings` namespace (from `config/config.yaml` via `load_settings()`) and a `data_bundle` dict.

**Resolver functions (all safe, with graceful fallback):**

| Resolver | YAML Key | Default |
|---|---|---|
| `_resolve_lr` | `training.lr` or `optimizer.lr` | raises ValueError (no silent 0) |
| `_resolve_device` | `training.device` | cuda if available |
| `_resolve_batch_size` | `training.batch_size` or `data.batch_size` | 16 |
| `_resolve_grad_accum` | `training.gradient_accumulation_steps` | 1 |
| `_resolve_epochs` | `training.epochs` or `training.num_epochs` | 1 |
| `_resolve_early_stopping_patience` | `training.early_stopping_patience` | 3 |
| `_resolve_min_epochs` | `training.min_epochs` | 1 |
| `_resolve_min_delta` | `training.early_stopping_min_delta` | 0.0 |
| `_resolve_spike_warn_threshold` | `training.spike_warn_threshold` | 100.0 |
| `_resolve_spike_decay_factor` | `training.spike_decay_factor` | 0.85 |
| `_resolve_spike_skip_threshold` | `training.spike_skip_threshold` | 150.0 |
| `_resolve_spike_lr_scale` | `training.spike_lr_scale` | 0.5 |
| `_resolve_normalizer_alpha` | `loss.normalizer_alpha` | None (uses EMA default) |
| `_resolve_task_weights` | `task_weights` map | uniform 1.0 per task |
| `_resolve_monitor_task_weights` | `monitor_task_weights` map | falls back to task_weights |
| `_resolve_grad_scaler_init_scale` | `precision.grad_scaler_init_scale` | None (torch default) |

**MT-FACTORY-NOLEGACY-CFG:** The factory no longer routes through `ModelConfigLoader.load_multitask_config` (the legacy strict dataclass parser). All knobs are resolved directly from `settings` by the resolver functions above.

**`_build_model_config`:** Maps `settings.model` → `MultiTaskTruthLensConfig`. Silently drops five known runtime-only YAML keys (`torch_compile`, `compile_mode`, `gradient_checkpointing`, `flash_attention`, `hidden_dim`) that belong to `TrainingSetupConfig`, so they don't trigger the "unknown field" warning on `MultiTaskTruthLensConfig`.

**`_build_monitoring_config`:** Maps `settings.monitoring` → `MonitoringConfig` field by field. Previous code passed the raw AttrDict directly, causing `AttributeError` on fields the YAML didn't define (`MONITORING-CFG-FIX`).

---

### 5.6 `loss_engine.py` — Multi-Task Loss Aggregation

**Wraps:** `MultiTaskLoss` from `src/models/loss/`

**Config:** `LossEngineConfig`

| Field | Default | Notes |
|---|---|---|
| `task_types` | required | `{task_name: "multiclass"/"multilabel"/"binary"/"regression"}` |
| `gradient_accumulation_steps` | 1 | Pre-scales task weights (LOSS-3) |
| `normalization` | `"active"` | `"active"` / `"sum"` / `"mean"` (CFG-5) |
| `class_weights` | None | Per-task class weight tensors |
| `pos_weights` | None | Per-task positive weight tensors (binary/multilabel) |
| `use_focal` | None | Per-task focal loss flags |
| `focal_gamma` | None | Per-task gamma values |
| `valid_label_indices` | None | Per-task surviving multilabel column indices |

**Single-task guard (MT-1):** When only one task is registered, the EMA normalizer and coverage tracker are disabled. This prevents the normalizer from inflating gradients on a task it has no peers to compare against.

**LOSS-3 (Pre-scaling):** Task weights are multiplied by `gradient_accumulation_steps` at construction time. This compensates for the fact that the training step divides the loss by `grad_accum` before `.backward()` — without pre-scaling, task weights would be implicitly divided by `grad_accum` on every step.

**Normalization modes:**
- `"active"`: the EMA normalizer scales each task's loss so they contribute equally to the gradient signal regardless of absolute magnitude.
- `"sum"`: raw sum of task losses.
- `"mean"`: mean of task losses.

---

### 5.7 `loss_functions.py` — Loss Primitives

**`FocalLoss`** (Lin et al., 2017):
```python
FL(p) = -alpha * (1 - p)^gamma * log(p)
```
- `gamma` default 2.0; higher values focus more on hard examples.
- Works for binary and multiclass (one-vs-all).

**`compute_class_weights(labels, num_classes)`:**
- Inverse-frequency weighting matching sklearn's `"balanced"` strategy.
- Formula: `n_samples / (n_classes × bincount)`.

**`compute_pos_weight(labels)`:**
- Binary/multilabel positive weight: `neg_count / pos_count`, smoothed and clipped to 100.
- Prevents division by zero when a column is all-positive or all-negative.

**Per-type dispatchers:**

| Function | Task Type | Notes |
|---|---|---|
| `binary_loss` | binary | BCE with logits + optional pos_weight + optional focal |
| `multiclass_loss` | multiclass | CrossEntropy + optional class_weights + optional focal |
| `multilabel_loss` | multilabel | BCE per-label + optional pos_weight; NaN-ignore mask (LOSS-4) |
| `regression_loss` | regression | MSE; finite-value mask before reduction |

**LOSS-4 (NaN ignore in multilabel):** A `torch.isfinite(labels)` mask is applied before computing BCE. This handles missing-value sentinels that survive joins in the label pipeline without poisoning the loss for the full batch.

---

### 5.8 `loss_balancer.py` — Imbalance Analysis

The third layer of the three-layer imbalance strategy:

```
Layer 1: TaskScheduler          — between-task imbalance (sampling frequency)
Layer 2: data-level samplers    — within-task exposure (WeightedRandomSampler)
Layer 3: loss_balancer          — within-task gradient signal (this module)
```

**Entry points:**

| Function | Input | Output |
|---|---|---|
| `plan_for_labels(labels, task_type, ...)` | raw labels array | `LossBalancingPlan` |
| `plan_for_dataframe(df, label_columns, task_type, ...)` | DataFrame + column names | `LossBalancingPlan` |

**`LossBalancingPlan` fields:**

| Field | Meaning |
|---|---|
| `class_weights` | Tensor for multiclass CE |
| `pos_weight` | Tensor for binary/multilabel BCE |
| `use_focal` | Whether to switch CE → FocalLoss |
| `focal_gamma` | Focal loss gamma |
| `valid_label_indices` | Surviving multilabel column indices |
| `dropped_label_indices` | Degenerate columns removed |
| `notes` | Human-readable list of decisions |

**Config thresholds (`LossBalancerConfig`):**

| Field | Default | Meaning |
|---|---|---|
| `weight_threshold` | 0.7 | Max class proportion above which class_weights fire |
| `focal_threshold` | 0.9 | Max proportion above which focal loss fires |
| `focal_gamma` | 2.0 | Focal loss gamma |
| `multilabel_min_pos_ratio` | 0.0 | Min positive ratio to keep a multilabel column |

**MULTILABEL-FOCAL-FIX:** The multilabel branch now mirrors the binary/multiclass focal-loss gate. It computes per-column skew as `max(pos_ratio, 1 - pos_ratio)` and fires focal loss when the worst-skewed column exceeds `focal_threshold`.

**Degenerate column detection:** Delegates to `src.utils.label_cleaning.remove_single_class_columns` — the same function the dataset factory uses — so the planner and the dataset always agree on which columns are valid. A mismatch would produce a shape error in the loss router.

---

### 5.9 `evaluation_engine.py` — Validation

**Streaming metric classes (GPU-resident):**

| Class | Task Types | Metrics Computed |
|---|---|---|
| `StreamingAccuracy` | multiclass | accuracy |
| `StreamingF1` | multilabel, binary | micro F1 (precision/recall from TP/FP/FN) |
| `StreamingMSE` | regression | mean squared error |

**PERF-1 (No per-batch sync):** Accumulators are `torch.float64` tensors allocated lazily on the first batch's device. The `.item()` host sync only happens once in `compute()`.

**PERF-2 (DDP correctness):** Each `Streaming*` class implements `sync_distributed()` which all_reduces raw numerators and denominators (not pre-divided averages). This is mathematically correct for variable-size shards (`drop_last=False`).

**EVAL-MULTILABEL-SLICE:** When `valid_label_indices` is set in `EvaluationConfig`, the evaluator slices logits with `index_select(-1, idx_t)` before comparing to labels. This mirrors `TaskLossRouter._multilabel_loss` exactly, ensuring training and evaluation always apply the same column selection.

**GPU-4 (Avoid redundant model.to):** The engine probes `next(model.parameters()).device` and only moves the model if it differs from the target device. Re-moving a DDP-wrapped model breaks its bucket assignment.

**MT-2 (Binary task support):** Previously the `binary` task type had no metric allocated, so binary-task early stopping silently saw no values. Now `binary` maps to `StreamingF1` with sigmoid thresholding.

**EDGE-2 (Label normalization):** `batch["labels"]` may be a raw tensor (single-task collate) or a `{task: tensor}` dict (multi-task collate). The engine normalizes both to a dict before the per-task loop.

---

### 5.10 `experiment_tracker.py` — Tracking

**Backends:** `"mlflow"` | `"wandb"` | `"none"` (default)

**`ExperimentTrackerConfig` fields:**

| Field | Notes |
|---|---|
| `backend` | Backend name |
| `project_name` | MLflow experiment / W&B project |
| `run_name` | Run display name |
| `tracking_uri` | MLflow tracking server URI |
| `tags` | `{str: str}` metadata |
| `group` | W&B run group (used by CV and tuning for grouping folds/trials) |

**Distributed safety:** Every logging method calls `_is_main()` which checks `dist.get_rank() == 0`. Non-main ranks return immediately without logging.

**Error safety:** All backend calls go through `_safe(fn, *args)` which swallows exceptions and logs a warning. A broken tracking server never crashes training.

**N-HIGH-3 (Step counter fix):** When the caller passes an explicit `step=` argument, the internal counter is NOT advanced. Previous code unconditionally did `self._step = step + 1`, causing the internal counter to shadow then diverge from the caller's global step — leading to backwards-stepping metrics in MLflow.

**`_flatten(metrics, prefix)`:** Recursively flattens nested dicts to `/`-delimited keys (`{"a": {"b": 1}} → {"a/b": 1}`).

**Context manager:** `__enter__` / `__exit__` calls `finish()` on exit, supporting `with ExperimentTracker(...) as t:` usage.

---

### 5.11 `distributed_engine.py` — DDP

**`DistributedConfig` fields:**

| Field | Default | Notes |
|---|---|---|
| `backend` | `"nccl"` | Auto-falls back to `"gloo"` on CPU hosts |
| `init_method` | `"env://"` | Standard torchrun env var init |
| `use_ddp` | True | |
| `find_unused_parameters` | False | Set True only if model has conditional branches |
| `gradient_as_bucket_view` | True | Memory-efficient gradient bucketing |

**GPU-3 (NCCL→gloo fallback):** If `backend="nccl"` but `torch.cuda.is_available()` is False, the engine falls back to `"gloo"` instead of crashing. This enables CPU-only runs (CI, dev machines, Replit Reserved-VM trial tier).

**N-CRIT-3 (No post-optimizer move):** `wrap_model()` probes the model's current device and raises `RuntimeError` if it differs from the target CUDA device. Previously it called `model.to(device)` here (after the optimizer was built), which silently invalidated the optimizer's parameter references — causing "expected all tensors to be on the same device" on the first `optimizer.step()`.

**Utilities:**

| Method | Purpose |
|---|---|
| `create_sampler(dataset)` | Returns `DistributedSampler` for training data |
| `barrier()` | `dist.barrier()` — process synchronization point |
| `all_reduce(tensor)` | SUM + normalize to mean across world_size |
| `broadcast(tensor, src)` | Broadcast from rank `src` |
| `is_main_process()` | True iff `rank == 0` |
| `cleanup()` | `dist.destroy_process_group()` |

---

### 5.12 `instrumentation.py` — AutoDebugEngine

**Component hierarchy:**

```
AutoDebugEngine
├── LossTracker       — bias-corrected EMA per task
├── LossStats         — rolling variance (window=50, using statistics.fmean)
├── GradTracker       — per-step grad norm history (window=50)
├── SpikeDetector     — ratio + z-score spike detection
├── AnomalyClassifier — maps signals to: nan_loss / exploding_gradients /
│                       vanishing_gradients / logit_collapse / loss_spike / normal
├── FailureMemory     — circular buffer of failure events (max 500 per type)
└── GradNorm          — optional; computes adaptive task weights (Chen 2018)
```

**`LossTracker`:**
- Bias-corrected EMA: `ema_t = α·loss + (1-α)·ema_{t-1}`, output `= ema_t / (1 - (1-α)^t)`
- N-CRIT-2: Non-finite losses skip the EMA update (carry forward previous value) and increment a per-task `_nan_counter` for downstream classification. Previously raised RuntimeError, crashing the whole step.

**`LossStats`:**
- Uses `statistics.fmean` and `statistics.variance` over a deque.
- N-MED-1: Previous implementation allocated a CUDA tensor per task per step for variance — expensive and caused accidental GPU traffic. Host-side `statistics` module is 10× faster here.

**`GradTracker`:**
- REC-3: When the trainer passes `cached_grad_norm` (already computed before `optimizer.zero_grad`), the engine appends it directly without re-iterating parameters. Without this, it would see `p.grad = None` everywhere (post-`zero_grad`) and record 0.0, triggering false "vanishing_gradients" alarms.

**`AnomalyClassifier` labels:**

| Label | Trigger |
|---|---|
| `nan_loss` | `math.isfinite(loss)` is False |
| `exploding_gradients` | `grad_norm > 1000` |
| `vanishing_gradients` | `grad_norm < 1e-7` |
| `logit_collapse` | `logits.std() < 1e-4` |
| `loss_spike` | `loss / (ema + 1e-8) > 2.5` |
| `normal` | none of the above |

**`_decide_action` outputs:**

| Action | Trigger |
|---|---|
| `"stop_training"` | `nan_loss` |
| `"reduce_lr"` | `exploding_gradients` or spike |
| `"intervene"` | `severity > 0.8` |
| `"check_dataloader"` | `throughput_trend < -0.5` |
| `"none"` | all normal |

---

### 5.13 `monitor_engine.py` — Step-Level Health

**`MonitoringConfig` fields:**

| Field | Default | Notes |
|---|---|---|
| `spike_threshold` | 3.0 | Loss/EMA ratio above which spike fires |
| `ema_alpha` | 0.1 | EMA decay for loss smoothing |
| `health_threshold` | 0.3 | Health score below which REDUCE_LR fires |
| `enable_grad_monitor` | True | Enables periodic grad norm computation |
| `grad_monitor_interval` | 100 | Steps between grad norm checks |
| `enable_throughput` | True | Tracks samples/second |
| `throughput_ema_alpha` | 0.2 | EMA smoothing for throughput |
| `anomaly_on_nan` | True | Return NAN action instead of raising |

**LOSS-1 (Unified SpikeDetector):** Previously there were two `SpikeDetector` implementations — one here (pure ratio) and one in `instrumentation` (bias-corrected EMA + z-score). They could fire conflicting policies in the same step. Now `MonitoringEngine` imports `SpikeDetector` from `instrumentation` and delegates.

**N-CRIT-1 (Non-finite loss):** `_extract_loss` previously raised `RuntimeError` on a non-finite loss value, which crashed before the `anomaly_on_nan` policy ran — making that flag a no-op. Now it returns `float("nan")` and lets the `torch.isfinite` guard downstream apply the policy.

**Health score formula:**
```python
stability = 1.0 - min(|loss - ema| / (ema + 1e-9), 1.0)
grad_penalty = min((grad_norm - 10) / 50, 1.0)  if grad_norm > 10
health = max(0.0, stability - grad_penalty)
```

---

### 5.14 `cross_validation.py` — K-Fold CV

**Primary functions:**

| Function | Purpose |
|---|---|
| `cross_validate_task(task, df, create_trainer_fn, params, ...)` | Runs K-fold CV for one task |
| `cross_validate_all_tasks(datasets, create_trainer_fn, params, ...)` | Runs `cross_validate_task` per task |
| `build_splits(df, label_column, n_splits, seed)` | Returns list of (train_idx, val_idx) pairs |
| `resolve_metric(task, metrics, strategy)` | Extracts scalar score from eval metrics dict |
| `build_dashboard(results)` | Summarizes CV results into per-task mean/std/folds |

**`build_splits` fallback logic (EDGE-1):**
```
StratifiedKFold(y=label_column)
  → if label_column missing: KFold (warns; expected for multilabel)
  → if smallest class < n_splits: KFold (warns; low-resource task)
```

**N-MED-3 (Per-fold seeding):** Each fold calls `set_seed(seed + fold_id)`. Previously the global seed was set once before the loop; fold 2 started from fold 1's leftover RNG state, inflating apparent variance reduction.

**EDGE-6 (NameError in cleanup):** The `finally` block uses `"trainer" in locals()` before `del trainer`. Previously `del trainer` would `NameError` if `create_trainer_fn` raised before the binding.

**`resolve_metric` priority order (strategy="auto"):**

| Task type | Keys tried (in order) |
|---|---|
| multilabel | `micro_f1`, `eval_micro_f1`, `val_loss`, `eval_loss` |
| multiclass | `accuracy`, `eval_accuracy`, `val_loss`, `eval_loss` |
| other | `f1`, `eval_f1`, `val_loss`, `eval_loss` |

---

### 5.15 `hyperparameter_tuning.py` — Optuna HPO

**Primary functions:**

| Function | Purpose |
|---|---|
| `tune_task(task, df, create_trainer_fn, n_trials, ...)` | Single-task tuning |
| `tune_all_tasks(datasets, ...)` | Iterates `tune_task` per task |
| `build_objective(task, df, create_trainer_fn, ...)` | Constructs Optuna objective function |
| `create_study(multi_objective, storage, task)` | Builds `optuna.Study` |

**Search space per trial:**

| Parameter | Distribution |
|---|---|
| `lr` | log-uniform [1e-6, 5e-4] |
| `batch_size` | categorical {8, 16, 32} |
| `epochs` | integer [2, 6] |
| `weight_decay` | uniform [0.0, 0.1] |

**N-HIGH-1 (base_params preserved):** The trial parameters are merged on top of `base_params` (`{**base_params, **trial_params}`). Previously the objective constructed params from ONLY the four trial keys, silently dropping every other caller-defined config key (gradient accumulation, max_grad_norm, scheduler type, etc.).

**N-LOW-5 (Tracking routed through ExperimentTracker):** Previously this module had its own MLflow/W&B calls that bypassed the tracker's distributed-safe paths. Now Optuna trial metrics and params flow through `ExperimentTracker(group=f"tune_{task}")`.

**`_resolve_direction(task)` (BUG-8):** Optuna's optimization direction defaults to `"minimize"`. For classification tasks (binary/multiclass/multilabel), the primary metric is accuracy or F1 — which must be maximized. This function calls `get_task_type()` and returns `"maximize"` for classification and `"minimize"` for regression.

**Multi-objective mode:** When `multi_objective=True`, the study minimizes `(score, -std)` or maximizes `(score, score)` depending on task direction (both objectives use the same direction), producing a Pareto front over `{best_score × least_variance}`.

---

### 5.16 `task_scheduler.py` — Task Sampling

**Strategies:**

| Strategy | Description |
|---|---|
| `round_robin` | Cycles through tasks in fixed order (default) |
| `random` | Uniform random selection |
| `weighted` | Weighted random using `task_weights` |
| `adaptive` | Softmax over EMA losses; high-loss tasks sampled more often |

**`adaptive` strategy rationale (N-LOW-7):**

Higher-loss tasks get higher selection probability — the scheduler focuses gradient steps on tasks where the joint encoder still has room to improve. This is the "focus on hard tasks" convention. Negating the scores would implement "focus on easy tasks" — keep this comment synchronized with `AutoDebugEngine._decide_action` and `GradNorm.compute` if the convention ever changes.

**CFG-6 (Single-task fast path):** With one task, every strategy collapses to "always return that task." The scheduler short-circuits `next_task()` and skips all RNG/softmax work. If a non-trivial strategy is wired for one task, a warning is logged.

**`update_losses(task_losses)`:** Updates EMA loss per task (used by adaptive strategy). Filters non-finite and non-positive values.

---

### 5.17 `training_utils.py` — Shared Utilities

| Function / Class | Purpose |
|---|---|
| `get_device(device)` | Returns `torch.device`, auto-detects CUDA |
| `move_batch_to_device(batch, device, non_blocking)` | Recursively moves tensors; `non_blocking=True` only when tensor is pinned AND device is CUDA |
| `compute_grad_norm(model)` | L2 norm of all parameter gradients; canonical implementation used by both training step and monitor |
| `get_current_lr(optimizer)` | Reads LR from first param group |
| `compute_throughput(batch_size, duration)` | Samples per second; returns 0.0 if duration ≤ 0 |
| `TrainingMetrics` | Dataclass holding per-step metrics: `task_losses`, `losses`, `grad_norm`, `lr`, `throughput` |

`move_batch_to_device` handles tensors, dicts, lists, and tuples recursively. Non-tensor values pass through unchanged. The `non_blocking` flag is gated on `batch.is_pinned()` — unlike previous implementations that set it unconditionally and gave a false impression of async H2D copies on un-pinned tensors.

---

### 5.18 `__init__.py` — Public API

All stable symbols are re-exported here. Optuna-backed symbols (`tune_task`, `tune_all_tasks`, `create_study`, `build_objective`) are loaded lazily via `__getattr__` so importing `src.training` does not pull Optuna into memory unless the caller uses those symbols.

---

## 6. Loss Functions & Optimization

### Loss Type Routing

```
LossEngine.compute(outputs)
    └── MultiTaskLoss.forward(task_logits, labels)
            └── TaskLossRouter.route(task, logits, labels)
                    ├── binary_loss()          → BCE + optional pos_weight + focal
                    ├── multiclass_loss()      → CE  + optional class_weights + focal
                    ├── multilabel_loss()      → BCE/label + optional pos_weight + NaN-mask
                    └── regression_loss()      → MSE + finite mask
```

### Multi-Task Loss Weighting

Each task's raw loss `L_task` is transformed before aggregation:

```
1. Task weight:      L_task *= w_task × grad_accum_steps    [LOSS-3]
2. EMA normalization: L_task /= ema_scale_task              [normalization="active"]
3. Sum across tasks: L_total = Σ L_task
4. Divide for step:  backward on L_total / grad_accum_steps
```

### FocalLoss Formula

```
p_t = sigmoid(logits) if binary else softmax(logits)[true_class]
FL = -alpha × (1 - p_t)^gamma × log(p_t + eps)
```

`gamma=2.0` reduces well-classified examples' weight by ~75% (`(1-0.9)^2 = 0.01` for `p=0.9`), concentrating learning on hard examples.

### Class Weight Formula

```
class_weight[c] = n_samples / (n_classes × count[c])
```

Matches sklearn's `class_weight="balanced"`. Normalizes so the weighted sum of samples across classes equals `n_samples`.

### Positive Weight Formula (Binary/Multilabel)

```
pos_weight = (neg_count + smoothing) / (pos_count + smoothing)
pos_weight = clip(pos_weight, max=100)
```

Applied as the `pos_weight` argument to `F.binary_cross_entropy_with_logits`. Smoothing prevents ÷0 when a label column is all-one.

---

## 7. Hyperparameters & Configuration Reference

### Training Knobs

| Parameter | YAML Path | Default | Notes |
|---|---|---|---|
| Learning rate | `training.lr` or `optimizer.lr` | required | No silent default |
| Batch size | `training.batch_size` or `data.batch_size` | 16 | |
| Epochs | `training.epochs` or `training.num_epochs` | 1 | |
| Grad accumulation | `training.gradient_accumulation_steps` | 1 | |
| Max grad norm | `training.max_grad_norm` | 1.0 | Gradient clipping |
| Weight decay | `training.weight_decay` or `optimizer.weight_decay` | 0.0 | |
| Seed | `project.seed` | 42 | |

### Early Stopping Knobs

| Parameter | YAML Path | Default |
|---|---|---|
| Patience | `training.early_stopping_patience` | 3 |
| Min delta | `training.early_stopping_min_delta` | 0.0 |
| Min epochs | `training.min_epochs` | 1 |
| Monitor metric | passed via `params_override` | `"val_loss"` |
| Maximize metric | passed via `params_override` | False |

### Spike / Gradient Watchdog

| Parameter | YAML Path | Default |
|---|---|---|
| Warn threshold | `training.spike_warn_threshold` | 100.0 |
| Decay factor | `training.spike_decay_factor` | 0.85 |
| Skip threshold | `training.spike_skip_threshold` | 150.0 |
| LR scale | `training.spike_lr_scale` | 0.5 |

### AMP / Precision

| Parameter | YAML Path | Default |
|---|---|---|
| Use AMP | `training.use_amp` or `precision.use_amp` | True |
| AMP dtype | `training.amp_dtype` | `"float16"` |
| Allow TF32 | `training.allow_tf32` | True |
| GradScaler init scale | `precision.grad_scaler_init_scale` | None (65536) |

### Compile / Memory

| Parameter | YAML Path | Default |
|---|---|---|
| Use compile | `model.torch_compile` | True |
| Compile mode | `model.compile_mode` | `"reduce-overhead"` |
| Gradient checkpointing | `model.gradient_checkpointing` | True |

### Loss Balancer

| Parameter | Config Class | Default |
|---|---|---|
| Class weight threshold | `LossBalancerConfig.weight_threshold` | 0.7 |
| Focal loss threshold | `LossBalancerConfig.focal_threshold` | 0.9 |
| Focal gamma | `LossBalancerConfig.focal_gamma` | 2.0 |
| Multilabel min pos ratio | `LossBalancerConfig.multilabel_min_pos_ratio` | 0.0 |

### Task Scheduler

| Parameter | Config Class | Default |
|---|---|---|
| Strategy | `TaskSchedulerConfig.strategy` | `"round_robin"` |
| Temperature (adaptive) | `TaskSchedulerConfig.temperature` | 1.0 |
| EMA alpha | `TaskSchedulerConfig.ema_alpha` | 0.1 |
| Min probability | `TaskSchedulerConfig.min_prob` | 1e-3 |

### HPO Search Space

| Parameter | Distribution |
|---|---|
| `lr` | log-uniform [1e-6, 5e-4] |
| `batch_size` | {8, 16, 32} |
| `epochs` | int [2, 6] |
| `weight_decay` | uniform [0.0, 0.1] |

---

## 8. Validation Strategy

### Architecture

Validation runs after every epoch using `EvaluationEngine.evaluate(model, val_loader)`.

```python
@torch.inference_mode()
def evaluate(model, dataloader):
    model.eval()
    metrics = _init_metrics()          # one Streaming* per task

    for batch in dataloader:
        batch = _move_batch(batch)
        outputs = model(**model_batch)
        _update_metrics(metrics, outputs, batch)

    return _compute_metrics(metrics)   # sync DDP, then compute()
```

### Per-Task Metric Selection

| Task type | Metric class | Output key |
|---|---|---|
| `multiclass` | `StreamingAccuracy` | `{task}_score` (accuracy) |
| `multilabel` | `StreamingF1` | `{task}_score` (micro F1) |
| `binary` | `StreamingF1` | `{task}_score` (F1) |
| `regression` | `StreamingMSE` | `{task}_score` (MSE) |

### Weighted Composite Score

When multiple tasks are active, `Trainer._inject_weighted_composite()` adds:

```python
metrics["WEIGHTED-COMPOSITE-METRIC"] = Σ(w_task × task_score) / Σ(w_task)
```

using `monitor_task_weights` (from `settings.monitor_task_weights` YAML key), which defaults to `task_weights` if not separately specified.

### DDP Validation Correctness

Under distributed training:

1. Each rank evaluates only its shard (via `DistributedSampler(shuffle=False, drop_last=False)`).
2. `Streaming*.sync_distributed()` all_reduces raw (numerator, denominator) — not pre-divided scalars.
3. Final `compute()` runs after the all_reduce, producing a globally correct metric.

Averaging pre-divided rank-local scores is mathematically wrong when ranks have different shard sizes.

### Multilabel Column Alignment

The evaluation engine receives `valid_label_indices` from `EvaluationConfig`. Before computing predictions, it slices `logits` to the surviving columns:

```python
idx_t = torch.as_tensor(valid_idx, dtype=torch.long, device=logits.device)
logits = logits.index_select(-1, idx_t)
```

This matches the training loss router's slicing exactly, preventing shape mismatches.

---

## 9. Checkpointing & Logging

### Experiment Tracking

```python
tracker = ExperimentTracker(ExperimentTrackerConfig(
    backend="mlflow",                   # or "wandb" or "none"
    project_name="truthlens",
    run_name="run_2026",
    tracking_uri="http://mlflow:5000",
    tags={"env": "prod"},
    group="cv_fold_group",             # for CV / tuning grouping
))

tracker.log_config(params)              # hyperparameters
tracker.log_metrics({"loss": 0.3}, step=100)
tracker.log_artifact("model.pt")
tracker.watch_model(model)             # W&B gradient histograms
tracker.finish()                       # always called in Trainer's finally block
```

### What Gets Logged

At each step (via `TrainingStep`):
- `train/loss` per task
- `train/grad_norm`
- `train/lr`
- `train/throughput`
- `debug/*` (from `AutoDebugEngine.step()`)
- `monitor/*` (from `MonitoringEngine.update()`)
- `time/elapsed` (added automatically by tracker)

At each epoch:
- `eval/{task}_score` per task
- `eval/WEIGHTED-COMPOSITE-METRIC`
- `eval/val_loss`

### Checkpointing

The `Trainer` accepts a `checkpoint` plug-in (any object with `.save(model, metrics)` and `.load()` methods). The `create_trainer_fn` reads `params.get("checkpoint")` — pass a `CheckpointEngine` instance through `params` to enable checkpointing. The engine itself is not defined in `src/training/`; it is expected to be provided externally or from `src/models/`.

---

## 10. Config Integration

### Settings Namespace

The multi-task factory reads from a `settings` object that is attribute-accessible and optionally dict-like (both AttrDict and plain dict work via `_get(obj, key, default)`). Produced by `src.utils.settings.load_settings()`.

### YAML Section Mapping

```yaml
project:
  seed: 42

training:
  lr: 2e-5
  batch_size: 16
  epochs: 10
  weight_decay: 0.01
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
  use_amp: true
  amp_dtype: "float16"
  allow_tf32: true
  early_stopping_patience: 3
  early_stopping_min_delta: 0.001
  min_epochs: 2
  spike_warn_threshold: 100.0
  spike_decay_factor: 0.85
  spike_skip_threshold: 150.0
  spike_lr_scale: 0.5

model:
  model_name: "roberta-base"
  dropout: 0.1
  torch_compile: true            # → TrainingSetupConfig.use_compile
  compile_mode: "reduce-overhead"
  gradient_checkpointing: true   # → TrainingSetupConfig.use_gradient_checkpointing

optimizer:
  lr: 2e-5
  weight_decay: 0.01

precision:
  use_amp: true
  grad_scaler_init_scale: 1024.0

loss:
  normalizer_alpha: 0.1

monitoring:
  spike_threshold: 3.0
  ema_alpha: 0.1
  health_threshold: 0.3
  enable_grad_monitor: true

task_weights:
  bias: 1.0
  ideology: 1.2
  propaganda: 0.8
  factuality: 1.0
  emotion: 0.7
  narrative_frame: 0.5

monitor_task_weights:        # decoupled from loss weights
  bias: 1.0
  ideology: 1.0
  propaganda: 1.0
  factuality: 1.0
```

### `params_override` Mechanism

The `Trainer` reads its loop parameters (epochs, patience, min_epochs, min_delta, monitor_metric, maximize_metric) from `self.params_override` (a dict). This dict is populated by the factory with all resolved settings. Optuna can pass a different `params` dict with trial-level overrides (`epochs=3`, `lr=5e-5`) which flow through `create_trainer_fn` → `Trainer.params_override` without touching the YAML.

---

## 11. Optimization & Efficiency

### AMP (Automatic Mixed Precision)

- Enabled by default (`use_amp=True`).
- Dtype: `float16` (default) or `bfloat16`.
- `GradScaler` is created by `create_grad_scaler()`. Returns a no-op scaler on CPU or bfloat16 (no dynamic range scaling needed).
- PyTorch ≥2.3 uses `torch.amp.autocast(device_type, dtype)`; ≤2.2 uses `torch.cuda.amp.autocast`. The `get_amp_components` shim handles both.

### torch.compile

- Enabled via `use_compile=True` (COMPILE-RE-ENABLED after audit).
- Mode: `"reduce-overhead"` (default) — fuses kernel launches, beneficial for training loops with many small ops.
- Must run before DDP wrap (N-LOW-3).

### Gradient Checkpointing

- Enabled via `use_gradient_checkpointing=True`.
- Trades compute for memory: activations are discarded after the forward pass and recomputed during backward.
- Saves approximately 60% of activation memory, enabling larger batch sizes.

### TF32 on Ampere GPUs

- `torch.backends.cuda.matmul.allow_tf32 = True` — uses TF32 for matmul on A100/RTX30xx (up to 8× FP32 throughput with minor precision loss).
- `torch.backends.cudnn.allow_tf32 = True` — same for cuDNN convolutions.

### Pinned Memory + Non-Blocking Transfers

`move_batch_to_device` uses `non_blocking=True` only when:
- Source tensor is in pinned (page-locked) memory
- Destination is a CUDA device

This enables overlap of H2D copy with GPU compute. Setting `pin_memory=True` in `DataLoaderConfig` (default) pins the DataLoader's output buffers.

### GPU-Resident Streaming Metrics

Validation accumulators (TP, FP, FN, sum_sq, count) are held as `float64` tensors on the GPU. The host sync (`float(tensor.item())`) happens once per epoch in `compute()`, not once per batch.

### DDP Memory Efficiency

`gradient_as_bucket_view=True` makes DDP store gradients as views into the communication buckets, avoiding an additional copy per gradient synchronization.

---

## 12. Failure Handling & Debugging

### Gradient Spike Response

```
Step's pre-clip grad_norm:
  < 100:   Normal step
  100-150: Log warning + reduce all optimizer LRs × spike_decay_factor (0.85)
  ≥ 150:   Skip optimizer step entirely, log error
```

The LR reduction uses `_reduce_lr()` which iterates `optimizer.param_groups` and multiplies each `lr` in place. A factor ≥ 1.0 is a no-op (SPIKE-LR-DISABLED).

### NaN Loss Policy

1. `MonitoringEngine._extract_loss()` returns `float("nan")` instead of raising.
2. `MonitoringEngine.update()` checks `torch.isfinite` and returns `action=NAN`.
3. `LossTracker.update()` skips EMA update and increments `_nan_counter`.
4. `AnomalyClassifier.classify()` returns `"nan_loss"`.
5. `AutoDebugEngine._decide_action()` returns `"stop_training"`.
6. `Trainer` checks the action and can abort the epoch.

This chain ensures a single NaN loss does not crash the step — it is classified, recorded, and actioned at the loop level.

### AutoDebugEngine Actions

| Action | Meaning | Recommended Response |
|---|---|---|
| `stop_training` | NaN loss — gradient is contaminated | Investigate data, reduce LR, check label correctness |
| `reduce_lr` | Exploding gradients or spike | Automatic via `_reduce_lr()`; consider lowering `max_grad_norm` |
| `intervene` | Severity > 0.8 | Manual inspection of loss curves |
| `check_dataloader` | Throughput falling | Investigate DataLoader workers, I/O bottleneck |
| `none` | Normal | No action |

### EDGE Cases Handled

| Code | Situation | Fix |
|---|---|---|
| EDGE-1 | Multilabel task has no single label column for stratification | Falls back to KFold |
| EDGE-2 | `batch["labels"]` is a raw tensor instead of a per-task dict | Normalized to dict |
| EDGE-6 | `create_trainer_fn` raises before `trainer` is bound | `"trainer" in locals()` guard before `del trainer` |

### Debugging Checklist

1. **Loss is NaN from step 1:** Check label correctness, reduce LR, disable AMP briefly to rule out overflow.
2. **Grad norm is always 0:** Check that `zero_grad()` is not called before `compute_grad_norm()`.
3. **Early stopping triggers immediately:** Check `monitor_metric` key matches what evaluator emits; verify `maximize_metric` direction.
4. **DDP crash on first step:** Verify model is moved to CUDA before `build_optimizer` (N-CRIT-3 / GPU-1).
5. **Shape error in multilabel eval:** Ensure `valid_label_indices` is passed to both `LossEngineConfig` and `EvaluationConfig`.
6. **MLflow metrics stepping backwards:** Check that `step=global_step` is passed consistently to `tracker.log_metrics()`.

---

## 13. Extensibility Guide

### Adding a New Task

1. Register the task in `src/config/task_config.py` (`get_all_tasks`, `get_task_type`, `get_output_dim`).
2. Add a data contract in `src/data_processing/data_contracts.py`.
3. Add a model head in `src/models/multitask/multitask_truthlens_model.py`.
4. Add a loss router in `src/models/loss/task_loss_router.py`.
5. The task automatically appears in `create_multitask_trainer_fn` (it calls `get_all_tasks()`).
6. Add the task's weight to `config/config.yaml` under `task_weights`.

### Adding a New Tracking Backend

1. Extend `ExperimentTracker._init_backend()` with the new backend name.
2. Add the backend's `log_metric`, `log_params`, `log_artifact`, and `finish` calls in their respective methods, wrapped in `self._safe(...)`.
3. Set `backend="your_backend"` in `ExperimentTrackerConfig`.

### Adding a New Task Scheduler Strategy

1. Add a method `_your_strategy(self) -> str` to `TaskScheduler`.
2. Add a branch in `next_task()`:
   ```python
   elif strategy == "your_strategy":
       return self._your_strategy()
   ```
3. Set `strategy="your_strategy"` in `TaskSchedulerConfig`.

### Adding a New CV Metric Strategy

1. Extend `resolve_metric(task, metrics, strategy)` with a new `strategy` string.
2. Pass `metric_strategy="your_strategy"` to `cross_validate_task()`.

### Swapping the Optimizer

1. Modify `build_optimizer` in `src/models/optimization/optimizer_factory.py`.
2. The factory in `create_trainer_fn` calls `build_optimizer(model, lr, weight_decay)` — no changes needed in `src/training/`.

### Enabling Optional GradNorm Balancing

```python
# In create_trainer_fn or create_multitask_trainer_fn, after building TrainingStep:
from src.training.instrumentation import AutoDebugEngine

instrumentation = AutoDebugEngine(tasks=list(task_types.keys()), use_gradnorm=True)
# Pass shared_params=model.encoder.parameters() to instrumentation.step()
```

`GradNorm.compute()` will return per-task gradient-normed weights that can replace static `task_weights` in `LossEngineConfig`.

---

## 14. Common Pitfalls & Known Risks

### Critical (N-CRIT)

| Code | File | Issue | Fix |
|---|---|---|---|
| N-CRIT-1 | `monitor_engine.py` | `_extract_loss` raised RuntimeError on NaN before `anomaly_on_nan` policy ran | Returns `float("nan")` instead |
| N-CRIT-2 | `instrumentation.py` | `LossTracker.update` raised RuntimeError on NaN, crashing healthy tasks in same step | Skips EMA update, increments `_nan_counter` |
| N-CRIT-3 | `distributed_engine.py` | `wrap_model` called `model.to(device)` AFTER optimizer build, invalidating param refs | Raises RuntimeError if device mismatch detected |

### High Severity (N-HIGH)

| Code | File | Issue | Fix |
|---|---|---|---|
| N-HIGH-1 | `hyperparameter_tuning.py` | Objective silently dropped all base_params keys not in trial space | Merges `{**base_params, **trial_params}` |
| N-HIGH-3 | `experiment_tracker.py` | Explicit `step=` argument was unconditionally overwriting internal counter | Only auto-advances counter when step is None |

### Medium Severity (N-MED)

| Code | File | Issue | Fix |
|---|---|---|---|
| N-MED-1 | `instrumentation.py` | Allocated CUDA tensor per task per step for variance computation | Uses `statistics.variance` on CPU deque |
| N-MED-3 | `cross_validation.py` | All folds shared same RNG state (set once before loop) | `set_seed(seed + fold_id)` per fold |

### Low Severity (N-LOW)

| Code | File | Issue | Fix |
|---|---|---|---|
| N-LOW-2 | `evaluation_engine.py` | Regression eval didn't mask NaN sentinel labels | `torch.isfinite(labels)` mask applied |
| N-LOW-3 | `training_setup.py` | `optimize_model` must run before DDP wrap | Documented ordering constraint |
| N-LOW-5 | `hyperparameter_tuning.py` | Optuna runs bypassed ExperimentTracker's distributed-safe paths | Routes through `ExperimentTracker(group=...)` |
| N-LOW-6 | `trainer.py` | Cleanup (tracker.finish, distributed.cleanup) only ran on success | Moved to `try/finally` |
| N-LOW-7 | `task_scheduler.py` | Adaptive strategy convention undocumented | Inline comment + this doc |
| N-LOW-8 | `monitor_engine.py` | Duplicate L2 norm reduction loop | Delegates to `training_utils.compute_grad_norm` |

### GPU-Specific

| Code | File | Issue | Fix |
|---|---|---|---|
| GPU-1 | `create_trainer_fn.py` | Multiple redundant `model.to(device)` calls; optimizer built before move | Single `model.to(device)` before `build_optimizer` |
| GPU-2 | `evaluation_engine.py` | `non_blocking=True` set unconditionally, misleading for un-pinned tensors | Gated on `tensor.is_pinned()` |
| GPU-3 | `distributed_engine.py` | NCCL backend crashed on CPU-only hosts | Auto-fallback to `gloo` |
| GPU-4 | `evaluation_engine.py` | `model.to(device)` called on every evaluate(), breaking DDP wrappers | Device probe; only moves if different |

### Loss-Specific

| Code | File | Issue | Fix |
|---|---|---|---|
| LOSS-1 | `monitor_engine.py` | Two conflicting SpikeDetector implementations could fire in same step | Unified: MonitorEngine imports from instrumentation |
| LOSS-3 | `loss_engine.py` | Task weights silently divided by grad_accum in effective gradient | Pre-scale weights by grad_accum_steps at construction |
| LOSS-4 | `loss_functions.py` | NaN label sentinels in multilabel BCE poisoned loss | `isfinite` mask before reduction |

### Bug Resolutions (BUG-*)

| Code | File | Issue | Fix |
|---|---|---|---|
| BUG-7 | `create_trainer_fn.py` | Scheduler built with default 1000 training steps → LR collapsed early | Compute `steps_per_epoch × epochs` |
| BUG-8 | `hyperparameter_tuning.py` | Optuna defaulted to "minimize" for classification metrics | `_resolve_direction` returns "maximize" for classifiers |
| BUG-9 | `create_trainer_fn.py` | Optuna trial `epochs` not forwarded to Trainer | Passed via `params_override` |

---

## 15. Example Usage

### 15.1 Single-Task Training

```python
from transformers import AutoTokenizer
from src.training import create_trainer_fn

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

params = {
    "lr": 2e-5,
    "batch_size": 16,
    "epochs": 5,
    "tokenizer": tokenizer,
    "max_length": 512,
    "weight_decay": 0.01,
    "grad_accum": 4,
    "max_grad_norm": 1.0,
    "amp": True,
    "use_compile": True,
    "gradient_checkpointing": True,
    "monitor_metric": "bias_score",
    "maximize_metric": True,
}

trainer = create_trainer_fn(
    task="bias",
    train_df=train_df,
    val_df=val_df,
    params=params,
)

trainer.train()
metrics = trainer.evaluate()
print(metrics)  # {"bias_score": 0.87}
```

### 15.2 Multi-Task Training

```python
from src.training.create_multitask_trainer_fn import create_multitask_trainer_fn
from src.utils.settings import load_settings
from transformers import AutoTokenizer

settings = load_settings("config/config.yaml")
tokenizer = AutoTokenizer.from_pretrained(settings.model.model_name)

data_bundle = {
    "bias":       {"train": bias_train_df,    "val": bias_val_df},
    "ideology":   {"train": ideology_train_df, "val": ideology_val_df},
    "factuality": {"train": fact_train_df,    "val": fact_val_df},
    # ... other tasks
}

trainer = create_multitask_trainer_fn(
    settings=settings,
    data_bundle=data_bundle,
    tokenizer=tokenizer,
)

trainer.train()
```

### 15.3 Cross-Validation

```python
from src.training import cross_validate_task, create_trainer_fn

results = cross_validate_task(
    task="ideology",
    df=full_df,
    create_trainer_fn=create_trainer_fn,
    params=params,
    label_column="label",
    n_splits=5,
    seed=42,
    metric_strategy="auto",
)

print(f"CV mean: {results['mean']:.4f} ± {results['std']:.4f}")
print(f"Folds: {results['num_successful_folds']}/{5}")
```

### 15.4 Hyperparameter Tuning

```python
from src.training import tune_task, create_trainer_fn

result = tune_task(
    task="bias",
    df=full_df,
    create_trainer_fn=create_trainer_fn,
    n_trials=30,
    multi_objective=False,
    storage="sqlite:///optuna.db",  # optional persistence
    base_params={
        "tokenizer": tokenizer,
        "max_length": 512,
        "grad_accum": 4,
    },
    tracker_backend="mlflow",
)

print(result["best_params"])
print(f"Best score: {result['best_score']:.4f}")
```

### 15.5 Distributed Training (torchrun)

```bash
torchrun --nproc_per_node=4 train.py
```

```python
# train.py
from src.training.distributed_engine import DistributedEngine, DistributedConfig
from src.training.create_multitask_trainer_fn import create_multitask_trainer_fn

dist_engine = DistributedEngine(DistributedConfig(backend="nccl"))
dist_engine.initialize()

trainer = create_multitask_trainer_fn(
    settings=settings,
    data_bundle=data_bundle,
    tokenizer=tokenizer,
)

# Pass distributed engine through params_override or directly
trainer.distributed = dist_engine
trainer.train()
```

### 15.6 Experiment Tracking with MLflow

```python
from src.training.experiment_tracker import ExperimentTracker, ExperimentTrackerConfig

tracker = ExperimentTracker(ExperimentTrackerConfig(
    backend="mlflow",
    project_name="truthlens-prod",
    run_name="roberta-base-v2",
    tracking_uri="http://localhost:5000",
    tags={"task": "bias", "env": "prod"},
))

tracker.log_config(params)
# ... training ...
tracker.log_metrics({"eval/bias_score": 0.87}, step=1000)
tracker.finish()
```

### 15.7 Loss Balancing Inspection

```python
from src.training.loss_balancer import plan_for_dataframe, LossBalancerConfig

plan = plan_for_dataframe(
    train_df,
    label_columns=["label"],
    task_type="multiclass",
    num_classes=3,
    config=LossBalancerConfig(weight_threshold=0.6, focal_threshold=0.85),
)

print(plan.notes)
# ["class_weights enabled (max_ratio=0.72 >= 0.6)"]
print(plan.class_weights)
# tensor([0.45, 1.20, 0.85])
```

---

## 16. Simple Explanation (For Non-Technical Reviewers)

### What Does This Code Do?

Imagine you are training a team of six specialist detectives, each responsible for spotting a different kind of misinformation: bias, ideology, propaganda, factuality, emotions used deceptively, and narrative framing. These six detectives share one shared brain (the encoder — the part that reads and understands text) but each has their own specialized judgment (a "head" — the part that makes the final call for their specialty).

The `src/training/` folder is the **training academy** for these detectives.

### How Does Training Work?

1. **Setup:** The academy first checks that everything is in order — the right tools are ready, the training materials are prepared, and a practice drill is run to catch any problems before real training begins.

2. **Each day of training (an epoch):** The academy picks a batch of news articles and shows them to all six detectives simultaneously. Each detective makes their best guess. A scoring system (the loss function) measures how wrong they were. The more wrong they were, the more they learn.

3. **Harder subjects get more attention:** If one detective is struggling more than the others (their loss is higher), the scheduler picks their practice cases more often. This is like a teacher spending extra time on students who need it most.

4. **Preventing cheating:** To make sure the learning is genuine, a "gradient clipper" stops any single piece of feedback from being so extreme that it throws off everything else — like a sensible coach who won't let a player go from "zero to hero" in one practice.

5. **Progress check (validation):** After each day, the detectives are tested on cases they haven't seen before. Their scores are recorded.

6. **Knowing when to stop:** If the scores stop improving for several days in a row, training ends early. There's no point continuing if nothing is getting better.

### What Happens If Something Goes Wrong?

The training academy has an automatic monitoring system that watches for three main problems:

- **Loss spikes:** If a detective suddenly gets wildly confused on a batch (the loss spikes up dramatically), the academy automatically reduces the intensity of the training session to prevent it from spiraling.

- **Disappearing/exploding gradients:** If the feedback signals become too tiny to learn from (vanishing) or so huge they corrupt everything (exploding), the system detects this and takes corrective action.

- **NaN (Not a Number) losses:** If the math completely breaks down and produces undefined values, training is flagged and can be safely stopped before corrupting the model.

### Why Are There So Many Files?

Each file handles one distinct job — just like a real organization has separate departments:

| File | Real-World Analogy |
|---|---|
| `trainer.py` | The academy director — runs the overall schedule |
| `training_step.py` | The drill instructor — runs each individual practice session |
| `loss_engine.py` | The grading system — scores how well each detective did |
| `evaluation_engine.py` | The exam board — runs the fair tests at end of each day |
| `monitor_engine.py` | The health monitor — watches for signs of stress or breakdown |
| `experiment_tracker.py` | The record keeper — logs all scores and settings to a database |
| `distributed_engine.py` | The multi-campus coordinator — lets multiple machines train in parallel |
| `cross_validation.py` | The auditor — runs the same experiment multiple times to confirm results |
| `hyperparameter_tuning.py` | The optimizer — experiments with different training intensities to find the best approach |
| `task_scheduler.py` | The timetable planner — decides which detective gets practice next |

### Why Does This Matter?

Getting the training right is critical for TruthLens. If the training is biased toward easy examples, the detectives won't learn to spot subtle misinformation. If the training is numerically unstable, the model produces garbage. If experiment tracking is broken, the team cannot reproduce successful runs or explain their results. Every piece in `src/training/` solves a specific, well-understood failure mode — documented in the code and in this file.
