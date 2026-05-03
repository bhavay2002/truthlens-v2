# `src/training/` — Production-Grade Audit

**Scope:** every file under `src/training/` (15 modules) and every cross-boundary touchpoint with `src/data_processing`, `src/models`, `src/evaluation`, `src/config`, `src/monitoring`, `src/utils`.

**Date:** 2026-04-28
**Auditor:** Replit Agent
**Build context:** TruthLens AI · FastAPI / PyTorch 2.5.1 (CPU-only) · transformers 4.46.3 · Python 3.11

---

## 1 · Files audited

| # | File | LOC | Purpose |
|---|------|----:|---------|
| 1  | `trainer.py`               | 426 | Top-level orchestrator (epoch loop, val, ES, ckpt) |
| 2  | `training_step.py`         | 553 | Per-step engine (fwd / loss / bwd / opt / monitor) |
| 3  | `training_setup.py`        | 239 | Runtime + AMP + sanity check + `torch.compile` |
| 4  | `training_utils.py`        | 313 | Device, grad, throughput, precision helpers |
| 5  | `loss_engine.py`           | 297 | Wrapper around `MultiTaskLoss` (single + multi) |
| 6  | `loss_functions.py`        | 126 | Pure binary / multiclass / multilabel / regression |
| 7  | `evaluation_engine.py`     | 338 | Streaming metrics + DDP-correct reductions |
| 8  | `distributed_engine.py`    | 181 | DDP init / wrap / sampler / collectives |
| 9  | `create_trainer_fn.py`     | 307 | The single factory that wires everything |
| 10 | `cross_validation.py`      | 251 | StratifiedKFold-driven CV + dashboard |
| 11 | `experiment_tracker.py`    | 231 | MLflow / W&B / no-op backend |
| 12 | `hyperparameter_tuning.py` | 324 | Optuna study + objective + multi-task loop |
| 13 | `instrumentation.py`       | 472 | `AutoDebugEngine` (EMA, spike, gradnorm, severity) |
| 14 | `monitor_engine.py`        | 253 | EMA + health + spike + action policy |
| 15 | `task_scheduler.py`        | 187 | round-robin / random / weighted / adaptive |

Total: **~4,700 LOC** of training code.

---

## 2 · Architecture map (verified)

```
                           create_trainer_fn(task, train_df, val_df, params)
                                            │
            ┌───────────────────────────────┼──────────────────────────────────┐
            ▼                               ▼                                  ▼
       build_model        build_dataset / build_dataloader              get_task_type
   (src.models.registry)  (src.data_processing)                         (src.config)
            │                               │
            └──> model.to(device)  <------- GPU-1 invariant: ONE move, BEFORE optimizer
                       │
                       ▼
                build_optimizer(model)            ◄──── (src.models.optimization)
                build_scheduler(opt, cfg)
                       │
       ┌───────────────┼─────────────────────────────────────────┐
       ▼               ▼                                          ▼
  LossEngine      MonitoringEngine    TaskScheduler   ExperimentTracker
                                                           │
                                                           ▼
                                      TrainingStep ── (run per batch) ──┐
                                                                         │
                                  EvaluationEngine ── streaming, DDP-safe┤
                                                                         ▼
                                                                     Trainer
                                                                         │
                                            ┌────────────────────────────┼────────────┐
                                            ▼                            ▼            ▼
                                CheckpointEngine               DistributedEngine   tracker
                            (src.models.checkpointing)            (DDP wrap)       (logs)
```

The boundaries are **clean**: `src.training` does not import sideways into `src.evaluation` (it uses its own `EvaluationEngine`), it correctly defers loss math to `src.models.loss.MultiTaskLoss`, and the only `src.config` touchpoint is `get_task_type`.

---

## 3 · Confirmed-resolved issues (regression baseline)

The following defects were already present as fixed-with-comment in the code at audit time; verified by reading the implementation and adjacent comments. **DO NOT regress**.

| Tag | Location | What it fixes |
|-----|----------|---------------|
| GPU-1 | `create_trainer_fn.py:90`, `trainer.py:97-110`, `training_step.py:102-117` | Single `model.to(device)` BEFORE `build_optimizer`. Two formerly-redundant moves now only validate device match. |
| GPU-2 | `training_utils.py:76-103`, `training_step.py:509-517` | `non_blocking=True` is gated on `tensor.is_pinned() and dst.type=='cuda'`. |
| GPU-3 | `distributed_engine.py:71-89` | NCCL backend auto-falls back to gloo on CPU-only hosts. |
| GPU-4 | `evaluation_engine.py:176-189` | `model.to(device)` only when device actually differs (preserves DDP wrap). |
| MT-1 | `loss_engine.py:88-145, 151-167` | Single-task path force-disables EMA normalizer / coverage / `attach_balancer`. |
| MT-2 | `evaluation_engine.py:221-231, 275-291` | `binary` task type wired through `StreamingF1` + sigmoid threshold. |
| MT-3 | `training_step.py:156-503`, `training_setup.py:138-148` | `dry_run=True` runs forward+loss+backward without mutating optimizer / sched / monitor / scaler / task scheduler / tracker. |
| LOSS-1 | `monitor_engine.py:18, 141`, `training_step.py:449-470` | One `SpikeDetector` shared between monitor + instrumentation; LR reduction de-duped per step. |
| LOSS-2 | `training_step.py:202-219, 525-528` | `_filter_batch` removed — full multi-task forward each step. |
| LOSS-3 | `loss_engine.py:71-83`, `create_trainer_fn.py:212-214` | Per-task weights pre-scaled by `grad_accum` to compose correctly with `loss / grad_accum`. |
| LOSS-4 | `loss_functions.py:73-113` | Multilabel ignore-mask now `isfinite & != ignore_index` (NaN labels skipped). |
| BUG-4 | `trainer.py:282-289, 338-355` | Checkpoint persists optimizer / scheduler / scaler state. |
| BUG-5 | `training_step.py:328-358` | `unscale_` BEFORE `clip_grad_norm_`; gated on accumulation boundary. |
| BUG-6 | `training_step.py:380-385, 537-549` | Scheduler advanced only on successful AMP step; `_reduce_lr` patches `base_lrs`. |
| BUG-7 | `create_trainer_fn.py:179-200` | Real `num_training_steps` threaded into LambdaLR. |
| BUG-9 | `trainer.py:113-126`, `create_trainer_fn.py:297` | Optuna `epochs` honored via `params_override`. |
| BUG-10| `loss_engine.py:199-211` | `shared_parameters` forwarded into MultiTaskLoss; balancer hooks not double-invoked. |
| REC-1 | `loss_engine.py:213-226` | Removed 2N+2 redundant `torch.isfinite` syncs per step. |
| REC-2 | `loss_engine.py:238-247` | Removed unused `mean_loss.item()` host sync. |
| REC-3 | `instrumentation.py:306-326`, `training_step.py:340-358, 424-441` | `cached_grad_norm` reuses the `clip_grad_norm_` value across all three sites. |
| PERF-1| `evaluation_engine.py:29-150, 300-302` | Streaming metrics keep accumulators on-device, sync only at `compute()`. |
| PERF-2| `create_trainer_fn.py:138-167`, `evaluation_engine.py:308-324` | Validation sharded across DDP ranks; metrics merge raw `(num, den)` pairs. |
| PERF-3| `training_setup.py:217-229` | `torch.compile` gated on CUDA; warns on failure. |
| PERF-5| `training_step.py:129-150` | Feature-dict logging does **one** `.cpu().tolist()` per tensor. |
| CFG-2 | `training_step.py:37-42, 530-549` | `spike_lr_scale` factored out of two hardcoded sites. |
| CFG-3 | `trainer.py:147-161, 418-420` | `log_every_steps` / `checkpoint_every_steps` are constructor args. |
| CFG-4 | `trainer.py:74-81, 422-423` | `setup_config` overridable (frozen dataclass escape hatch). |
| CFG-5 | `loss_engine.py:25-41` | `normalization` strategy documented + auto-overridden for single-task. |
| CFG-6 | `task_scheduler.py:45-60, 85-87` | Single-task short-circuit + warning when non-trivial strategy is wired. |

These represent the bulk of the prior audit work and are all in good shape.

---

## 4 · NEW findings (this audit)

The following are **new** defects introduced or never previously addressed. They are not yet annotated in code.

### 🔴 CRITICAL — will crash or silently corrupt training

#### N-CRIT-1 · `MonitoringEngine._extract_loss` raises BEFORE the `anomaly_on_nan` guard runs

**File:** `monitor_engine.py:115-219`

```python
def update(self, outputs, *, model=None, batch_size=None):
    self.step += 1
    loss = self._extract_loss(outputs)               # ← raises on non-finite tensor
    if self.config.anomaly_on_nan and not torch.isfinite(torch.tensor([loss])):
        return self._build_metrics(loss, None, None, 0.0, MonitorAction.NAN)  # ← unreachable
```

`_extract_loss` (L207-219) calls `torch.isfinite(loss)` and raises `RuntimeError` for any non-finite tensor — meaning the `anomaly_on_nan` policy at L131 is **dead code** for the common path (tensor losses). Any non-finite loss that reaches the monitor (e.g. caller bypasses `TrainingStep`'s `skip_nan_loss` and passes the raw value here) will crash the run instead of returning a graceful `MonitorAction.NAN`.

**Fix:** in `_extract_loss`, return `float('nan')` (or the raw `.item()`) for non-finite tensors and let the caller's `anomaly_on_nan` guard at L131 decide policy.

---

#### N-CRIT-2 · `instrumentation.LossTracker.update` crashes the run on any non-finite per-task loss

**File:** `instrumentation.py:41-59`

```python
def update(self, losses):
    for t, v in losses.items():
        val = _to_float(v)
        if not _is_finite(val):
            raise RuntimeError(f"Non-finite loss: {t}")          # ← un-trapped
```

`TrainingStep.run` (L432-441) calls `self.instrumentation.step(losses=task_losses, ...)` with **per-task** losses. Even when the **aggregate** `total_loss` is finite (and so passes the L283 `skip_nan_loss` check), an individual masked task can be non-finite (e.g. a head that received only `ignore_index` labels in this batch and hits a degenerate path inside `MultiTaskLoss`). The `LossTracker` then raises and crashes the run, **bypassing the entire `skip_nan_loss=True` contract** that `TrainingStep` advertises. Worse, `AnomalyClassifier.classify` already exists to *report* `nan_loss` non-fatally — `LossTracker` shouldn't be raising at all.

**Fix:** clamp / skip the EMA update on non-finite values (e.g. record `self._failure_counter['nan_loss'] += 1` and `continue`). The classifier downstream will still emit `failure="nan_loss"` so the policy layer keeps its `stop_training` action.

---

#### N-CRIT-3 · `DistributedEngine.wrap_model` re-moves the model AFTER `build_optimizer`

**File:** `distributed_engine.py:110-129` + `trainer.py:163-168`

```python
# trainer.py
if self.distributed:
    self.distributed.initialize()
    self.model = self.distributed.wrap_model(self.model)        # ← AFTER opt built

# distributed_engine.py
def wrap_model(self, model):
    device = torch.device(f"cuda:{self.local_rank}")
    model = model.to(device)                                    # ← VIOLATES GPU-1
    model = DDP(model, device_ids=[self.local_rank], ...)
```

This is the **exact** invariant `create_trainer_fn.py:80-90` documents and protects against. The single-process / single-GPU path is fine because `device == cuda:0` and the move is a no-op, but on a true multi-rank launch where `LOCAL_RANK > 0`, the optimizer was built against `cuda:0` parameters and the in-place move to `cuda:LOCAL_RANK` invalidates every reference. First `optimizer.step()` then raises “expected all tensors to be on the same device”.

**Fix:** assert that `next(model.parameters()).device == torch.device(f'cuda:{local_rank}')` and refuse to re-move. The correct DDP recipe is: `model.to(local_rank)` BEFORE `build_optimizer` (already the contract in `create_trainer_fn`). `wrap_model` should *only* call `DDP(...)`. Alternatively, build the optimizer inside `wrap_model` (rejected: breaks separation of concerns).

---

### 🟠 HIGH — wrong by construction, fails on the first realistic call

#### N-HIGH-1 · `hyperparameter_tuning.objective` cannot satisfy `_validate_params`

**File:** `hyperparameter_tuning.py:107-136` + `create_trainer_fn.py:33-39`

```python
# objective() builds:
params = {"lr": ..., "batch_size": ..., "epochs": ..., "weight_decay": ...}
cv_result = cross_validate_task(..., params=params)     # → create_trainer_fn(params)

# create_trainer_fn._validate_params requires:
required = ["lr", "batch_size", "tokenizer"]            # ← tokenizer never present
```

Every Optuna trial will raise `ValueError("Missing required param: tokenizer")` and `objective` re-raises as `optuna.TrialPruned` — meaning **every trial is pruned** and `study.best_value` accesses `study.best_trial` on an empty study (`ValueError`). Same defect for `num_workers`, `monitor_config`, `tracker_config` — the tuner has no way to thread Trainer-level dependencies through.

**Fix:** accept a `base_params: Dict[str, Any]` argument on `tune_task` and merge: `params = {**base_params, **trial_params}`. Document in the docstring that `tokenizer` (and any non-Optuna deps) must live in `base_params`.

---

#### N-HIGH-2 · `cross_validation.build_splits` always uses `StratifiedKFold` — crashes for multilabel/regression

**File:** `cross_validation.py:61-79`

```python
splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
return list(splitter.split(df, y))
```

`StratifiedKFold.split(X, y)` requires `y` to be a 1-D array of class indicators. For:
* `multilabel` tasks (e.g. `framing_techniques`, `propaganda_techniques`) `y` is `np.ndarray` of lists or 2-D indicator → `ValueError: Supported target types are: ('binary', 'multiclass'). Got 'multilabel-indicator'`.
* `regression` tasks (e.g. `bias_intensity`) `y` is continuous → same error.

**Fix:**
```python
ttype = get_task_type(task)
if ttype == "multilabel":
    splitter = MultilabelStratifiedKFold(...)   # iterstrat package, OR fall back to KFold
elif ttype == "regression":
    splitter = KFold(...)                       # or StratifiedKFold over qcut-binned y
else:
    splitter = StratifiedKFold(...)
```

The `task` argument is *already* in scope inside `cross_validate_task`; pass it down to `build_splits`.

---

#### N-HIGH-3 · `ExperimentTracker.log_metrics` auto-increments `_step` even when an explicit `step` is passed

**File:** `experiment_tracker.py:124-157`

```python
def log_metrics(self, metrics, *, step=None, prefix=None):
    ...
    step = step if step is not None else self._step
    self._step = step + 1                       # ← mutated regardless of branch
```

Two bugs in one:
1. **Step collision.** `Trainer._train_epoch` (L259-270) and `Trainer.train` (L218-219) both call `tracker.log_metrics(..., step=self.global_step)` — which is the same value when validation runs immediately after the last training step of the epoch. With `global_step=N` from training and `global_step=N` again from val, both calls have `step=N`. W&B drops the second; MLflow keeps both but plots are non-monotonic. The auto-increment then sets `self._step = N+1` which collides with the *next* training-step log.
2. **Internal counter drift.** Every explicit-step call resets the implicit counter to `step+1`, so callers that mix explicit and implicit step semantics will see the implicit counter jumping around unpredictably.

**Fix:**
```python
if step is None:
    step = self._step
    self._step += 1
# else: do NOT touch self._step
```

Also: when training and validation share `global_step`, validation should log under a `val/` prefix and pass `step=self.global_step` *with the prefix already set* — but the underlying step-collision is the tracker's contract, not the caller's.

---

### 🟡 MEDIUM — correctness-preserving but expensive or fragile

#### N-MED-1 · `instrumentation.LossStats.update` allocates a fresh tensor every step

**File:** `instrumentation.py:71-84`

```python
var = float(torch.tensor(list(h)).var(unbiased=True).item()) if len(h) > 1 else 0.0
```

`list(h)` materialises up to 50 elements per task per step, allocates a new CPU `Tensor`, computes `.var()` (which allocates intermediates), then `.item()` syncs. Per step cost: O(W × T) Python-loop allocation × T tasks. Use **Welford's online variance** (constant memory, no allocation) or `statistics.pvariance(h)` (pure-Python, faster than tensor allocation for W ≤ 100).

```python
import statistics
mean = statistics.fmean(h)
var = statistics.pvariance(h, mu=mean) if len(h) > 1 else 0.0
```

---

#### N-MED-2 · `training_step.py` uses hardcoded `step % 50 == 0` for feature observability

**File:** `training_step.py:225`

```python
if step % 50 == 0:  # avoid slowdown
    ...
    log_feature_stats(...)
    log_feature_summary(...)
```

The Trainer already exposes `log_every_steps` (CFG-3), but the feature-observability gate is hardcoded — so a fast Optuna trial with `log_every_steps=1` still only logs features every 50 steps, and a long production run with `log_every_steps=500` over-logs features 10× more often than it logs metrics. Drive the cadence from the same knob (or add a `feature_log_every_steps`).

---

#### N-MED-3 · `cross_validation.cross_validate_task` doesn't seed-vary across folds

**File:** `cross_validation.py:99, 119-124`

```python
set_seed(seed)                                  # ← outer, once
for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):
    trainer = create_trainer_fn(task=task, ..., params=params)   # → set_seed(params.seed=42)
```

Every fold ends up calling `set_seed(42)` (or whatever `params["seed"]` is), so dropout masks and shuffle orders are *identical across folds*. Real CV variance is then under-reported (mean is unaffected, `std` is artificially low). Pass `seed = base_seed + fold_id` into a per-fold copy of params.

---

#### N-MED-4 · `cross_validate_task` `del trainer` references potentially-unbound name

**File:** `cross_validation.py:115-169`

```python
try:
    trainer = create_trainer_fn(...)
    ...
except Exception:
    logger.exception("Fold %d failed", fold_id)
finally:
    try:
        del trainer                             # ← UnboundLocalError if create_trainer_fn raised
    except Exception:
        pass
```

The `try/except` traps `UnboundLocalError`, so this is *currently* harmless — but it's a correctness landmine. Use `locals().pop('trainer', None)` or initialise `trainer = None` before the `try`.

---

#### N-MED-5 · `Trainer.monitor_metric` default never matches `EvaluationEngine` output keys

**File:** `trainer.py:45, 212-215` + `evaluation_engine.py:308-324`

`Trainer.__init__(monitor_metric="val_loss")` and `_update_early_stopping` reads `val_metrics.get(self.monitor_metric)`. But `EvaluationEngine._compute_metrics` only returns `{task}_score` keys — never `val_loss` (loss is consumed inside metrics, not exposed). Result: `metric_value = None` on every epoch → early stopping never increments `no_improve_epochs` → the run consumes the full epoch budget regardless of plateau. `create_trainer_fn` *does* override via `params.get("monitor_metric", "val_loss")` so the bug is dormant unless the caller knows to pass e.g. `"binary_score"` — which is undiscoverable without reading both files.

**Fix:** either (a) return `val_loss` from `EvaluationEngine.evaluate` (the engine has access to model.eval logits — could compute it cheaply alongside metrics), OR (b) auto-derive the default `monitor_metric` from `task` inside `create_trainer_fn` (`f"{task}_score"`) and emit a warning when the caller leaves the default.

---

### 🔵 LOW — cosmetic / future-proofing

#### N-LOW-1 · Deprecated AMP API across three files

* `training_step.py:119` `torch.cuda.amp.GradScaler(enabled=...)`
* `training_step.py:259` `torch.cuda.amp.autocast(enabled=...)`
* `training_setup.py:96, 100, 103-107` same.

Torch 2.4+ recommends `torch.amp.GradScaler("cuda", ...)` and `torch.amp.autocast("cuda", ...)`. The current code emits a `FutureWarning` per training run on torch 2.5.1. Functionally identical for now; will break in a future release.

---

#### N-LOW-2 · `evaluation_engine._update_metrics` mask uses `int` ignore_index for float multilabel labels

**File:** `evaluation_engine.py:267-273`

```python
elif ttype == "multilabel":
    preds = (torch.sigmoid(logits) > self.config.threshold).int()
    mask = labels != self.config.ignore_index           # ignore_index: int = -100
```

`loss_functions.LOSS-4` documents that multilabel labels may use `NaN` as a sentinel (and that `ignore_index` may be a float `-100.0`). The eval mask here only matches the integer `-100` cast, doesn't capture `NaN` at all (NaN comparisons are always False), and doesn't match the same ignore semantics the loss layer uses. Validation accuracy will silently include "ignored" positions.

**Fix:** mirror `multilabel_loss`'s mask:
```python
mask = torch.isfinite(labels) & (labels != self.config.ignore_index)
```

---

#### N-LOW-3 · `optimize_model` runs after the optimizer is built

**File:** `training_setup.py:212-238` + `trainer.py:85`

```python
self.model = optimize_model(self.model)     # in Trainer.__init__, AFTER create_trainer_fn
                                            # has already built the optimizer
```

`torch.compile` does not relocate parameter storage (so the optimizer's parameter refs stay valid), but it *can* swap module identity, which breaks any code that later does `isinstance(model, X)` (incl. `_unwrap_model`'s `isinstance(self.model, DistributedDataParallel)` check — `torch.compile`-wrapped DDP returns a `OptimizedModule`, not `DDP`). On CUDA the call is gated to compile only post-DDP-wrap or pre — currently it runs at L85, *before* `self.distributed.wrap_model` at L168, so compile + DDP could end up in either order depending on whether `distributed` is set. The interaction is undocumented.

**Fix:** make the order explicit: compile *after* DDP wrap if both are enabled, never before. Or ban the combination until tested. The current `mode="reduce-overhead"` (CUDA Graphs) is also incompatible with DDP without `static_graph=True`; document the constraint.

---

#### N-LOW-4 · `Trainer.cfg = ModelConfigLoader.load_multitask_config(config_path)` may fail with default empty path

**File:** `trainer.py:56`, `create_trainer_fn.py:286`

`create_trainer_fn` passes `config_path=params.get("config_path", "")` and `Trainer.__init__` immediately calls `ModelConfigLoader.load_multitask_config("")`. Whether this raises depends on the loader's empty-string handling (not in scope of this audit), but the contract is fragile — `Trainer` requires the YAML *only* to read `training.num_epochs` and `training.early_stopping_patience`, both of which are *also* present in `params_override`. The YAML lookup is dead unless the caller passes `config_path`.

**Fix:** make `config_path` optional and only load when truthy, falling back to defaults from `params_override`.

---

#### N-LOW-5 · `hyperparameter_tuning.init_tracking` duplicates `ExperimentTracker`

**File:** `hyperparameter_tuning.py:21-92` vs `experiment_tracker.py`

Both implement MLflow + W&B init / log / finalize. They drift independently — e.g. `init_tracking` uses `mlflow.log_metric(k, float(v))` per key in a loop; `ExperimentTracker.log_metrics` does the same but flattens nested dicts and adds `time/elapsed`. A bug fixed in one will not propagate to the other (already true: `_safe` swallow logic exists only in `ExperimentTracker`).

**Fix:** delete `init_tracking`/`finalize_tracking`/`log_trial` and route Optuna trials through `ExperimentTracker(group=f"tune_{task}")`. The `group` field already exists on `ExperimentTrackerConfig` (L22) for exactly this purpose.

---

#### N-LOW-6 · `Trainer.cleanup` skipped on exception

**File:** `trainer.py:182-238`

`tracker.finish()` and `distributed.cleanup()` only run on the **happy path** at L234-238. Any exception inside `_train_epoch` or `evaluate` will leak the MLflow run / W&B handle / NCCL process group. Wrap the whole `train()` body in `try/finally`.

---

#### N-LOW-7 · `TaskScheduler._adaptive` softmax uses *higher* loss → *higher* probability

**File:** `task_scheduler.py:122-138`

```python
scores = [self._ema_losses[t] for t in self.tasks]
scaled = [s / self.config.temperature for s in scores]
exp = [math.exp(x) for x in scaled]
```

The intent of an "adaptive" scheduler (cf. Sener & Koltun 2018) is usually to give *more* training to *harder* tasks → softmax of *positive* loss is correct iff "hard task = high loss = sample more often". That **is** the implemented behaviour, so this is **likely intentional**, but the absence of a sign comment means a future maintainer will flip it. Add a one-line comment fixing the convention.

---

#### N-LOW-8 · `compute_grad_norm` (training_utils) and `_compute_grad_norm` (training_setup) and `MonitoringEngine._compute_grad_norm` (monitor_engine) and `GradTracker.update` (instrumentation) are **four** copies of the same loop

Already de-duplicated *at runtime* by REC-3's `cached_grad_norm` plumbing — but the four implementations remain in the source tree and will drift. Pick one (`training_utils.compute_grad_norm`) and have the other three call it.

---

## 5 · Cross-module integration audit

| Boundary | Verdict |
|----------|---------|
| `src.training` → `src.models.registry.model_factory.build_model` | ✅ Contract honoured (`task`, `config=params`). |
| `src.training` → `src.models.optimization.optimizer_factory.build_optimizer` | ✅ `model`, `lr`, `weight_decay`. |
| `src.training` → `src.models.optimization.lr_scheduler.build_scheduler` | ✅ `num_training_steps` correctly threaded (BUG-7). |
| `src.training` → `src.models.loss.MultiTaskLoss` | ✅ via `LossEngine`; `shared_parameters` forwarded (BUG-10). |
| `src.training` → `src.models.checkpointing.CheckpointEngine` | ✅ optimizer/scheduler/scaler persisted (BUG-4). |
| `src.training` → `src.data_processing.dataloader_factory` | ✅ `DataLoaderConfig` correctly populated; resolved `num_workers` / `pin_memory` honoured. |
| `src.training` → `src.data_processing.dataset_factory.build_dataset` | ✅ `task`, `df`, `tokenizer`, `max_length`. |
| `src.training` → `src.config.task_config.get_task_type` | ✅ used in `cross_validation.resolve_metric`, `loss_engine`, `hyperparameter_tuning._resolve_direction`. **MISSING:** `cross_validation.build_splits` (N-HIGH-2). |
| `src.training` → `src.evaluation.*` | ✅ Not used; `EvaluationEngine` is self-contained inside `src.training`. (This is consistent with the architecture map but means metrics are duplicated between `src.training.evaluation_engine` and `src.evaluation` — out of audit scope.) |
| `src.training` → `src.monitoring.feature_logger` | ⚠️ Hardcoded cadence (N-MED-2). |
| `src.training` → `src.utils.seed_utils.set_seed` | ⚠️ Same seed every fold (N-MED-3). |

---

## 6 · Concurrency / DDP correctness review

| Concern | Status |
|---------|--------|
| Single `model.to()` before optimizer build | ✅ in `create_trainer_fn`; ❌ violated in `DistributedEngine.wrap_model` (N-CRIT-3). |
| Validation sharded across ranks | ✅ (PERF-2). |
| Streaming metrics use SUM-reduce of raw counters | ✅ (PERF-2 in `evaluation_engine`). |
| `set_epoch` called on `DistributedSampler` | ✅ (`trainer.py:197-198`). |
| `barrier()` before validation | ✅ (`trainer.py:207-208`). |
| `is_main_process` gate on tracker / checkpoint / log | ✅. |
| Gloo fallback when NCCL unavailable | ✅ (GPU-3). |
| `cleanup()` always called | ❌ (N-LOW-6). |

---

## 7 · Numerical / AMP correctness review

| Concern | Status |
|---------|--------|
| `unscale_` before `clip_grad_norm_` | ✅ (BUG-5). |
| Scheduler advance gated on successful AMP step | ✅ (BUG-6). |
| `skip_nan_loss` uniform across forward+loss | ✅ (`training_step.py:258-277`). |
| `skip_nan_loss` honoured by instrumentation | ❌ (N-CRIT-2). |
| `skip_nan_loss` honoured by monitor | ❌ (N-CRIT-1). |
| Multilabel ignore-mask handles NaN labels | ✅ in loss; ❌ in eval (N-LOW-2). |
| AMP API on torch 2.5+ | ⚠️ deprecated (N-LOW-1). |

---

## 8 · Performance review

| Hot path | Cost / step before fixes | After current code | Remaining gain |
|----------|--------------------------|--------------------|----------------|
| `torch.isfinite` syncs in loss layer | 2N+2 | 1 (REC-1) | — |
| `mean_loss.item()` host sync | 1 | 0 (REC-2) | — |
| Grad-norm computation | 3× | 1× (REC-3, cached) | — |
| Feature-dict logging | O(items × keys) syncs | 1 sync per key (PERF-5) | — |
| Streaming-metric host syncs in val | per batch per metric | once at compute (PERF-1) | — |
| `LossStats` variance | 1 tensor alloc / step / task | same | **N-MED-1** (Welford) |
| `_compute_grad_norm` duplicate loops | 4 copies | 1 effective at runtime, 4 in source | **N-LOW-8** (consolidate) |

---

## 9 · Edge cases

| Case | Handled | Where |
|------|---------|-------|
| Empty dataloader | partial — `_train_epoch` simply iterates `for batch in self.train_loader` (no batches → epoch is silent no-op, no warning). |
| Batch is not a dict | ✅ `training_step.py:189-194` raises with a clear error. |
| NaN / Inf labels (multilabel) | ✅ `loss_functions.LOSS-4`; ❌ `evaluation_engine` (N-LOW-2). |
| Imbalanced binary | ✅ `binary_loss` accepts `pos_weight`. |
| Single-task LossEngine with balancer attached | ✅ explicitly rejected (MT-1). |
| Single-task scheduler with non-trivial strategy | ✅ warned + short-circuited (CFG-6). |
| AMP gradient overflow | ✅ scheduler skipped (BUG-6). |
| Optimizer momentum / LR step / loss-scale on resume | ✅ persisted (BUG-4). |
| CV with multilabel / regression | ❌ `StratifiedKFold` will crash (N-HIGH-2). |
| Optuna tuning with required `tokenizer` param | ❌ every trial pruned (N-HIGH-1). |
| Tracker step collisions between training and validation logs | ❌ (N-HIGH-3). |
| Multi-rank DDP with optimizer built before wrap | ❌ (N-CRIT-3). |

---

## 10 · Recommended remediation order

1. **N-CRIT-1, N-CRIT-2** — cheap, unblocks `skip_nan_loss=True` end-to-end. Fix together with one helper `_safe_finite(loss, default=float('nan'))`.
2. **N-CRIT-3** — required before any real DDP launch.
3. **N-HIGH-1** — unblocks Optuna; otherwise the entire `hyperparameter_tuning.py` module is unreachable.
4. **N-HIGH-2** — unblocks CV for any non-multiclass task.
5. **N-HIGH-3** — required before *any* non-trivial tracker dashboard.
6. **N-MED-1..5** — quality-of-life and correctness-of-stats.
7. **N-LOW-1..8** — schedule into the next maintenance pass.

Estimated effort (single contributor): **CRIT 2 h, HIGH 3 h, MED 3 h, LOW 4 h ≈ 1.5 days.**

---

## 11 · Final score

| Dimension | Score | Notes |
|-----------|------:|-------|
| Architecture & module boundaries | **9 / 10** | Clean separation; only `evaluation_engine` duplication with `src.evaluation` is debatable. |
| Single-process correctness | **8 / 10** | Solid; the three CRIT defects are silent or rare-path. |
| Multi-rank / DDP correctness | **6 / 10** | N-CRIT-3 blocks any real launch. |
| AMP / numerical safety | **7 / 10** | Strong on the happy path; `skip_nan_loss` is bypassed by instrumentation+monitor. |
| Performance hygiene | **9 / 10** | Excellent — almost all GPU↔CPU syncs are already eliminated; only `LossStats` and the four-copy grad-norm loop remain. |
| Configurability | **8 / 10** | CFG-2..6 all landed; missing knobs for feature-log cadence and CV per-fold seed. |
| Test coverage / edge cases | **6 / 10** | Multilabel/regression CV will crash; tuning pipeline cannot run as wired. |
| Observability / tracker contract | **6 / 10** | Step-collision (N-HIGH-3) corrupts dashboards; duplicated tracker code in tuning. |
| Code quality (DRY / deprecations / naming) | **7 / 10** | Four grad-norm loops; deprecated AMP API; otherwise clean. |
| Documentation in code (audit comments) | **9 / 10** | The existing GPU/MT/LOSS/BUG/REC/PERF/CFG annotations are exemplary — preserve this discipline for the new findings. |

**Overall production-readiness: 7.5 / 10.**

Translated: *"works correctly on a single CPU/GPU happy path, has documented and disciplined fixes for the known sharp edges, but has three new critical defects that will surface the moment the system is exercised under (a) non-finite per-task losses, (b) multi-rank DDP, or (c) hyperparameter tuning / cross-validation on anything other than a multiclass task."*

Address the three CRIT items and the three HIGH items and the score moves to **9.0 / 10**.
