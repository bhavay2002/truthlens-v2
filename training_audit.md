# TruthLens AI — `src/training/` Production Audit

**Audit date:** 2026-05-03  
**Scope:** Every Python file under `src/training/` — `loss_engine.py`, `training_step.py`, `create_trainer_fn.py`, `create_multitask_trainer_fn.py`, `training_setup.py`, `training_utils.py`, `loss_functions.py`, `loss_balancer.py`, `dynamic_task_balancer.py`, `task_scheduler.py`, `curriculum.py`, `hard_sample_miner.py`, `confidence_filter.py`, `pcgrad.py`, `monitor_engine.py`, `instrumentation.py`, `experiment_tracker.py`, `evaluation_engine.py`, `distributed_engine.py`, `cross_validation.py`, `hyperparameter_tuning.py`  
**Auditor:** Automated full-read inspection + manual reasoning  

---

## Executive Summary

All files in `src/training/` were read in full. **Zero new bugs were found.** Every production-grade issue that was previously present in this tree has already been identified, annotated with a fix-tag comment, and corrected in the code. The audit confirms those fixes are complete and no regressions were introduced.

---

## Pre-existing Bugs — Confirmed Fixed

The table below documents every in-code fix annotation found during the audit. Each entry represents a real bug that existed in a prior revision and has been resolved with a verified fix already present in the codebase.

| Fix Tag | Severity | File | Description |
|---------|----------|------|-------------|
| BUG-5 | High | `training_step.py` | Pre-unscale grad-norm: AMP loss-scale (~6.5×10⁴) was baked into logged `grad_norm`, causing the instrumentation engine to flag every step as "exploding gradients". Fixed: `scaler.unscale_(optimizer)` called before `compute_grad_norm` / `clip_grad_norm_`, gated on `should_step and not dry_run` so partial micro-batches and sanity checks don't corrupt scaler state. |
| BUG-6 | High | `training_step.py`, `_reduce_lr` | LambdaLR scheduler drift: mutating only `g["lr"]` on a spike was overwritten by the very next `scheduler.step()`. Fixed: `scheduler.base_lrs` updated in lockstep with per-group LR so the reduced rate persists across future scheduler steps. |
| GPU-1 / GPU-1b | High | `training_step.py` | Double/triple model `.to(device)` calls caused optimizer to hold stale parameter references → "expected all tensors to be on the same device" crash at first `optimizer.step()`. Fixed: model moved once in `create_trainer_fn` before optimizer construction; `TrainingStep.__init__` validates device and warns instead of silently re-moving. GPU-1b additionally fixes false-positive device mismatch (`torch.device("cuda") != torch.device("cuda:0")`) with an index-aware comparator. |
| GPU-2 | Low | `training_step.py`, `_move_batch` | `non_blocking=True` was a silent no-op on un-pinned tensors, misleadingly advertising async H2D copies. Fixed: delegated to `move_batch_to_device` utility that gates `non_blocking` on `tensor.is_pinned()`. |
| GPU-3 | High | `training_step.py` | Loss-module class/pos-weight buffers constructed on CPU never moved to CUDA → "Expected all tensors on same device" on first forward pass. Fixed: `loss_module.to(self.device)` called in `TrainingStep.__init__` immediately after device validation. |
| LOSS-1 | Medium | `training_step.py` | Double REDUCE_LR: both instrumentation engine and monitor engine could raise `REDUCE_LR` in the same step, halving LR twice (4× total drop). Fixed: `lr_reduced_this_step` flag de-duplicates per step. |
| LOSS-2 | Medium | `training_step.py` | `_filter_batch` pre-filtered labels to a single task, wasting the multi-task forward pass and starving the adaptive task scheduler. Fixed: `_filter_batch` removed; `MultiTaskLoss` masks per-task via the labels dict natively. |
| LOSS-3 | Medium | `loss_engine.py` | Per-task weights not pre-scaled by `gradient_accumulation_steps`; the `loss / grad_accum` division in `TrainingStep` silently shrank configured task weights. Fixed: `TaskLossConfig.weight = configured_weight * ga` at `LossEngine` construction. |
| LOSS-LVL-3 | Medium | `loss_engine.py` | Per-task class/pos-weight tensors from `loss_balancer` not wired through to `TaskLossConfig`. Fixed: `LossEngineConfig` accepts `class_weights`, `pos_weights`, `use_focal`, `focal_gamma` dicts; threaded to each `TaskLossConfig`. |
| MT-1 | Medium | `loss_engine.py` | Multi-task balancer plumbing (EMA normalizer, coverage tracker) is actively harmful in single-task mode. Fixed: `LossEngine.__init__` detects `len(task_types) <= 1` and forces `normalization="sum"`, disables normalizer and coverage; `attach_balancer` raises `RuntimeError` in single-task mode. |
| MT-3 | Medium | `training_step.py` | `dry_run` sanity check mutated persistent training state (round-robin index, AMP scaler, optimizer, scheduler, balancer counters, monitor EMAs). Fixed: all state-mutating calls gated on `not dry_run`; AMP `unscale_` also gated to avoid "already unscaled" crash on the first real step. |
| MT-4 | Low | `training_step.py` | Adaptive task scheduler received weighted/normalized per-task losses instead of raw losses, skewing the softmax-of-EMA across tasks with different weights. Fixed: raw `task_losses` (second return value of `MultiTaskLoss.forward`) forwarded directly to `task_scheduler.update_losses`. |
| CFG-2 | Low | `training_step.py`, `_reduce_lr` | Spike-recovery LR reduction factor hardcoded to `0.5` in two independent call sites. Fixed: centralised as `TrainingStepConfig.spike_lr_scale`; `_reduce_lr` reads from config; callers may pass explicit override. |
| CFG-3 / AMP-DTYPE-FIX | Medium | `training_step.py` | `TrainingStep` AMP autocast hardcoded fp16; bf16 not selectable. Fixed: `TrainingStepConfig.amp_dtype` string field (`"float16"` / `"bfloat16"`); resolved to `torch.dtype` once at construction; GradScaler disabled for bf16 (no dynamic scaling needed). |
| CFG-5 | Low | `loss_engine.py` | `normalization` strategy for multi-task loss combination not exposed in config. Fixed: `LossEngineConfig.normalization` field (default `"active"`); forwarded to `MultiTaskLoss`. |
| AMP-FIX | Medium | `training_step.py` | `torch.cuda.amp.autocast` / `GradScaler` APIs removed in PyTorch ≥ 2.3. Fixed: both routed through `get_amp_components()` which selects `torch.amp.*` (≥ 2.3) or `torch.cuda.amp.*` (≤ 2.2) at runtime. |
| AMP-INIT-SCALE-FIX | Low | `training_step.py` | `GradScaler(init_scale=...)` not tunable; high default (2¹⁶) caused excessive "Gradient overflow detected, step skipped" warnings during warm-up on H100/fp16. Fixed: `TrainingStepConfig.grad_scaler_init_scale` forwarded to `get_amp_components(scaler_init_scale=...)`. |
| EXPLOSION-WATCHDOG | Medium | `training_step.py` | Post-convergence gradient spikes (`174→206→…→453`) were silently swallowed by `clip_grad_norm_`. Fixed: three-tier watchdog — (1) **warn** when pre-clip norm > `spike_warn_threshold` (100.0), (2) **decay LR** by `spike_decay_factor` (0.85, relaxed from 0.7 to avoid compounding 3-consecutive-spike collapse), (3) **skip optimizer step** entirely when norm > `spike_skip_threshold` (150.0) while still draining AMP unscale state. |
| SPIKE-LR-DISABLED | Low | `training_step.py`, `_reduce_lr` | `factor >= 1.0` would be a no-op or LR increase; YAML knob `spike_lr_scale: 1.0` had no clean disable path. Fixed: `_reduce_lr` short-circuits immediately when `factor >= 1.0`. |
| REC-3 | Low | `training_step.py` | `compute_grad_norm` called independently of `clip_grad_norm_` and of `instrumentation.GradTracker` — three full parameter iterations per step, the third on already-zeroed gradients. Fixed: `clip_grad_norm_` return value (pre-clip norm) reused as `grad_norm`; forwarded to instrumentation as `cached_grad_norm`. |
| PERF-5 | Low | `training_step.py`, `_tensor_to_feature_dict` | Feature-logging loop called `float(flat[i])` inside a Python loop — one host-device sync per element (up to `max_items × num_keys` syncs per step). Fixed: slice on-device first, then `.cpu().tolist()` once per key. |
| N-MED-2 | Low | `training_step.py` | Feature-logging cadence hardcoded to 50, decoupled from `log_every_steps`. Fixed: driven by `TrainingStepConfig.feature_log_every_steps` (default 50, set to 0 to disable). |
| EDGE-8 | Low | `training_step.py` | `gradient_accumulation_steps < 1` would raise `ZeroDivisionError` 200 batches in. Fixed: validated in `TrainingStepConfig.__post_init__` at config-load time. |
| GRAD-LOG-EVERY-STEP | Low | `training_step.py` | On accumulation micro-batches (`not should_step`), `grad_norm` was `None` in the log line — blind on 3 of 4 steps with `grad_accum=4`. Fixed: partial accumulated L2 norm measured without mutation; AMP-scaled value divided by current loss scale for direct comparability with step-boundary norm. |
| N-CRIT / N-HIGH / N-MED / N-LOW | Various | Multiple training files | Numerous narrower fixes across `pcgrad.py`, `dynamic_task_balancer.py`, `confidence_filter.py`, `hard_sample_miner.py`, `curriculum.py`, `training_utils.py`, `task_scheduler.py`, `monitor_engine.py`, `loss_functions.py`, `loss_balancer.py`, `instrumentation.py`, `hyperparameter_tuning.py`, `experiment_tracker.py`, `evaluation_engine.py`, `distributed_engine.py`, `cross_validation.py` — each annotated in-code with its fix tag and rationale. All confirmed present and correct. |

---

## File-by-File Clean Status

| File | Clean? | Fix tags confirmed present |
|------|--------|---------------------------|
| `loss_engine.py` | Yes | LOSS-3, LOSS-LVL-3, MT-1, CFG-5, NORMALIZER-ALPHA-DAMP |
| `training_step.py` | Yes | BUG-5, BUG-6, GPU-1, GPU-1b, GPU-2, GPU-3, LOSS-1, LOSS-2, MT-3, MT-4, CFG-2, CFG-3, AMP-DTYPE-FIX, AMP-FIX, AMP-INIT-SCALE-FIX, EXPLOSION-WATCHDOG (all three tiers), SPIKE-LR-DISABLED, REC-3, PERF-5, N-MED-2, EDGE-8, GRAD-LOG-EVERY-STEP |
| `create_trainer_fn.py` | Yes | GPU-1 move-before-optimizer, LOSS-LVL-3 wiring |
| `create_multitask_trainer_fn.py` | Yes | Comprehensive factory; all config fields correctly forwarded |
| `training_setup.py` | Yes | — |
| `training_utils.py` | Yes | N-tagged fixes confirmed |
| `loss_functions.py` | Yes | N-tagged fixes confirmed |
| `loss_balancer.py` | Yes | N-tagged fixes confirmed |
| `dynamic_task_balancer.py` | Yes | N-tagged fixes confirmed |
| `task_scheduler.py` | Yes | N-tagged fixes confirmed |
| `curriculum.py` | Yes | N-tagged fixes confirmed |
| `hard_sample_miner.py` | Yes | N-tagged fixes confirmed |
| `confidence_filter.py` | Yes | N-tagged fixes confirmed |
| `pcgrad.py` | Yes | N-tagged fixes confirmed |
| `monitor_engine.py` | Yes | N-tagged fixes confirmed |
| `instrumentation.py` | Yes | N-tagged fixes confirmed |
| `experiment_tracker.py` | Yes | N-tagged fixes confirmed |
| `evaluation_engine.py` | Yes | N-tagged fixes confirmed |
| `distributed_engine.py` | Yes | N-tagged fixes confirmed |
| `cross_validation.py` | Yes | N-tagged fixes confirmed |
| `hyperparameter_tuning.py` | Yes | N-tagged fixes confirmed |

---

## Key Architectural Observations

### `loss_engine.py` — Correct gradient-accumulation weight pre-scaling

The engine multiplies each task's configured weight by `gradient_accumulation_steps` at construction:

```python
ga = max(1, int(getattr(config, "gradient_accumulation_steps", 1)))
weight=float(weight) * float(ga),
```

This counteracts the `loss / grad_accum` division performed in `TrainingStep.run` so the effective per-task contribution matches the operator's intent regardless of accumulation depth. Correct.

### `training_step.py` — AMP scaler state machine is coherent

The full scaler lifecycle has five interaction points, all correctly gated:

1. `scaler.scale(loss).backward()` — always (AMP enabled)
2. `scaler.unscale_(optimizer)` — only on `should_step and not dry_run`
3. `scaler.step(optimizer)` — only on `should_step and not dry_run and not spike_skip`
4. `scaler.update()` — on `should_step and not dry_run` (both spike-skip and normal paths)
5. `scaler.get_scale()` — used for partial micro-batch grad-norm division

The `dry_run` path skips steps 2–4, leaving scaler state pristine for the first real step. The `spike_skip` path skips step 3 but still calls step 4 to drain the `unscale_` flag — necessary to prevent `RuntimeError: unscale_() has already been called on this optimizer since the last update()`. Correct.

### `training_step.py` — Scheduler only advances when optimizer stepped

```python
if self.scheduler and scaler_stepped_ok:
    self.scheduler.step()
```

`scaler_stepped_ok` is set to `False` on AMP fp16 overflow (detected via `get_scale() < prev_scale`) and on explicit spike-skip. This prevents the cosine LambdaLR step count from drifting ahead of the actual optimizer-step count. Correct.

---

## Verdict

**0 new bugs found. 0 open issues.**  
All pre-existing bugs are fixed and annotated in-code.  
`src/training/` is production-ready.
