# Inference Layer Audit — `src/inference/`

**Audit scope:** v7 latency + throughput review  
**Files audited:** 21 (all files under `src/inference/`)  
**Date:** 2026-05-03  
**Status:** Complete — 3 bugs fixed, 1 informational finding documented

---

## File Inventory

| File | Role |
|------|------|
| `__init__.py` | Public re-exports |
| `constants.py` | Single-source-of-truth constants (`INFERENCE_CACHE_VERSION`, `DEFAULT_INFERENCE_BATCH_SIZE`, `DEFAULT_MAX_LENGTH`, `REPORT_VERSION`) |
| `schema.py` | Typed output dataclasses (`TaskPrediction`, `PredictionOutput`, `ReportSchema`, etc.) |
| `inference_config.py` | `InferenceConfigLoader` — YAML → `InferenceConfig` |
| `inference_engine.py` | `InferenceEngine` — HF model forward, calibration, `predict_for_evaluation` |
| `inference_pipeline.py` | `PredictionPipeline` — multi-head model wrapper |
| `model_loader.py` | `ModelLoader`, `UnifiedPredictor`, tokenizer cache |
| `prediction_service.py` | `PredictionService` — cache, logger, monitor orchestration |
| `predict_api.py` | Thin public API (`predict`, `predict_batch`, `predict_full`, `get_metrics`) |
| `postprocessing.py` | `Postprocessor` — sigmoid/softmax dispatch, per-task thresholds |
| `feature_preparer.py` | `FeaturePreparer` — flatten → vectorise → scale → select |
| `inference_cache.py` | `InferenceCache` — LRU memory + disk cache, single-flight |
| `inference_logger.py` | `InferenceLogger` — structured JSON prediction logs |
| `monitoring.py` | `InferenceMonitor` — rolling latency/confidence/entropy windows |
| `drift_detection.py` | `DriftDetector` — KL, JS, PSI, Wasserstein drift |
| `batch_inference.py` | `BatchInferenceEngine` — CSV → predictions → reports |
| `run_inference.py` | CLI entry point for offline evaluation |
| `analyze_article.py` | `ArticleAnalyzer` — full analysis orchestrator |
| `report_generator.py` | `ReportGenerator` — structured report assembly |
| `result_formatter.py` | `ResultFormatter` — api/dashboard/research output shapes |
| `single_pass_analyzer.py` | `SinglePassAnalyzer` — single-encoder-pass async fan-out |

---

## Pre-existing Fixes (already in codebase)

The inference layer carries extensive inline audit commentary. The following classes of issue were already resolved before this audit:

| Tag | Description |
|-----|-------------|
| CRIT-1 | Single `InferenceConfig` dataclass shared by engine and loader (no silent field drop) |
| CRIT-2 | Nested `{task: {...}}` output contract from `predict_for_evaluation`; `_meta` scratch key correctly skipped in all but one path (fixed below) |
| CRIT-3/8 | Task type driven by `TASK_CONFIG` YAML — no hardcoded `_BINARY_TASKS` overrides |
| CRIT-4 | `fake_probability` only emitted for legacy binary label maps |
| CRIT-5 | Model path from settings, not hardcoded `"models"` literal |
| CRIT-6 | `FeaturePipeline` runs full extraction instead of a stub dict |
| CRIT-7 | Calibration failures surface as errors; startup warning when no calibrator attached |
| DEV-1 | Weights loaded in fp32; AMP autocast handles mixed-precision compute |
| DEV-2 | `TRUTHLENS_AMP_DTYPE` env var drives bf16/fp16/fp32 choice everywhere |
| DEV-3 | `requires_grad_(False)` applied after `eval()` to eliminate autograd overhead |
| DEV-4 | Isotonic calibration avoids GPU→CPU→GPU ping-pong |
| LAT-1 | `predict_full_batch` batches N texts in one forward pass instead of N serial calls |
| LAT-2 | Multiprocessing pool removed; in-process flattener used for all batch sizes |
| LAT-3 | Cache `set` serialises exactly once (no double JSON round-trip) |
| LAT-4 | Disk reads outside the global lock (only dict mutation under lock) |
| LAT-5 | Single-flight via `InferenceCache.get_or_compute` |
| LAT-6 | `torch.compile` opt-in, not unconditional |
| LAT-7 | Warmup uses representative-length input instead of a single token |
| MEM-1 | `keep_outputs_on_device` defers CPU transfer to end of eval loop |
| MEM-2 | Disk LRU eviction + mtime touch on read |
| MEM-3 | Atomic temp-file write with `os.replace`; fsync on both compressed and plain paths |
| PP-1 | `UnifiedPredictor` dispatches sigmoid vs softmax by task type |
| PP-2 | Per-task thresholds loaded from training-produced `thresholds.json` |
| PP-3 | `task_types` required parameter in `Postprocessor.process` — no silent default |
| PP-4 | Correct entropy formula per task type (Bernoulli vs categorical) |
| CFG-2 | `INFERENCE_CACHE_VERSION` constant shared across cache and predict_api |
| CFG-5 | `DEFAULT_INFERENCE_BATCH_SIZE` / `DEFAULT_MAX_LENGTH` shared across all entry points |
| CFG-6 | `REPORT_VERSION` read from constants module |
| CFG-7 | Loader validates present fields but does not error on absent optional fields |
| REC-1 | `ArticleAnalyzer._run_prediction` surfaces real fields, not missing keys |
| REC-2 | `predict_full` in `PredictionService` uses namespaced cache key |
| REC-3 | Process-wide tokenizer cache avoids double HF vocabulary load |
| REC-4 | `ReportGenerator.generate_report` raises if `profile` passed without `aggregation` |
| MT-3 | Per-task calibrator mapping accepted alongside legacy single-calibrator slot |
| UNUSED-FIX (multiple) | `InferenceMonitor` actually wired and updated per request |

---

## Findings

### INF-FIX-001 — `batch_inference.py` L185–190: Wrong kwargs to `ReportGenerator.generate_report` *(BUG — FIXED)*

**Severity:** High  
**File:** `src/inference/batch_inference.py`  
**Function:** `BatchInferenceEngine._process_batch`

**Root cause:**  
The call to `ReportGenerator.generate_report` passed three keyword arguments that do not exist in its signature:

```python
# Before (broken)
report = self.report_generator.generate_report(
    article_text=text,
    bias_analysis={"bias": bias_val},    # ← not a parameter
    emotion_analysis={"emotion": emotion_val},  # ← not a parameter
    credibility_score=None,               # ← not a parameter
)
```

`generate_report` uses a keyword-only signature (`*,`). Python raises `TypeError: generate_report() got an unexpected keyword argument 'bias_analysis'` on **every** batch inference run, making `BatchInferenceEngine.run` completely non-functional.

**Fix applied:**

```python
# After (correct)
report = self.report_generator.generate_report(
    article_text=text,
    predictions=preds,
)
```

`preds` is the per-item dict (`bias`, `ideology`, `propaganda_probability`, `emotion`) already constructed earlier in the same loop iteration — the correct payload for the `predictions` parameter.

---

### INF-FIX-002 — `run_inference.py` L139–155: Evaluation loop crashes on `"_meta"` key *(BUG — FIXED)*

**Severity:** High  
**File:** `src/inference/run_inference.py`  
**Function:** `main` (evaluation branch)

**Root cause:**  
`InferenceEngine.predict_for_evaluation` returns a dict with shape:

```
{
    "main": {"logits": ..., "probabilities": ..., "predictions": ..., ...},
    "_meta": {"texts": [...]},
}
```

The evaluation loop in `run_inference.py` iterated all keys without a guard:

```python
for task, out in outputs.items():
    logits = out["logits"]   # KeyError: '_meta' → {"texts": [...]}
    probs  = out["probabilities"]
```

Every CLI invocation with `--evaluate` raised `KeyError: 'logits'` immediately.

**Fix applied:**

```python
for task, out in outputs.items():
    # Guard against "_meta" and any other non-task scratch keys.
    if not isinstance(out, dict) or "logits" not in out:
        continue
    logits = out["logits"]
    probs  = out["probabilities"]
```

---

### INF-FIX-003 — `single_pass_analyzer.py` L194: AMP dtype hardcoded to `float16` *(BUG — FIXED)*

**Severity:** Medium  
**File:** `src/inference/single_pass_analyzer.py`  
**Function:** `SinglePassAnalyzer._forward`

**Root cause:**  
Every other inference entry point (`InferenceEngine`, `PredictionPipeline`) reads `TRUTHLENS_AMP_DTYPE` to choose between bf16, fp16, and fp32 (DEV-2 fix applied project-wide). `SinglePassAnalyzer._forward` was overlooked and still hardcoded:

```python
# Before
torch.autocast(device_type=self.device.type, dtype=torch.float16)
```

When a model is trained under bf16 (the default), running the single-pass path under fp16 narrows the dynamic range and overflows unbalanced-head logits, silently producing wrong predictions.

**Fix applied:**  
Added a module-level `_resolve_amp_dtype()` function that reads `TRUTHLENS_AMP_DTYPE` (consistent with the implementation in `inference_engine._resolve_amp_dtype_engine`). The `_forward` method now calls it on each invocation:

```python
_amp_dtype = _resolve_amp_dtype()
amp_ctx = (
    torch.autocast(device_type=self.device.type, dtype=_amp_dtype)
    if self.config.amp
    and self.device is not None
    and self.device.type == "cuda"
    and _amp_dtype is not None
    else _nullctx()
)
```

---

### INF-INFO-001 — `inference_engine.py` L488: Schema validation in `predict_full` always warns *(INFO)*

**Severity:** Informational  
**File:** `src/inference/inference_engine.py`  
**Function:** `InferenceEngine.predict_full`

**Observation:**  
`predict_full` calls `PredictionOutput(**result)` to validate the service output against the schema dataclass. However, `PredictionOutput.__init__` requires a `tasks: Dict[str, TaskPrediction]` field, whereas `PredictionService.predict()` returns a flat dict `{label, confidence, fake_probability}`. The validation always raises `TypeError`, which is caught by the surrounding `try/except` and logged as a warning. The result is then returned unvalidated.

```python
try:
    result = PredictionOutput(**result).model_dump()  # ← always fails
except Exception as e:
    logger.warning("Schema validation failed: %s", e)  # ← always fires
```

This is a correctness concern (the schema guard provides no protection) and generates noise in production logs on every `predict_full` call.

**Recommendation:**  
Either wire the correct output shape into `predict_full` (building a `PredictionOutput` from the structured engine output before the service layer flattens it), or remove the dead schema validation block. The validation block should be placed earlier in the call chain where the full `{task: TaskPrediction}` structure is still intact.

No code change applied in this audit pass — resolving this correctly requires aligning the service layer's return contract with the schema, which is a broader interface change.

---

## Summary

| ID | Severity | File | Status |
|----|----------|------|--------|
| INF-FIX-001 | High | `batch_inference.py` | **Fixed** |
| INF-FIX-002 | High | `run_inference.py` | **Fixed** |
| INF-FIX-003 | Medium | `single_pass_analyzer.py` | **Fixed** |
| INF-INFO-001 | Info | `inference_engine.py` | Documented — deferred |

---

## Performance Assessment

The inference layer is well-optimised for production latency and throughput:

| Area | Status |
|------|--------|
| Single-flight deduplication (LAT-5) | Implemented via `InferenceCache.get_or_compute` |
| Batch tokenisation (LAT-7) | Warmup with representative-length input |
| AMP mixed-precision (DEV-2) | Env-driven dtype, bf16 default |
| Parameter freezing (DEV-3) | `requires_grad_(False)` after `eval()` |
| CPU transfer deferral (MEM-1) | `keep_outputs_on_device` flag in evaluation loop |
| Tokenizer sharing (REC-3) | Process-wide LRU cache keyed by resolved path |
| Disk cache atomicity (MEM-3) | `os.replace` + `fsync` on both compressed and plain paths |
| Batch API (LAT-1) | `predict_full_batch` uses one forward pass for N texts |
| Per-task thresholds (PP-2) | Loaded from training-produced `thresholds.json` |
| torch.compile (LAT-6) | Opt-in only; compilation disabled by default |

**Overall assessment:** The inference layer is production-grade after the three bug fixes above. The latency and throughput architecture is sound, with no remaining hot-path serialisation bottlenecks or redundant allocations. The informational schema validation issue (INF-INFO-001) should be addressed in the next interface revision cycle.
