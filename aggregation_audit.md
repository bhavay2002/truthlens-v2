# Aggregation Layer Audit — TruthLens AI

**Audit scope:** `src/aggregation/` (16 files)  
**Auditor:** automated v8 audit pass  
**Date:** 2026-05-03  
**Status:** all findings fixed — 12/12 aggregation tests passing  

---

## Files audited

| File | Lines |
|---|---|
| `aggregation_config.py` | config dataclasses & group definitions |
| `aggregation_pipeline.py` | main orchestration |
| `aggregation_validator.py` | output range + consistency checks |
| `aggregation_metrics.py` | batch statistics & monitoring |
| `calibration.py` | temperature / sigmoid / isotonic calibrators |
| `feature_builder.py` | AggregatorFeatureBuilder |
| `feature_mapper.py` | section profile construction |
| `hybrid_scorer.py` | neural × rule blending |
| `neural_aggregator.py` | MLP + Attention aggregator architectures |
| `risk_assessment.py` | risk level classification |
| `score_explainer.py` | integrated-gradients + profile explanations |
| `score_normalizer.py` | per-section normaliser |
| `score_schema.py` | Pydantic output models |
| `truthlens_score_calculator.py` | section → composite score calculator |
| `weight_manager.py` | adaptive weight computation |
| `aggregator_trainer.py` | NeuralAggregator training loop |

---

## Critical bugs (FIXED)

### CRIT-AG-SCALE — `_aggregate` averaged raw non-unit values

**File:** `truthlens_score_calculator.py` → `TruthLensScoreCalculator._aggregate`  
**Severity:** Critical  

**Problem:** The method collected every numeric value from a section dict and averaged them without bounds checking per element. Upstream feature extractors emit mixed-scale signals — e.g. `{"probability": 0.82, "intensity": 150.0}` — so the mean was `75.4`, which the final clamp truncated silently to `1.0`. Every section containing a single raw-count feature was pinned at maximum confidence, nullifying ranking differentiation.

**Fix:** Clamp each value to `[0, 1]` before inclusion in the running sum so the average is computed in the probability space regardless of upstream feature scale:

```python
vals = [
    min(1.0, max(0.0, float(v)))
    for v in section_data.values()
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
]
```

---

### CRIT-AG-BLEND — graph correction applied to every section

**File:** `truthlens_score_calculator.py` → `TruthLensScoreCalculator.compute_scores`  
**Severity:** Critical  

**Problem:** The line

```python
val = base_val + self.graph_influence_cap * graph_signal
```

was executed unconditionally for **every** section including `bias`, `emotion`, `narrative`, `discourse`, `ideology`, and `analysis`. The graph signal (centrality, density, consistency) is meaningful only within the `graph` credibility section; adding it to unrelated sections inflated their scores and distorted the manipulation/credibility composites.

**Fix:** Gate the correction on `section == "graph"`:

```python
if section == "graph":
    val = base_val + self.graph_influence_cap * graph_signal
else:
    val = base_val
```

---

### NORM-AG-EXP — `explain_profile` emitted unnormalised section sums

**File:** `score_explainer.py` → `ScoreExplainer.explain_profile`  
**Severity:** Critical (score quality)  

**Problem:** The profile explanation pass accumulated raw feature values per section (`section_scores[section] += val`). A section with five features each at `0.9` produced a section score of `4.5`. Downstream in `compute_scores` the blend clipped that to `1.0`, so every feature-rich section was indistinguishable from a section with one maximally-confident feature. The explanation signal carried no gradient information across sections.

**Fix:** Clamp each feature value to `[0, 1]` during accumulation, track the per-section feature count, and divide after the loop to yield a per-section **mean** in `[0, 1]`:

```python
clamped = min(1.0, max(0.0, val))
section_scores[section] = section_scores.get(section, 0.0) + clamped
section_counts[section] = section_counts.get(section, 0) + 1
# after loop:
section_scores[section] = section_scores[section] / (section_counts[section] or 1)
```

---

## Performance bugs (FIXED)

### PERF-AG-TRAINER — double forward pass in `evaluate`

**File:** `aggregator_trainer.py` → `AggregatorTrainer.evaluate`  
**Severity:** Performance  

**Problem:** `evaluate` called `self.compute_loss(x, y, rule)` (which runs `self.aggregator(x)` internally) and then immediately called `self.aggregator(x)` **again** to collect `credibility_score` for AUC computation. Every validation batch performed two full forward passes — 2× GPU/CPU compute wasted per batch.

**Fix:** Extend `compute_loss` return dict with `"pred"` (the already-computed tensor) and read it in `evaluate`:

```python
# compute_loss now returns:
return {"total": total, "bce": L_bce, "mse": L_mse, "cal": L_cal, "pred": pred}

# evaluate uses:
all_preds.extend(losses["pred"].cpu().tolist())   # no second forward pass
```

---

### PERF-AG-BATCH — `score_batch` was a Python loop mislabeled "vectorised"

**File:** `hybrid_scorer.py` → `HybridScorer.score_batch`  
**Severity:** Performance  

**Problem:** The method was documented as a "vectorised wrapper" but iterated every sample in Python. For a batch of N articles this invoked N Python function calls and N dict allocations instead of a single NumPy operation.

**Fix:** Fast path for the common case — all neural scores present, static alpha — that computes the blend in a single NumPy array operation. The Python-loop fallback is retained for dynamic-alpha and neural-disabled cases.

---

## Edge-case bugs (FIXED)

### CAL-AG-1D — `IsotonicCalibrator.transform` 1D output not clipped

**File:** `calibration.py` → `IsotonicCalibrator.transform`  
**Severity:** Edge case  

**Problem:** The 2D branch normalised its output by row sum. The 1D branch returned raw isotonic regression output with no explicit bounding. While isotonic regression trained on binary labels is theoretically in `[0, 1]`, floating-point accumulation in sklearn's PAVA implementation can produce values marginally outside the interval.

**Fix:**
```python
calibrated = np.clip(self.models[0].transform(arr), 0.0, 1.0)
```

---

### BATCH-METRIC — `compute_batch_metrics` keys sourced from first sample only

**File:** `aggregation_metrics.py` → `compute_batch_metrics`  
**Severity:** Edge case (silent data loss)  

**Problem:** `keys = batch_scores[0].keys()` — if any subsequent sample contained a metric key not in the first sample, that key was silently dropped. The monitoring history could be systematically incomplete for heterogeneous batches.

**Fix:** Build the union of keys across all samples; default missing entries to `0.0`:

```python
keys: set = set()
for sample in batch_scores:
    keys.update(sample.keys())
aggregated[k].append(sample.get(k, 0.0))
```

---

## Pre-existing test contract gaps (FIXED)

These were failing before this audit pass due to API drift between the implementation and test expectations.

### TST-AG-SCHEMA — `TruthLensScoreModel` missing flat `truthlens_*` fields

**Files:** `score_schema.py`  
**Test:** `tests/aggregation/test_score_schema.py`  

**Problem:** Tests constructed `TruthLensScoreModel` with flat `truthlens_bias_score`, `truthlens_emotion_score`, … fields. Model had `extra="forbid"` and no such fields → `ValidationError` on every construction.

**Fix:** Added nine optional `truthlens_*` float fields to `TruthLensScoreModel` with `[0, 1]` range validators. Made `tasks`, `manipulation_risk`, `credibility_score`, `final_score` default to `{}` / `0.0` so the model can be constructed from either the flat or the structured interface.

---

### TST-AG-VECTOR — `truthlens_score_vector` function missing

**Files:** `truthlens_score_calculator.py`  
**Test:** `tests/aggregation/test_truthlens_score_calculator.py`  

**Problem:** Test imported `truthlens_score_vector` which did not exist in the module → `ImportError` on collection.

**Fix:** Added the function. It accepts a flat `Dict[str, float]` with `truthlens_*` keys and returns a stable-order `np.float32` array of shape `(9,)`. Raises `RuntimeError` on missing key.

---

### TST-AG-WGT — `credibility_bias_penalty` excluded from credibility group sum

**Files:** `aggregation_config.py`, `weight_manager.py`  
**Test:** `tests/aggregation/test_weight_manager.py::test_grouped_normalization_sums_each_group_to_one`  

**Problem:** Test expected `discourse + graph + credibility_bias_penalty + analysis_influence_credibility == 1.0` after normalisation. The penalty key was classified as a `SCALAR_WEIGHT_KEY` (excluded from group normalisation), so the sum was ~`0.83` instead of `1.0`.

**Fix:** Moved `credibility_bias_penalty` from `SCALAR_WEIGHT_KEYS` into `WEIGHT_GROUPS["credibility"]`. The value remains in `[0, 1]` after normalisation (each group member ≤ 1 when group sum = 1), so the calculator's `np.clip` on the penalty term is a no-op. `SCALAR_WEIGHT_KEYS` is now an empty tuple, kept for import back-compat.

---

### TST-AG-ADJ — `adjust_weight` silently accepted unknown keys

**Files:** `weight_manager.py`  
**Test:** `tests/aggregation/test_weight_manager.py::test_adjust_weight_rejects_unknown_key`  

**Problem:** `adjust_weight("not_a_valid_key", 0.1)` added the unknown key to `self.weights`, bypassing group-normalisation invariants. Test expected `KeyError`.

**Fix:** Added a guard before the assignment:

```python
if key not in self.weights:
    raise KeyError(key)
```

---

### TST-AG-PIPELINE — `_inject_analysis_sections` and `normalize_profile` missing

**Files:** `aggregation_pipeline.py`  
**Tests:** `tests/aggregation/test_aggregation_pipeline.py`  

**Problem:** Two tests called methods that did not exist on `AggregationPipeline` → `AttributeError`.

**Fix:** Added both as static methods:
- `_inject_analysis_sections(profile, analysis_modules)` — returns a shallow-copied profile with new sections merged in without mutating the input dict.
- `normalize_profile(profile)` — returns a copy of the profile with numeric values cast to `float`, leaving booleans and other non-numerics verbatim.

---

## Verified correct (no change needed)

| ID | Item | Finding |
|---|---|---|
| REC-AG-1 | `extract_task_signals` caching | Runs once, TaskSignal reused; confidence/entropy not recomputed. ✓ |
| CRIT-AG-3 | Missing section handling | `compute_scores` logs missing sections at DEBUG, does not raise. ✓ |
| CRIT-AG-4 | Graph key lookup | `_GRAPH_SIGNAL_KEYS` covers all keys emitted by `graph_analysis.py`. ✓ |
| CRIT-AG-6 | Calibration at logit boundary | `FeatureMapper` applies sigmoid before passing to calibrator. ✓ |
| CRIT-AG-7 | Confidence/entropy double-application | Modulation happens once in `WeightManager`; calculator accepts but ignores the kwargs. ✓ |
| CRIT-AG-UNUSED | Validator key access | Pipeline passes `{"scores": flat_scores}` (plain dict of floats) — validator reads it correctly. ✓ |
| WGT-AG-2 | `graph_influence_cap` / `explanation_blend` | Constructor-injected from `AggregationConfig.fusion`. ✓ |
| WGT-AG-3 | Weight source of truth | `WeightManager.DEFAULT_WEIGHTS` is the single source; `ScoreWeights` dataclass removed. ✓ |
| WGT-AG-4 | Symmetric scale clip | `(0.5, 2.0)` range, equal headroom for boost and attenuation. ✓ |
| NORM-AG-1 | Per-section max-norm | `normalize=False` in pipeline; normaliser passthrough only. ✓ |
| CFG-AG-1 | `model_version` config-driven | Pulled from `config.model_version` everywhere. ✓ |
| TOK-AG-1 | Keyword exact-match | `word in keys` (set membership) post-detok, not substring containment. ✓ |
| TOK-AG-2 | `_detok` surface form | Uses `convert_tokens_to_string` with manual `##`/`Ġ`/`▁` fallback. ✓ |
| TOK-AG-3 | Subword grouping | `_merge_subwords` groups pieces before section attribution. ✓ |
| TOK-AG-4 | Padding masking | Importance multiplied by attention mask before normalisation. ✓ |
| GPU-AG-2 | Gradient accumulation | `scaled.grad.zero_()` called before backward when gradient present. ✓ |
| PERF-AG-2 | Alpha-linspace broadcast | Single fused multiply, not 32 concat ops. ✓ |
| NeuralAggregator | MLP + Attention architectures | Correct sigmoid output, proper residual path. ✓ |
| AggregatorTrainer loss | BCE + λ1·MSE + λ2·soft-ECE | Mathematically correct, gradient flows through all terms. ✓ |
| WeightManager groups | Group normalisation | Each group normalised to 1.0 by `_normalize_group` after confidence/entropy scaling. ✓ |
| HybridScorer alpha | Dynamic alpha formula | Convex combination of `base_alpha` and `min_alpha`, clamped to `[min_alpha, max_alpha]`. ✓ |
| run_batch | Thread-pool parallelism | `batch_max_workers` respected; stateless pipeline safe for threads. ✓ |
| AggregationValidator | validate key access | Pipeline passes `{"scores": flat_scores}` — validator iterates the plain float dict correctly. ✓ |

---

## Summary

| Category | Count | Status |
|---|---|---|
| Critical bugs | 3 | Fixed |
| Performance bugs | 2 | Fixed |
| Edge-case bugs | 2 | Fixed |
| Pre-existing test gaps | 5 | Fixed |
| Verified correct | 23 | No change |

**Total fixes: 12 across 9 files.**  
All changes confined to `src/aggregation/` per project constraint.  
Test result: **12/12 aggregation tests passing**.
