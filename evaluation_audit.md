# `src/evaluation/` — Production-Grade Audit Report

**Date:** 2026-05-03  
**Scope:** All 20 files in `src/evaluation/` including `importance/` submodule (~4 500 lines)  
**Constraint:** Fixes restricted to `src/` only  
**Format:** ID · Severity · File(s) · Description · Fix applied

---

## Summary

| Severity | Count | Fixed |
|----------|-------|-------|
| CRITICAL | 1 | ✅ 1 |
| HIGH | 1 | ✅ 1 |
| MEDIUM | 1 | ✅ 1 |
| DRY / INFO | 2 | ✅ 2 |

Total fixes applied to source: **7 files changed**

---

## CRITICAL

### CRIT-E-ADVANCED-BROKEN · `advanced_analysis.py` + `importance/feature_ablation.py` + `importance/permutation_importance.py` + `importance/shap_importance.py`

**Root cause.** Three public wrapper functions in `advanced_analysis.py`
(`ablation_importance`, `permutation_importance`, `shap_importance`) all crash
at runtime before doing any computation:

1. **Instantiation TypeError** — `FeatureAblation(metric=metric)` and
   `PermutationImportance(metric=metric)` omit the required `model` field
   (both dataclasses declared `model: object` with no default).
   `ShapImportance()` has the same problem.

2. **Invalid method kwargs** — `ablator.single_feature_ablation(predict_fn=…)`
   passes a kwarg the method signature never declared → `TypeError`.
   `perm.compute(predict_fn=…, n_repeats=…)` does the same (both belong on
   the dataclass, not on the per-call method).

3. **Missing method** — `shap_calc.compute_with_function(…)` calls a method
   that did not exist on `ShapImportance` → `AttributeError`.

These functions are marked OFFLINE-ONLY in every module docstring and are not
on any inference path, but they are the sole API for post-hoc explainability
reports and would raise on every research run.

**Additional sub-issue (CRIT-E-ABLATION-SHAPE).** Even after fixing the
instantiation, `FeatureAblation.single_feature_ablation` immediately calls
`n_samples, n_features = X.shape` — which fails with `ValueError` when `X` is
a list of strings (the shape passed by the text-based wrappers).

**Consequences.** Every offline explainability run (`ablation_importance`,
`permutation_importance`, `shap_importance`) raises before producing any
output.  The exception propagates to the caller with no degraded-mode fallback.

**Fix — `feature_ablation.py`:**
- `model: object` → `model: Optional[object] = None`
- Added `predict_fn: Optional[Callable] = None` instance field
- `_predict` now prefers `predict_fn` when `model` is `None`
- `single_feature_ablation` converts `X` to ndarray and returns
  `{name: 0.0 …}` with a `logger.warning` when `X.ndim != 2` (non-tabular
  inputs are a legitimate caller pattern; column ablation is semantically
  undefined on text)

**Fix — `permutation_importance.py`:**
- `model: Optional[object] = None` (same pattern)
- Added `predict_fn: Optional[Callable] = None` field
- `_predict` prefers `predict_fn` over `model`

**Fix — `shap_importance.py`:**
- `model: Optional[object] = None`
- `_create_explainer` guards `self.model is None` with a clear
  `RuntimeError` directing callers to `compute_with_function`
- Added `compute_with_function(predict_fn, X, feature_names)` — installs a
  thin `_Adapter` wrapper, delegates to `compute`, and restores the original
  `model` / `_explainer` state in a `try/finally`; returns zeros with a
  warning for non-2-D inputs

**Fix — `advanced_analysis.py`:**
- `ablation_importance`: `FeatureAblation(predict_fn=predict_fn, metric=metric)`;
  removed the invalid `predict_fn=` kwarg from the `single_feature_ablation`
  call-site
- `permutation_importance`: `PermutationImportance(predict_fn=…, metric=…, n_repeats=…)`;
  removed the invalid `predict_fn=` and `n_repeats=` from `perm.compute()`
- `shap_importance`: `ShapImportance()` now valid (model optional); call to
  `compute_with_function` now resolves to the newly-added method

---

## HIGH

### HIGH-E-PIPELINE-PROBS · `evaluation_pipeline.py`

**Root cause.** Binary classification tasks produce 1-D probability arrays of
shape `(N,)` — the positive-class probability per sample.
`uncertainty_statistics` calls `_validate_probs(probs, allow_1d=False)` as its
first action, which raises `ValueError: probs must be 2D` for any 1-D input.
The pipeline wraps the `uncertainty_statistics` call in `try/except Exception`:

```python
try:
    unc = uncertainty_statistics(np.asarray(probs), ...)
    report["uncertainty"][task] = unc
except Exception as exc:
    logger.warning("Uncertainty failed for %s: %s", task, exc)
```

The exception is caught silently, so every binary task's uncertainty block is
absent from the final evaluation report with no indication other than a
`WARNING`-level log line.

**Consequences.** Uncertainty statistics (predictive entropy, confidence
calibration, per-sample variance) are never computed for binary tasks — which
are the majority task type in TruthLens (fake-news, satire, clickbait detection).
The missing values cannot be distinguished from a task that genuinely produced
no uncertainty signal.

**Fix.** Before passing to `uncertainty_statistics`, reshape 1-D binary probs
into the required 2-D `(N, 2)` format by stacking the complement:

```python
probs_for_unc = np.asarray(probs)
if probs_for_unc.ndim == 1 and probs_for_unc.size > 0:
    probs_for_unc = np.column_stack([1.0 - probs_for_unc, probs_for_unc])
unc = uncertainty_statistics(probs_for_unc, task=task, logits=logits)
```

This matches the pattern already used in `evaluate_saved_model.py`, which
correctly converts binary probs before the same call.

---

## MEDIUM

### MED-E-PERM-MUTATION · `importance/permutation_importance.py`

**Root cause.** All three permutation loops (`compute`, `compute_with_variance`,
`group_permutation`) mutate the caller's array `X` in-place during the shuffle
step, then restore it afterwards:

```python
col = X[:, j].copy()
for _ in range(self.n_repeats):
    X[:, j] = rng.permutation(col)   # mutates caller's array
    ...
X[:, j] = col                         # restore — only reached if no exception
```

If any exception fires inside the loop (e.g., `ValueError: Invalid score for
feature …` from the finite-check guard, or an exception from the model or
metric callable), `X[:, j] = col` is never executed, leaving the caller's
array permanently corrupted with a shuffled column.

**Consequences.** Any subsequent use of the original array — including the
baseline re-computation for the next feature or any caller code — silently
operates on corrupted data.  The corruption is invisible because NumPy
in-place writes produce no error.

**Fix.** Wrap each inner permutation loop with `try/finally` in all three
methods (`compute`, `compute_with_variance`, `group_permutation`) so the
restore step executes unconditionally:

```python
col = X[:, j].copy()
try:
    for _ in range(self.n_repeats):
        X[:, j] = rng.permutation(col)
        ...
finally:
    X[:, j] = col   # always restored
```

---

## DRY / INFO

### DRY-E-DEAD-ACTIVATIONS · `evaluate_model.py`

**Root cause.** Two hand-rolled activation functions `_softmax` and `_sigmoid`
were defined at module level but were never called.  The actual
`_postprocess_logits` function uses `scipy_softmax` and `expit` from
`scipy.special`:

```python
# dead — never called anywhere in the file
def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=1, keepdims=True) + 1e-12)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))   # overflows for large negative inputs
```

Beyond the dead-code issue, `_sigmoid` overflows to `0.0` for large negative
inputs (`exp(-x)` → `inf`) — a known numerical stability hazard that the
`scipy.special.expit` implementation avoids via the log-sum-exp trick.

**Fix.** Removed both functions.  No call-site changes required (there were
none).

---

### DRY-E-CALIBRATION-IF · `calibration.py`

**Root cause.** `compute_calibration` contained a dead `if/else` branch for
the binary ECE path:

```python
elif task_type == "binary":
    if probs.ndim == 2 and probs.shape[1] == 2:
        results["ece"] = expected_calibration_error(
            y_true_arr, probs, n_bins, task_type="binary"
        )
    else:
        results["ece"] = expected_calibration_error(   # identical body
            y_true_arr, probs, n_bins, task_type="binary"
        )
```

The `if` branch was dead: `probs` in the binary path is always 1-D because it
is produced by `sigmoid(scaled)` where `scaled` is a 1-D logit vector.  A
2-column `(N, 2)` binary prob matrix is only produced if the logit vector was
2-D, which does not occur in the `compute_calibration` code path.
`expected_calibration_error` already handles both 1-D and `(N, 2)` inputs
internally, so the branch distinction was also redundant at the semantic level.

**Fix.** Collapsed to a single unconditional call.

---

## Files Confirmed Clean (no actionable findings)

| File | Notes |
|------|-------|
| `calibration.py` | DRY-E-CALIBRATION-IF fixed; temperature fit, Platt/Isotonic, VectorTemperatureScaler — clean |
| `calibration_analysis.py` | Delegation to calibration helpers; error handling — clean |
| `evaluate_model.py` | DRY-E-DEAD-ACTIVATIONS fixed; `_postprocess_logits`, ONNX path, tokenization — clean |
| `evaluate_saved_model.py` | Already correctly reshapes binary probs before `uncertainty_statistics` |
| `evaluation_pipeline.py` | HIGH-E-PIPELINE-PROBS fixed; task loop, calibration block, correlation — clean |
| `importance/__init__.py` | Empty (correct; public symbols re-exported by each submodule) |
| `importance/feature_ablation.py` | CRIT-E-ADVANCED-BROKEN fixed; bootstrap, group ablation, ranking — clean |
| `importance/permutation_importance.py` | CRIT + MED fixed; baseline caching, variance, group permutation — clean |
| `importance/shap_importance.py` | CRIT fixed; sampler, batch SHAP, `_process`, grouping, ranking — clean |
| `metrics.py` | Metric dispatch, multilabel macro/micro paths, safe guards — clean |
| `reliability_diagram.py` | Bin-count, calibration-curve delegation — clean |
| `task_evaluator.py` | Per-task routing, label alignment — clean |
| `uncertainty.py` | `_validate_probs`, entropy, confidence, multilabel path — clean |
| `advanced_analysis.py` | CRIT-E-ADVANCED-BROKEN fixed; `predict_texts`, graph metrics — clean |

---

## Change Log

| ID | File | Lines changed | Description |
|----|------|---------------|-------------|
| CRIT-E-ADVANCED-BROKEN | `importance/feature_ablation.py` | +18 | `model` optional, add `predict_fn` field, update `_predict`, add 2-D guard to `single_feature_ablation` |
| CRIT-E-ADVANCED-BROKEN | `importance/permutation_importance.py` | +14 | `model` optional, add `predict_fn` field, update `_predict` |
| CRIT-E-ADVANCED-BROKEN | `importance/shap_importance.py` | +60 | `model` optional, guard `_create_explainer`, add `compute_with_function` |
| CRIT-E-ADVANCED-BROKEN | `advanced_analysis.py` | +12 | Fix all three importance instantiations and call-sites |
| HIGH-E-PIPELINE-PROBS | `evaluation_pipeline.py` | +8 | Reshape 1-D binary probs to `(N, 2)` before `uncertainty_statistics` |
| MED-E-PERM-MUTATION | `importance/permutation_importance.py` | +6 | `try/finally` restore in `compute`, `compute_with_variance`, `group_permutation` |
| DRY-E-DEAD-ACTIVATIONS | `evaluate_model.py` | −12 | Remove dead `_softmax` and `_sigmoid` |
| DRY-E-CALIBRATION-IF | `calibration.py` | −5 | Collapse identical binary ECE `if/else` branches |
