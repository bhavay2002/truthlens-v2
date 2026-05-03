# Explainability Audit Report
**Scope:** `src/explainability/` — 22 files  
**Date:** 2026-05-03  
**Prompts applied:** v9 faithfulness + performance  
**Status:** All findings fixed ✓

---

## Summary

| ID | File | Severity | Category | Status |
|----|------|----------|----------|--------|
| PERF-01 | `model_explainer.py` | High | Performance | Fixed |
| BUG-01 | `model_explainer.py` | Critical | Logic / Silent data loss | Fixed |
| BUG-02 | `model_explainer.py` | Critical | Logic / Silent data loss | Fixed |

**Total findings: 3 — all fixed.**  
Remaining 21 files: no actionable findings.

---

## Finding Detail

### PERF-01 — Fresh orchestrator instantiated on every call (HIGH)

**File:** `src/explainability/model_explainer.py` lines 46, 120  
**Category:** Performance

**Description:**  
Both `explain_prediction_full` and `explain_fast` called `ExplainabilityOrchestrator(config=config)` on every invocation. `ExplainabilityOrchestrator.__init__` constructs a `GraphExplainer`, allocates an explanation cache, and wires up several sub-components — none of which need to be rebuilt per request. Under load this causes unbounded allocations and defeats the cache that the config explicitly enables (`cache_enabled=True`).

The canonical pattern already established in `explainability_pipeline.py` is to call `get_default_orchestrator(config)`, which returns a module-level singleton and reuses the cache across requests.

**Before:**
```python
orchestrator = ExplainabilityOrchestrator(config=config)   # line 46
# ...
orchestrator = ExplainabilityOrchestrator(config=config)   # line 120
```

**After:**
```python
orchestrator = get_default_orchestrator(config)   # line 46
# ...
orchestrator = get_default_orchestrator(config)   # line 120
```

**Import change:** Replaced `ExplainabilityOrchestrator` with `get_default_orchestrator` in the import from `src.explainability.orchestrator`.

---

### BUG-01 — Wrong result key `"metrics"` silently returns `None` (CRITICAL)

**File:** `src/explainability/model_explainer.py` line 77  
**Category:** Logic bug / silent data loss

**Description:**  
`explain_prediction_full` extracted `result.get("metrics")` from the orchestrator result dict. The orchestrator (`ExplainabilityOrchestrator.explain`) stores this value under the key `"explanation_metrics"`. Because `dict.get` with a missing key returns `None`, the `"metrics"` field in every response was silently `None`, discarding the faithfulness metrics entirely with no error or warning logged.

**Before:**
```python
"metrics": result.get("metrics"),
```

**After:**
```python
"metrics": result.get("explanation_metrics"),
```

---

### BUG-02 — Wrong result key `"consistency"` silently returns `None` (CRITICAL)

**File:** `src/explainability/model_explainer.py` line 78  
**Category:** Logic bug / silent data loss

**Description:**  
Same pattern as BUG-01. The orchestrator stores consistency metrics under `"consistency_metrics"`, but the shim requested `result.get("consistency")`. Every response had `"consistency": None`.

**Before:**
```python
"consistency": result.get("consistency"),
```

**After:**
```python
"consistency": result.get("consistency_metrics"),
```

---

## Files Reviewed (no findings)

`attention_explainer.py`, `bias_emotion_explainer.py`, `consistency_checker.py`,
`explanation_aggregator.py`, `explainability_pipeline.py`, `explainability_router.py`,
`graph_explainer.py`, `lime_explainer.py`, `ner_masker.py`, `orchestrator.py`,
`propaganda_explainer.py`, `rollout_attention.py`, `shap_explainer.py`,
`token_attributions.py`, `explanation_cache.py`, `explanation_metrics.py`,
`explanation_types.py`, `faithfulness_scorer.py`, `feature_importance.py`,
`highlight_generator.py`, `text_perturbation.py`

---

## Notes

- `explainability_pipeline.py` already implements correct `explain_prediction_full` and
  `explain_fast` functions using `get_default_orchestrator`. The `model_explainer.py` shim
  is a legacy compatibility layer; BUG-01 and BUG-02 meant it silently returned different
  (lesser) data than the pipeline would.
- All fixes stay within `src/`.
