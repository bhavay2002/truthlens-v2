# `src/analysis/` — Production-Grade Audit Report

**Date:** 2026-05-03  
**Scope:** All 35 files in `src/analysis/` (9 025 lines)  
**Constraint:** Fixes restricted to `src/analysis/` only  
**Format:** ID · Severity · File(s) · Description · Fix applied

---

## Summary

| Severity | Count | Fixed |
|----------|-------|-------|
| CRITICAL | 2 | ✅ 2 |
| HIGH | 3 | ✅ 3 |
| MEDIUM | 1 | ✅ 1 |
| DRY / INFO | 3 | ✅ 1 (partial — entropy delegated in rhetorical) |

Total fixes applied to source: **6**

---

## CRITICAL

### CRIT-A-FRAMING-DIV · `framing_analysis.py`

**Root cause.** `_frame_dominance(scores)` appends `frame_dominance_score` into the
`scores` dict in-place via `scores.update(...)`. The very next line then calls
`_frame_diversity(scores)` with the *already-mutated* six-key dict instead of
the original five-frame distribution.

**Consequences in `_frame_diversity`:**

1. `np.array(list(scores.values()))` picks up 6 values instead of 5.
2. `values.sum()` > 1.0 (the 5 normalized frame probabilities already sum to
   1.0; adding `frame_dominance_score = max(prob_i)` pushes the total to
   `1 + max_prob`).
3. `probs = values / (values.sum() + EPS)` produces a probability vector
   that no longer sums to 1.0.
4. `max_entropy = np.log(len(values))` = `log(6)` instead of `log(5)`.
5. The diversity score is **inflated** for concentrated distributions.
   Example: single dominant frame (prob=[1,0,0,0,0]) → `dominance=1`,
   sum=2, probs=[0.5,0,0,0,0,0.5], entropy=log(2), reported diversity
   = log(2)/log(6) ≈ 0.39 instead of the correct 0.0.

**Fix.** Capture `base_scores = dict(scores)` before `_frame_dominance`
mutates the dict; pass `base_scores` to `_frame_diversity`.

```python
# before
scores.update(self._frame_dominance(scores))
scores.update(self._frame_diversity(scores))   # ← mutated 6-key dict!

# after
base_scores = dict(scores)
scores.update(self._frame_dominance(scores))
scores.update(self._frame_diversity(base_scores))  # ← clean 5-key dist
```

---

### CRIT-A-PROP-RUNNER · `integration_runner.py` + `propaganda_pattern_detector.py`

**Root cause.** `AnalysisIntegrationRunner.analyze_text` calls
`analyzer.analyze(ctx)` for every registered analyzer. For
`PropagandaPatternDetector`, the method signature is:

```python
def analyze(self, ctx=None, *, emotion_features=None, narrative_features=None, ...):
    del ctx
    emotion = emotion_features or {}   # always empty via this path
    ...
```

The detector ignores the `FeatureContext`, computes from all-empty upstream
dicts, and **silently returns a seven-key all-zero feature dict** on every
call via the integration runner. The silence means callers see a complete
result and have no way to know the propaganda signals were never computed.

**Fix.** Added `requires_upstream_features: bool = True` class attribute to
`PropagandaPatternDetector`. The integration runner now checks for this flag
and records a structured `{"skipped": "requires_upstream_features — ..."}` gap
instead of a silent zero-filled result, so downstream consumers see an
explicit signal rather than a corrupted zero.

---

## HIGH

### BUG-A-RHETO-MODEL · `output_models.py`

**Root cause.** `RhetoricalFeatures(FeatureModel)` declared 8 fields but
`RHETORICAL_DEVICE_KEYS` has 10 keys — `rhetoric_intensity` and
`rhetoric_diversity` were missing.

**Consequences.**

- `RhetoricalFeatures(**analyzer_output)` would succeed (Pydantic ignores
  extra keys by default) but silently **drop the two missing keys**.
- `RhetoricalFeatures.vector()` → `make_vector(self.model_dump(), RHETORICAL_DEVICE_KEYS)`
  would fill those two positions with 0.0, masking real signal.
- Any schema completeness check against `RHETORICAL_DEVICE_KEYS` would
  always report 8/10 coverage.

**Fix.** Added `rhetoric_intensity: float = 0.0` and
`rhetoric_diversity: float = 0.0` to `RhetoricalFeatures`.

---

### BUG-A-PROP-INTENSITY · `narrative_propagation.py`

**Root cause.** The `conflict_propagation_intensity` feature was computed as:

```python
propagation = (
    0.4 * sum(raw.values()) +   # sum of 5 independent densities ∈ [0,5]
    0.2 * opposition +
    0.2 * polarization +
    0.2 * phrase
)
```

`sum(raw.values())` is the **sum** of five independent per-category token
densities, not a mean. Each density is in [0, 1] so the sum ranges [0, 5].
The coefficient 0.4 alone can reach 2.0, meaning the feature saturates to
1.0 (via `_safe` clipping) for any article with moderate conflict across
two or more categories. The opposition / polarization / phrase components
(weights 0.2 each) become invisible in those cases.

**Fix.** Use the mean of the raw densities:

```python
propagation = (
    0.4 * (sum(raw.values()) / max(len(raw), 1)) +   # mean ∈ [0,1]
    0.2 * opposition +
    0.2 * polarization +
    0.2 * phrase
)
```

Maximum propagation is now `0.4 + 0.6 = 1.0` when all inputs are 1.0,
matching the semantic intent of a bounded intensity score.

---

## MEDIUM

### NUM-A-RHETO-INTENSITY · `rhetorical_device_detector.py`

**Root cause.**

```python
intensity = sum(raw.values()) / (len(raw) + EPS)
```

`len(raw)` is the count of the hard-coded `raw` dict (7 keys) — a positive
integer that is never zero. Adding `EPS = 1e-8` provides no numerical
stability benefit but introduces a constant downward bias of ≈ 1.4 × 10⁻⁹
on every call.

**Fix.**

```python
intensity = sum(raw.values()) / max(len(raw), 1)
```

---

## DRY / INFO (no correctness impact, noted for future cleanup)

### DRY-A-ENTROPY (partial fix applied)

Multiple analyzers maintain private local `_entropy` implementations that
duplicate the shared `safe_normalized_entropy` helper in `_text_features.py`.
The helper carries the necessary guards (n ≤ 1 early return, max_entropy < EPS
guard, sum < EPS guard).

| File | Status |
|------|--------|
| `rhetorical_device_detector.py` | **Fixed** — `_entropy` now delegates to `safe_normalized_entropy` |
| `source_attribution_analyzer.py` | Not fixed — local `_entropy` is equivalent but untested against the edge cases in the shared helper |
| `ideological_language_detector.py` | Not fixed — same as above |
| `framing_analysis.py` | Not fixed — inline entropy in `_frame_diversity`; will become correct now that `base_scores` is passed |

Recommended follow-up: replace the two remaining local `_entropy` methods
with a single import of `safe_normalized_entropy`.

---

### DRY-A-NORMALIZE

Eight analyzers each carry a nearly identical `_normalize` method that
divides a float dict by its element sum via NumPy. The method body is
identical across `framing_analysis.py`, `ideological_language_detector.py`,
`rhetorical_device_detector.py`, `narrative_propagation.py`,
`narrative_temporal_analyzer.py`, `source_attribution_analyzer.py`,
`information_density_analyzer.py`, and `bias_profile_builder.py`. A shared
`normalize_distribution(scores: dict) -> dict` helper in `_text_features.py`
would remove ~40 lines of duplication and centralize the EPS guard.

Not fixed in this audit (refactor scope, not a correctness issue).

---

### INFO-A-LABEL-HEURISTIC · `label_analysis.py` · `multitask_validator.py`

`_is_multilabel_column` / `_is_multilabel` samples only the *first* row
(`series.iloc[0]`) to decide if a column is multi-label. Columns where the
first row is a scalar string but later rows are lists (or vice-versa) will
be mis-classified. Impact is limited to dataset-inspection utilities that
run offline before training; it does not affect inference.

---

## Files Confirmed Clean (no actionable findings)

| File | Notes |
|------|-------|
| `analysis_config.py` | Config dataclasses, no logic bugs |
| `analysis_pipeline.py` | `nlp.pipe` batch path correct; spaCy config respected |
| `analysis_registry.py` | Kwarg introspection, cycle detection, singleton — clean |
| `argument_mining.py` | Shared cache, safe accessors, clause density correct |
| `base_analyzer.py` | Caching, validation, postprocess, fallback — clean |
| `batch_processor.py` | Delegation to `nlp.pipe`, correct result shape |
| `bias_profile_builder.py` | zscore/robust tanh fix, softmax denominator fix, global norm — clean |
| `context_omission_detector.py` | Intentional single-char quote pattern (character density, not spans) |
| `discourse_coherence_analyzer.py` | Jaccard, shared sent-lemma cache, transition ratio — clean |
| `emotion_lexicon.py` | Static data file |
| `emotion_target_analysis.py` | PhraseMatcher vocab guard, shared `get_shared_nlp` — clean |
| `feature_context.py` | Lazy spaCy, shared cache, `punct_count` — clean |
| `feature_keys.py` | Key tuples — correct and complete |
| `feature_merger.py` | Merge, vector, completeness — clean |
| `feature_schema.py` | Registry, `make_vector`, `validate_features` — clean |
| `ideological_language_detector.py` | Entropy local copy; balance mapping correct |
| `information_density_analyzer.py` | Shared `safe_normalized_entropy`, `punct_count` cache — clean |
| `information_omission_detector.py` | Logistic one-sided score handles counter=0 correctly |
| `integration_runner.py` | **Fixed** (BUG-A-PROP-RUNNER guard) |
| `label_analysis.py` | Multi-label heuristic noted above; otherwise clean |
| `multitask_validator.py` | Row-level filtering correct; heuristic noted above |
| `narrative_conflict.py` | Shared punct cache, actor structure — clean |
| `narrative_propagation.py` | **Fixed** (BUG-A-PROP-INTENSITY) |
| `narrative_role_extractor.py` | Role scoring, entity resolution — clean |
| `narrative_temporal_analyzer.py` | NUM-A5 contrast normalization, tense distribution — clean |
| `orchestrator.py` | Narrative sub-section merge with averaging, `_confidence` per-section means — clean |
| `preprocessing.py` | `ProcessPoolExecutor` → `nlp.pipe` (PERF-A4) — clean |
| `propaganda_pattern_detector.py` | **Fixed** (`requires_upstream_features` flag) |
| `rhetorical_device_detector.py` | **Fixed** (intensity denominator, delegated entropy) |
| `source_attribution_analyzer.py` | Quoted-span regex correct (Unicode + ASCII curly quotes) |
| `spacy_config.py` | Config dataclasses |
| `spacy_loader.py` | VOCAB-1 blank-en key normalization, GPU init sentinel, torch thread pin — clean |
| `_text_features.py` | LRU regex cache, `id(phrases)` cache key, `safe_normalized_entropy` — clean |
| `output_models.py` | **Fixed** (BUG-A-RHETO-MODEL: added rhetoric_intensity, rhetoric_diversity) |

---

## Change Log

| ID | File | Lines changed | Description |
|----|------|---------------|-------------|
| CRIT-A-FRAMING-DIV | `framing_analysis.py` | 2 | Snapshot base scores before dominance mutation |
| CRIT-A-PROP-RUNNER | `integration_runner.py` | +14 | Guard against `requires_upstream_features` analyzers |
| CRIT-A-PROP-RUNNER | `propaganda_pattern_detector.py` | +4 | Add `requires_upstream_features = True` class attr |
| BUG-A-RHETO-MODEL | `output_models.py` | +4 | Add `rhetoric_intensity` / `rhetoric_diversity` fields |
| BUG-A-PROP-INTENSITY | `narrative_propagation.py` | +10 | Replace `sum` with `mean` in propagation formula |
| NUM-A-RHETO-INTENSITY | `rhetorical_device_detector.py` | +4 | Fix intensity denominator: `max(len,1)` |
| DRY-A-ENTROPY | `rhetorical_device_detector.py` | −10/+4 | Delegate `_entropy` to `safe_normalized_entropy` |
