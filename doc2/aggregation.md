# `src/aggregation` — Technical Documentation

> **Target audience:** ML engineers and data engineers who need to understand, operate, extend, or debug the TruthLens aggregation layer.

---

## 1. Overview

`src/aggregation` is the **final fusion stage** of the TruthLens NLP pipeline. It sits between upstream inference (model logits, bias/emotion/narrative/graph analysis) and the API response layer. Its responsibilities are:

1. **Feature mapping** — translate raw multi-task model outputs (logits, probabilities, analysis-module dicts) into a uniform per-section feature profile.
2. **Calibration** — convert raw logits to well-calibrated probabilities at the logit boundary.
3. **Adaptive weighting** — scale each section's contribution by per-task confidence, entropy, and explainability signals.
4. **Score fusion** — produce three composite scores: `manipulation_risk`, `credibility_score`, and a final `truthlens_final_score`.
5. **Explanation** — attribute scores back to input tokens via Integrated Gradients or a lightweight profile-mode heuristic.
6. **Risk classification** — map continuous scores to `LOW / MEDIUM / HIGH` labels with configurable thresholds.
7. **Validation and monitoring** — range checks, logical-consistency checks, rolling metric history, entropy percentile tracking.

The aggregation layer is applied **once per article**, after upstream inference has finished and before the result is serialised to the API response.

---

## 2. Folder Architecture

```
src/aggregation/
├── __init__.py                  → Public surface re-exports (all symbols accessible via `from src.aggregation import …`)
├── aggregation_config.py        → Pydantic config models + YAML loader (single source of truth for group/weight constants)
├── aggregation_metrics.py       → Statistical utilities: calibration error, entropy, drift detection, rolling history
├── aggregation_pipeline.py      → Orchestrator: wires all sub-modules into a single `run()` call
├── aggregation_validator.py     → Post-scoring range and logical-consistency checks
├── calibration.py               → Calibrator classes: Temperature, Sigmoid, Isotonic, PassThrough + factory
├── feature_mapper.py            → Two-branch input adapter + single-pass TaskSignal extraction
├── risk_assessment.py           → Continuous score → LOW/MEDIUM/HIGH classification
├── score_explainer.py           → Integrated Gradients (model mode) and profile heuristic (lite mode)
├── score_normalizer.py          → Population-level ScoreNormalizer (offline fit, optional)
├── score_schema.py              → Pydantic output schemas (TaskScore, TruthLensScoreModel, etc.)
├── truthlens_score_calculator.py → Weighted composite score formulas (manipulation, credibility, final)
└── weight_manager.py            → Adaptive weight computation with confidence/entropy/explainability scaling
```

---

## 3. End-to-End Aggregation Flow

```
Upstream inference
       │
       ▼
 AggregationPipeline.run(model_outputs | profile, text, analysis_modules)
       │
       ├─ 1. Input normalisation ──────────────────────────────────────────
       │       Branch A: model_outputs  →  FeatureMapper.map_from_model_outputs
       │       Branch B: profile        →  _adapt_profile  →  FeatureMapper.map_from_model_outputs
       │
       ├─ 2. Signal extraction (single pass) ──────────────────────────────
       │       FeatureMapper.extract_task_signals
       │       → per-task TaskSignal { probability, confidence, entropy, max_class }
       │
       ├─ 3. Calibration (at logit boundary, inside FeatureMapper) ────────
       │       BaseCalibrator.transform(logits)  →  calibrated probabilities
       │
       ├─ 4. Explanation ──────────────────────────────────────────────────
       │       if model+tokenizer+text available: ScoreExplainer.explain_from_prediction (IG)
       │       else:                              ScoreExplainer.explain_profile (heuristic)
       │       → section_scores { "bias": 0.x, "emotion": 0.x, … }
       │
       ├─ 5. Adaptive weights ─────────────────────────────────────────────
       │       WeightManager.get_adaptive_weights(confidence, entropy, explanation_scores)
       │       → scaled + renormalised weight dict
       │
       ├─ 6. Score computation ────────────────────────────────────────────
       │       TruthLensScoreCalculator.compute_scores(profile, weights, explanation_scores)
       │       → section_scores, manipulation_risk, credibility_score, final_score
       │
       ├─ 7. Risk classification ──────────────────────────────────────────
       │       assess_truthlens_risks(scores)
       │       → { manipulation_risk, credibility_level, overall_truthlens_rating }
       │
       ├─ 8. Typed model assembly ─────────────────────────────────────────
       │       TruthLensScoreModel + TruthLensRiskModel + ExplanationModel
       │
       ├─ 9. Validation ───────────────────────────────────────────────────
       │       AggregationValidator.validate(scores)
       │       → { valid: bool, issues: [str] }
       │
       ├─10. Monitoring / uncertainty percentiles ─────────────────────────
       │       AggregationMetrics.update(scores)
       │       entropy p95 / p99 threshold alerts
       │
       └─11. Final Pydantic wrap ──────────────────────────────────────────
               TruthLensAggregationOutputModel(**result).model_dump()
               → typed, validated dict returned to caller
```

**Inter-module dependency order:**

```
aggregation_config  →  weight_manager  →  aggregation_pipeline
calibration         →  feature_mapper  →  aggregation_pipeline
score_schema        →  aggregation_pipeline
risk_assessment     →  aggregation_pipeline
score_explainer     →  aggregation_pipeline
truthlens_score_calculator ← aggregation_config (WEIGHT_GROUPS)
aggregation_metrics  ←  aggregation_pipeline
aggregation_validator ← aggregation_pipeline
score_normalizer     (standalone, loaded externally via state_dict)
```

---

## 4. File-by-File Deep Dive

---

### `aggregation_config.py`

**Purpose**

Single source of truth for every constant, threshold, and behavioral flag used across the aggregation layer. Eliminates the previously drifting copies of `WEIGHT_GROUPS` and `TASK_TO_GROUP` that silently broke per-group renormalization.

**Key Symbols**

| Symbol | Type | Description |
|---|---|---|
| `WEIGHT_GROUPS` | `Dict[str, tuple]` | Maps group names (`manipulation`, `credibility`, `final`) to the weight keys that belong to each group |
| `TASK_TO_GROUP` | `Dict[str, str]` | Maps task name (e.g. `"bias"`) to its group name (`"manipulation"`) |
| `SCALAR_WEIGHT_KEYS` | `tuple` | Keys treated as scalar multipliers, never renormalized (currently `"credibility_bias_penalty"`) |

**Config Models (Pydantic, `extra="forbid"`)**

| Class | Configures |
|---|---|
| `NormalizationConfig` | `method` (minmax/zscore/robust/quantile), `feature_range`, `clip` |
| `CalibrationConfig` | `method` (temperature/isotonic/sigmoid/none), `n_bins`, `enabled` |
| `UncertaintyConfig` | `enable_entropy`, `track_percentiles`, `p95_threshold`, `p99_threshold` |
| `WeightConfig` | `weights` dict, `version`, `allow_dynamic_adjustment`, `use_confidence`, `use_entropy`, `use_explainability`, `smoothing` |
| `RiskConfig` | `low_threshold` (0.3), `medium_threshold` (0.6), `uncertainty_penalty` (0.2) |
| `AttributionConfig` | `method` (integrated_gradients/shap/attention), `top_k`, `normalize`, confidence/entropy weighting flags |
| `FusionConfig` | `graph_influence_cap` (0.1), `explanation_blend` (0.5) |
| `DriftConfig` | `enabled`, `method` (kl/js/psi), `threshold` (0.1) |
| `MonitoringConfig` | `enabled`, `track_latency`, `track_confidence`, `track_entropy` |
| `AggregationConfig` | Composite root — nests all of the above + `task_types`, `batch_max_workers`, `strict_mode`, `enable_logging`, `enable_explanations`, `enable_risk` |

**`load_aggregation_config(config_path, override)`**

- Accepts either a standalone aggregation YAML or the global `config/config.yaml`.
- When pointed at the global config, extracts only the `aggregation:` sub-tree.
- Merges any `override` dict on top before constructing `AggregationConfig`.
- Returns a validated `AggregationConfig` instance.

**Edge cases**

- Missing `aggregation:` block in global config → the entire file is treated as a standalone aggregation YAML.
- `medium_threshold` ≤ `low_threshold` → Pydantic `ValueError` raised at construction time.

**Dependencies:** `pydantic`, `yaml`, stdlib only.

---

### `aggregation_metrics.py`

**Purpose**

Stateless numerical utilities for calibration quality, uncertainty, and distribution-drift measurement, plus a stateful `AggregationMetrics` collector that accumulates per-article score history for rolling diagnostics.

**Key Functions**

| Function | Inputs | Output | Notes |
|---|---|---|---|
| `compute_basic_stats(values)` | 1-D `np.ndarray` | `dict` with mean/std/min/max/median/p95/p99 | Safe against NaN/Inf via `_safe_array` |
| `compute_histogram(values, bins=10)` | 1-D array | `{counts, bin_edges}` | Fixed range [0, 1] |
| `expected_calibration_error(probs, labels, n_bins=15)` | predicted probs + true labels | ECE scalar | Handles 2-D (multiclass) and 1-D (binary) |
| `classwise_ece(probs, labels)` | 2-D probs + labels | `{class_0: float, …}` | One ECE per class |
| `brier_score(probs, labels)` | probs + labels | scalar | One-hot encoded for multiclass |
| `compute_entropy(probs)` | 2-D `[n, classes]` | 1-D entropy array | Row-wise Shannon entropy |
| `uncertainty_statistics(probs)` | 2-D probs | `{mean_entropy, p95_entropy, p99_entropy}` | Used for monitoring thresholds |
| `kl_divergence(p, q)` | two distributions | scalar | Both inputs L1-normalized before computation |
| `js_divergence(p, q)` | two distributions | scalar | Symmetric; uses midpoint `m = (p+q)/2` |
| `population_stability_index(expected, actual, bins=10)` | two 1-D arrays | scalar PSI | Percentile-based binning |
| `compute_distribution_shift(reference, current, bins=20)` | two 1-D arrays | `{kl, js, psi}` | Histogram density; L1-normalized |
| `compute_task_metrics(scores)` | `Dict[str, float]` | `{stats: …}` | Summary over one article |
| `compute_batch_metrics(batch_scores)` | `List[Dict[str, float]]` | per-key `{stats, histogram}` | Used for batch-level reporting |

**`AggregationMetrics` class**

| Method | Description |
|---|---|
| `update(scores)` | Appends a `Dict[str, float]` to `history` |
| `summarize()` | Calls `compute_batch_metrics(history)` |
| `reset()` | Clears history |
| `size()` | Returns current history length |

**Drift detection methods:**

- **KL divergence** — asymmetric, ∈ [0, ∞). Use when reference distribution is reliable.
- **JS divergence** — symmetric, ∈ [0, ln 2]. Preferred default.
- **PSI** — designed for score stability monitoring; values > 0.2 indicate significant shift.

**Dependencies:** `numpy` only.

---

### `aggregation_pipeline.py`

**Purpose**

Top-level orchestrator. Instantiates all sub-modules at construction time and wires them together in `run()`. Exposes both single-article and batch processing paths.

**Constructor**

```python
AggregationPipeline(config: Optional[AggregationConfig] = None)
```

Instantiates:
- `self.calibrator` — from `config.calibration.method` (passthrough until fitted offline)
- `self.mapper` — `FeatureMapper(strict=…, normalize=False, calibrator=…)`
- `self.weight_manager` — seeded from `config.weights.weights`, version, smoothing, uncertainty_penalty
- `self.calculator` — `TruthLensScoreCalculator(graph_influence_cap, explanation_blend)`
- `self.explainer` — `ScoreExplainer(method=…)`
- `self.risk_config` — converted from Pydantic `RiskConfig` via `from_pydantic_config`
- `self.validator` — `AggregationValidator()`
- `self.metrics` — `AggregationMetrics()` (rolling history, updated when `monitoring.enabled`)
- `self._task_types` — loaded from `AggregationConfig.task_types` or falls back to global app config

**`run()` — 14-step pipeline**

```python
def run(
    model_outputs: Optional[Dict[str, Any]] = None,
    *,
    text: Optional[str] = None,
    profile: Optional[Dict[str, Any]] = None,
    analysis_modules: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
```

| Step | Action |
|---|---|
| 1 | Accept either `model_outputs` or `profile`; convert profile via `_adapt_profile` |
| 2 | `mapper.map_from_model_outputs(source)` → `section_profile` |
| 3 | `mapper.extract_task_signals(source)` → per-task `{confidence, entropy}` cached once |
| 4 | (Calibration already applied inside `_extract_probs`) |
| 5 | `explainer.explain_from_prediction` (IG) or `explain_profile` (heuristic) → `explanation_scores` |
| 6 | `weight_manager.get_adaptive_weights(confidence, entropy, explanation_scores)` → `adaptive_weights` |
| 7 | `calculator.compute_scores(profile, weights, explanation_scores)` → `scores_raw` |
| 8 | `assess_truthlens_risks(scores)` → risk dict |
| 9 | Wrap in `TruthLensScoreModel`, `TruthLensRiskModel`, `ExplanationModel` |
| 10 | Assemble result dict with `schema_version`, `model_version`, `analysis_modules` |
| 11 | Attach external `analysis_modules` under namespaced key |
| 12 | Inject graph output if present in source |
| 13 | `validator.validate(flat_scores)` — range + logical consistency |
| 14 | `TruthLensAggregationOutputModel(**result).model_dump()` — Pydantic-validated dict |

Monitoring and uncertainty-percentile tracking run inside step 13a, before the final Pydantic wrap.

**`run_batch()`**

```python
def run_batch(
    batch_inputs: List[Dict[str, Any]],
    *,
    max_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
```

Fans out to a `ThreadPoolExecutor` when `max_workers > 1` (PERF-AG-5). Each article is fully stateless so parallel execution is safe.

**Edge cases**

- Empty `model_outputs={}` → warns operator; returns all-zero scores rather than crashing.
- `profile` passed alongside non-empty `model_outputs` → `model_outputs` wins.
- IG path requires: `explainer.model is not None`, `explainer.tokenizer is not None`, `text is not None`, `model_outputs is not None`. All other combinations silently fall back to profile mode.

**Dependencies:** all `src/aggregation/*`, `concurrent.futures`, `numpy`.

---

### `aggregation_validator.py`

**Purpose**

Lightweight post-scoring sanity checks to catch pipeline bugs before the result leaves the aggregation layer.

**`AggregationValidator.validate(result)`**

Input: `{"scores": {"credibility_score": float, "manipulation_risk": float, "final_score": float}}`

Checks performed:

1. **Range check** — each numeric score must be in `[0.0, 1.0]`.
2. **Logical consistency** — `credibility_score > 0.8 AND manipulation_risk > 0.8` is flagged as contradictory (a highly credible article should not simultaneously be highly manipulative).

Returns `{"valid": bool, "issues": [str]}`. A non-empty `issues` list triggers a warning log; the pipeline continues regardless (non-blocking).

**Dependencies:** `numpy`.

---

### `calibration.py`

**Purpose**

Converts raw logits to well-calibrated probability vectors. Applied inside `FeatureMapper._extract_probs` at the logit boundary (the only mathematically correct place).

**Class hierarchy**

```
BaseCalibrator
├── PassThroughCalibrator   — clip to [0,1], no transformation
├── TemperatureScaler       — single scalar T, gradient descent on NLL
├── SigmoidCalibrator       — per-class (a, b) Platt scaling
└── IsotonicCalibrator      — per-class isotonic regression (requires scikit-learn)
```

**`TemperatureScaler`**

- `fit(logits, labels)` — 100-step gradient descent on cross-entropy loss; learns scalar `T`.
- `transform(logits)` — returns `softmax(logits / T)`.
- Requires 2-D logits (`[n_samples, n_classes]`).

**`SigmoidCalibrator`**

- `fit(logits, labels)` — per-class gradient descent learning `a` (scale) and `b` (bias).
- `transform(logits)` — applies `sigmoid(a * logits + b)` per class.
- Safe for multilabel (independent per-class Bernoulli).

**`IsotonicCalibrator`**

- `fit(logits, labels)` — fits one `IsotonicRegression` per class; normalises output to sum to 1.
- Requires `scikit-learn`. Raises `ImportError` at construction if unavailable.

**`PassThroughCalibrator`**

- No parameters. Used when `method="none"` or calibrator has not been fitted yet.
- Simply clips input to `[0.0, 1.0]`.

**Factory**

```python
get_calibrator(method: str) -> BaseCalibrator
```

Accepted values: `"none"`, `"temperature"`, `"sigmoid"`, `"isotonic"`.

**Important:** Calibrators are fitted **offline** (e.g. via `scripts/calibrate.py`) and their state is loaded at startup. Until fitted, `PassThroughCalibrator` is used automatically.

**Dependencies:** `numpy`, optional `torch` (for tensor input), optional `scikit-learn`.

---

### `feature_mapper.py`

**Purpose**

Translates raw multi-task model outputs into a uniform per-section feature profile. Contains the single-pass `extract_task_signals` method that computes confidence and entropy exactly once per article (eliminating the triple recomputation that existed previously).

**`TaskSignal` dataclass**

```python
@dataclass
class TaskSignal:
    probability: np.ndarray   # cleaned probability vector
    confidence: float         # max(probability), ∈ [0, 1]
    entropy: float            # normalized entropy, ∈ [0, 1]
    max_class: int            # argmax of probability
    is_multilabel: bool       # Bernoulli (True) vs Categorical (False)
    had_nan: bool             # NaN detected in input (logged as warning)
```

**`DEFAULT_FEATURE_MAP`**

Defines the canonical sections and the raw output keys that feed them:

| Section | Raw key | Notes |
|---|---|---|
| `bias` | `bias_probability` | |
| `emotion` | `emotion_probability`, `emotion_intensity` | |
| `narrative` | `narrative_probability`, `narrative_score` | |
| `ideology` | `ideology_probability`, `ideology_score` | |
| `graph` | `graph_consistency` | |
| `discourse` | `discourse_probability` | Referenced by credibility formula |
| `argument` | `argument_probability` | |
| `analysis` | `analysis_probability` | Used in both manipulation and credibility formulas |

**`map_from_model_outputs()` — two-branch adapter**

- **Branch A** (`model_outputs[task]` contains `"probabilities"` or `"logits"`): extracts the winning-class probability → `{task}_probability` → routed through `DEFAULT_FEATURE_MAP`.
- **Branch B** (`model_outputs[task]` is a dict of `{feature_name: float}`): treated as a pre-built profile section and passed through verbatim (bypasses `DEFAULT_FEATURE_MAP`).

If both branches produce the same section name, Branch A keys take priority (they are more canonical).

**`extract_task_signals()` — single-pass confidence/entropy**

- Handles NaN-in-probabilities with an explicit warning (not silent coercion).
- Distinguishes `multilabel` (Bernoulli, per-label independent entropy) from `multiclass` (Categorical, renormalized entropy).
- Entropy is normalized to `[0, 1]` by dividing by `log(max(n_classes, 2))` for multiclass or `n_labels * log(2)` for multilabel.
- Returns `Dict[str, TaskSignal]` keyed by task name.

**Normalization** (`normalize=False` by default, NORM-AG-1)

When enabled, applies per-section max-norm. Disabled in the pipeline because values are already clipped to `[0, 1]` at emission and a second rescale adds no value.

**Dependencies:** `numpy`, `logging`.

---

### `risk_assessment.py`

**Purpose**

Maps continuous scores in `[0, 1]` to `LOW / MEDIUM / HIGH` risk labels, applying optional uncertainty penalties and key-level inversion.

**Core data structures**

```python
class RiskThresholds:
    low: float    # default 0.3
    medium: float # default 0.6

class RiskConfig:
    default: RiskThresholds
    per_key: Dict[str, RiskThresholds]   # overrides per score key
    invert_keys: Set[str]                # keys where high value = low risk
    weights: Dict[str, float]            # optional per-key weight multiplier
    uncertainty_penalty: float           # default 0.2
```

**`compute_risk_score(value, invert, uncertainty, config)`**

1. Validates `value` is finite and clips to `[0, 1]`.
2. If `invert=True`: `value = 1 - value` (used for `credibility_score` — high credibility means low risk).
3. Applies uncertainty penalty: `value *= (1 - uncertainty_penalty * uncertainty)`.
4. Clips result to `[0, 1]`.

**`score_to_level(score, thresholds)`**

```
score < low     → "LOW"
score < medium  → "MEDIUM"
score ≥ medium  → "HIGH"
```

**`assess_truthlens_risks(scores)`**

Dedicated wrapper that processes the three canonical TruthLens risk keys:

| Input key | Output key | Inverted? |
|---|---|---|
| `truthlens_manipulation_risk` | `manipulation_risk` | No |
| `truthlens_credibility_score` | `credibility_level` | Yes |
| `truthlens_final_score` | `overall_truthlens_rating` | No |

Uses a module-level `_DEFAULT_TRUTHLENS_CONFIG` instance allocated once at import time (PERF-AG-4).

**`assess_batch(batch_scores)`**

Applies `assess_risk_levels` to each dict in a list.

**`from_pydantic_config(pydantic_cfg, invert_keys)`**

Bridges the Pydantic `RiskConfig` (from `aggregation_config.py`) to the runtime `RiskConfig` class. Prevents the two shapes from drifting.

**Dependencies:** `numpy`.

---

### `score_explainer.py`

**Purpose**

Attributes pipeline scores back to input tokens. Supports two modes:
1. **Model mode** — Integrated Gradients over the encoder embeddings when a model and tokenizer are wired in.
2. **Profile mode** — lightweight section-feature heuristic when the model is unavailable (CPU/inference-only deployments).

**`SECTION_KEYWORDS`**

Exact-match vocabulary (post-detokenization) that maps words to sections:

```python
{
    "bias":      {"bias", "biased", "opinion", "opinions", "subjective"},
    "emotion":   {"happy", "sad", "anger", "angry", "fear", "joy", "joyful"},
    "narrative": {"story", "claim", "claims", "event", "events"},
    "discourse": {"however", "therefore", "because"},
    "graph":     {"relation", "relations", "connection", "connections"},
    "ideology":  {"liberal", "conservative"},
    "analysis":  {"evidence", "analysis"},
}
```

Note: substring matching was intentionally removed (TOK-AG-1) to prevent false attributions like `"joy"` → `"enjoyed"`.

**`ScoreExplainer.__init__(model, tokenizer, device, steps, method)`**

- `steps=32` alpha interpolation points for IG (balances compute vs. precision).
- Model placed on `device` at construction; eval mode set immediately.

**`_integrated_gradients(input_ids, attention_mask, task, target_idx)`**

Steps:
1. Extract word embeddings for the input.
2. Build `steps` interpolated inputs between a zero baseline and the actual embedding (broadcast alpha, single kernel launch — PERF-AG-2).
3. Forward all `steps` through the encoder + task head in one batch.
4. Backward on the target class logit sum.
5. Average gradients over the alpha dimension.
6. Hadamard product with `(embedding - baseline)` → IG attribution per token.
7. Zero out padding positions (TOK-AG-4).
8. L1 normalize.

**`_merge_subwords(tokens, importance)`**

Groups subword pieces back into whole-word surfaces (TOK-AG-3). Handles WordPiece (`##`), BPE (`Ġ`/U+0120), and SentencePiece (`▁`/U+2581) prefixes. Avoids N× attribution inflation for multi-piece words.

**`_section_scores(tokens, importance)`**

Calls `_merge_subwords`, then exact-matches each recovered surface form against `SECTION_KEYWORDS`.

**`explain_from_prediction(text, predictor_output, top_k=5)`**

Returns per-task `{top_tokens, section_scores, uncertainty}`. `top_tokens` is a list of `(word, importance)` tuples, sorted by `|importance|` descending.

**`explain_profile(profile, top_k=5)`**

Single-pass O(features) heuristic (REC-AG-4): iterates section → feature → value, accumulates section totals and a flat contribution list simultaneously, returns `{top_features, section_scores}`.

**Dependencies:** `numpy`, `torch`, `torch.nn`.

---

### `score_normalizer.py`

**Purpose**

Population-level normalizer fitted on a reference dataset and applied at inference time. **Not used per-article during inference** (NORM-AG-1 removed inline normalization). Intended for offline score calibration workflows and optional startup loading via `load_state_dict`.

**`ScoreNormalizer(method, feature_range, strict, clip)`**

| Method | Statistics fitted | Transform formula |
|---|---|---|
| `minmax` | min, max | `(x - min) / (max - min)` scaled to `feature_range` |
| `zscore` | mean, std | `(x - mean) / std` |
| `robust` | median, IQR | `(x - median) / IQR` |
| `quantile` | sorted array | rank / N (uniform distribution) |

**Key methods:**

- `fit(values)` — computes and stores statistics; sets `fitted=True`.
- `transform(values)` — applies stored statistics; raises `RuntimeError` if not fitted.
- `fit_transform(values)` — convenience composition.
- `normalize_probabilities(probs)` — row-L1-normalizes a 2-D probability matrix.
- `normalize_with_uncertainty(values, entropy)` — attenuates values by `(1 - entropy)` before transforming.
- `state_dict()` / `load_state_dict(state)` — serialization for offline persistence.

**GPU-AG-4 fix:** `_to_output` preserves the input tensor's dtype and device (including fp16/bf16), avoiding silent promotions to fp32 under autocast.

**Utility functions:**

- `log_scale(values)` — `log1p` transform for heavy-tailed distributions.
- `percentile_clip(values, low=1, high=99)` — clips outliers before normalization.
- `sigmoid_calibration(values)` — maps ℝ → (0, 1).

**Dependencies:** `numpy`, optional `torch`.

---

### `score_schema.py`

**Purpose**

Pydantic models that define the exact shape of the aggregation output. All output models use `frozen=True, extra="forbid"` so the contract is strictly enforced and serialized outputs are immutable.

**Model hierarchy**

```
TruthLensAggregationOutputModel
├── scores: TruthLensScoreModel
│   └── tasks: Dict[str, TaskScore]
├── raw_scores: Dict[str, float]
├── risks: TruthLensRiskModel
│   ├── manipulation_risk: Optional[RiskValue]
│   ├── credibility_level: Optional[RiskValue]
│   └── overall_truthlens_rating: Optional[RiskValue]
├── explanations: ExplanationModel
│   └── sections: Dict[str, ExplanationSection]
│       └── attributions: List[TokenAttribution]
├── metadata: Optional[SystemMetadata]
└── analysis_modules: Dict[str, Any]
```

**Key models:**

| Model | Fields | Notes |
|---|---|---|
| `TaskScore` | `score`, `confidence`, `probabilities`, `entropy` | `score` and `confidence` validated ∈ [0, 1] |
| `TruthLensScoreModel` | `tasks`, `manipulation_risk`, `credibility_score`, `final_score`, `uncertainty_summary` | |
| `RiskValue` | `level` (LOW/MEDIUM/HIGH), `score: Optional[float]` | |
| `TokenAttribution` | `token`, `importance`, `contribution`, `direction` (positive/negative) | |
| `ExplanationSection` | `method`, `top_features`, `attributions`, `section_score` | |
| `SystemMetadata` | `device`, `latency_ms`, `request_id`, `calibration_method`, `normalization_method` | |
| `TruthLensAggregationOutputModel` | root output | `extra="forbid"` — unknown keys raise `ValidationError` |

**Dependencies:** `pydantic`, stdlib.

---

### `truthlens_score_calculator.py`

**Purpose**

Computes the three composite scores (manipulation risk, credibility score, final TruthLens score) from the per-section feature profile using configurable weighted formulas.

**Constructor**

```python
TruthLensScoreCalculator(
    graph_influence_cap: float = 0.1,
    explanation_blend: float = 0.5,
)
```

Both parameters injected from `AggregationConfig.fusion` (WGT-AG-2).

**`compute_scores(profile, weights, explanation_scores)`**

Steps:
1. Copy `weights`; renormalize each of the three weight groups in-place so they sum to 1.
2. Log a debug warning for any required section absent from `profile`.
3. Aggregate any available graph signal key (`consistency`, `graph_density`, `avg_centrality`, etc.).
4. For each section: `val = mean(features) + graph_influence_cap × graph_signal`.
5. If explanation score available: `val = (1 - blend) × val + blend × explanation_score`.
6. Clip to `[0, 1]` → `section_scores`.
7. Compute composite scores.

**Composite score formulas:**

```
manipulation_risk =
    w["bias"] × s["bias"] +
    w["emotion"] × s["emotion"] +
    w["narrative"] × s["narrative"] +
    w["analysis_influence_manipulation"] × s["analysis"]

credibility_score =
    (w["discourse"] × s["discourse"] +
     w["graph"] × s["graph"] +
     w["analysis_influence_credibility"] × s["analysis"])
    × (1 − credibility_bias_penalty × s["bias"])

final_score =
    w["final_credibility"] × credibility +
    w["final_manipulation"] × (1 − manipulation) +
    w["final_ideology"] × (1 − ideology)
```

All outputs clipped to `[0, 1]`.

**Important:** Confidence and entropy are **not** applied here (CRIT-AG-7). They are baked into `weights` by `WeightManager.get_adaptive_weights` before this function is called. Applying them a second time would cause double-attenuation.

**`_aggregate(section_data)`**

Uses `math.fsum` (precision-aware) instead of `np.array(...).mean()` for the typical 1–3 feature vectors in each section (PERF-AG-3).

**Dependencies:** `numpy`, `math`, `.aggregation_config` (WEIGHT_GROUPS).

---

### `weight_manager.py`

**Purpose**

Manages the full lifecycle of aggregation weights: default initialization, config-file loading, adaptive scaling by confidence/entropy/explainability, group renormalization, smoothing, and thread-safe access.

**Default weights**

```python
DEFAULT_WEIGHTS = {
    # Manipulation group (renorm within group)
    "bias":                           0.40,
    "emotion":                        0.30,
    "narrative":                      0.20,
    "analysis_influence_manipulation": 0.10,
    # Credibility group (renorm within group)
    "discourse":                       0.55,
    "graph":                           0.35,
    "analysis_influence_credibility":  0.10,
    # Scalar (not renormalized)
    "credibility_bias_penalty":        0.20,
    # Final group (renorm within group)
    "final_credibility":               0.50,
    "final_manipulation":              0.30,
    "final_ideology":                  0.20,
}
```

**Constructor**

```python
WeightManager(
    weights: Optional[Dict[str, float]] = None,   # overrides merged with DEFAULT_WEIGHTS
    version: str = "v2",
    frozen: bool = False,                          # blocks adjust_weight() when True
    smoothing: float = 0.1,                        # α in convex blend base↔adaptive
    uncertainty_penalty: float = 0.2,              # entropy attenuation factor
    scale_clip: tuple = (0.5, 2.0),               # symmetric log-space clip for scale factors
)
```

At construction, the merged weights are validated, scalar keys clipped, and all groups renormalized.

**`get_adaptive_weights(confidence, entropy, explanation_scores)`**

1. Copy base weights to `scaled`.
2. For each group, compute a group-level scale factor:
   - `scale = 1.0`
   - If confidence provided: `scale *= mean_confidence_for_group`
   - If entropy provided: `scale *= max(0, 1 − uncertainty_penalty × mean_entropy_for_group)`
   - If explanation scores provided: `scale *= (1 + mean_explanation_score_for_group_keys)`
3. Clip scale to `scale_clip` range (0.5–2.0) — symmetric to avoid asymmetric bias (WGT-AG-4).
4. Apply scale to all keys in the group.
5. Renormalize each group (so the total stays 1.0 within the group).
6. Convex smooth: `w_final = (1 − smoothing) × w_base + smoothing × w_scaled`.
7. Final renormalization pass for floating-point cleanup.
8. Pass scalar keys (`credibility_bias_penalty`) through unchanged.

Returns `Dict[str, float]` — the adaptive weight vector ready for `TruthLensScoreCalculator`.

**`adjust_weight(key, value)`**

Thread-safe single-key update. Re-validates, re-clips scalars, and re-normalizes after each update. Raises `RuntimeError` if `frozen=True`.

**`load_weights_from_config(config_path)`**

Loads a JSON file (not YAML), merges over current weights, revalidates. Thread-safe via `self._lock`.

**`_validate_weights(weights)`**

- All values must be finite non-negative numerics.
- Each group must sum to > 0 (catches accidental all-zero group).

**`_aggregate_group_signal(signal, default)`**

Averages per-task `confidence` or `entropy` values into per-group aggregates, filtering NaN/Inf inputs. Returns `default` for any group with no usable values.

**Dependencies:** `numpy`, `json`, `threading`, `.aggregation_config`.

---

## 5. Aggregation Definitions Table

| Aggregated Feature | Source Column(s) | Grouping Key(s) | Aggregation Type | Weight | Description |
|---|---|---|---|---|---|
| `bias` section score | `bias_probability` | — | mean of features + graph adjustment | 0.40 (manipulation) | Probability of biased framing |
| `emotion` section score | `emotion_probability`, `emotion_intensity` | — | mean of features | 0.30 (manipulation) | Emotional manipulation signal |
| `narrative` section score | `narrative_probability`, `narrative_score` | — | mean of features | 0.20 (manipulation) | Narrative framing strength |
| `discourse` section score | `discourse_probability` | — | feature value | 0.55 (credibility) | Logical discourse quality |
| `graph` section score | `graph_consistency` | — | mean of graph signal keys | 0.35 (credibility) | Cross-entity consistency |
| `ideology` section score | `ideology_probability`, `ideology_score` | — | mean of features | — (final) | Ideological lean |
| `analysis` section score | `analysis_probability` | — | feature value | 0.10 each group | Dual-use analysis signal |
| `manipulation_risk` | bias, emotion, narrative, analysis | manipulation group | weighted sum | see formulas | Composite manipulation score |
| `credibility_score` | discourse, graph, analysis minus bias penalty | credibility group | weighted sum × penalty | see formulas | Composite credibility score |
| `final_score` | credibility, manipulation, ideology | final group | weighted sum of inversions | see formulas | Overall TruthLens score |
| `explanation_scores` | section features | per section | mean of absolute feature values | — | Attribution proxy when IG unavailable |
| `confidence` | task `probabilities` | per task | max(probability_vector) | — | Model confidence per task |
| `entropy` | task `probabilities` | per task | normalized Shannon entropy | — | Prediction uncertainty per task |

---

## 6. Data Contracts

### Input Schema (to `AggregationPipeline.run`)

**Branch A — model outputs:**
```python
{
    "bias": {
        "probabilities": List[float],  # or "logits": List[float]
        "logits": List[float],         # optional if probabilities present
    },
    "emotion": { "probabilities": [...], ... },
    "narrative": { ... },
    "discourse": { ... },
    "graph": { ... },
    "ideology": { ... },
    "analysis": { ... },
    "graph_output": Optional[Any],     # graph module output
}
```

**Branch B — pre-built profile:**
```python
{
    "bias":      {"probability": float, ...},
    "emotion":   {"probability": float, "intensity": float},
    "narrative": {"probability": float, "score": float},
    ...
}
```

All numeric values expected in `[0, 1]`; values outside this range are clipped.

### Output Schema (from `AggregationPipeline.run`)

```python
{
    "schema_version": str,           # e.g. "v2"
    "model_version": str,            # e.g. "truthlens-v2"
    "scores": {
        "tasks": {
            "bias": {"score": float, "confidence": None, "probabilities": None, "entropy": None},
            ...
        },
        "manipulation_risk": float,  # ∈ [0, 1]
        "credibility_score": float,  # ∈ [0, 1]
        "final_score": float,        # ∈ [0, 1]
        "uncertainty_summary": None,
    },
    "raw_scores": {
        "manipulation_risk": float,
        "credibility_score": float,
        "final_score": float,
        "section_scores": { "bias": float, ... },
    },
    "risks": {
        "manipulation_risk":          {"level": "LOW|MEDIUM|HIGH", "score": float},
        "credibility_level":          {"level": ..., "score": float},
        "overall_truthlens_rating":   {"level": ..., "score": float},
    },
    "explanations": {
        "sections": {
            "bias": {
                "method": "integrated_gradients",
                "top_features": ["word1", ...],
                "attributions": [{"token": str, "importance": float, "contribution": float, "direction": str}],
                "section_score": float,
            },
        },
    },
    "analysis_modules": {
        "weights":    Dict[str, float],
        "entropy":    Dict[str, float],
        "confidence": Dict[str, float],
        "validation": {"valid": bool, "issues": [str]},
        "uncertainty": {"p95": float, "p99": float, "exceeds_p95_threshold": bool, ...},  # optional
        "graph":       Any,                                                                # optional
        "external_analysis": Dict[str, Any],                                              # optional
    },
    "metadata": None,
}
```

All output scores are guaranteed finite via `_safe_unit` (NaN/Inf → 0.0) and clipped to `[0, 1]`.

---

## 7. Temporal & Windowing Logic

The aggregation layer processes **one article at a time** (stateless per-article computation). There are no rolling windows, sliding windows, or time-based grouping within a single pipeline call.

**Rolling history** is maintained by `AggregationMetrics` in memory across multiple calls to `run()`. This is a simple append-only list; no time-based eviction occurs. Call `metrics.reset()` to clear.

**Drift detection** (`aggregation_metrics.py`) operates on two static arrays (`reference` and `current`) supplied by the caller. The caller is responsible for partitioning the population across time windows before calling `compute_distribution_shift`.

**PSI binning** uses percentile-based breakpoints derived from the `expected` distribution. This is robust to irregular score distributions but requires the `expected` array to be representative.

---

## 8. Config Integration

All behavioral parameters flow through `AggregationConfig`, which reads the `aggregation:` block of `config/config.yaml`.

### Parameters and their effects

| Config path | Default | Effect |
|---|---|---|
| `aggregation.calibration.method` | `"temperature"` | Selects calibrator class at pipeline construction |
| `aggregation.weights.weights` | `{}` | Overrides to `DEFAULT_WEIGHTS` (partial override supported) |
| `aggregation.weights.smoothing` | `0.1` | α in convex blend base↔adaptive weights |
| `aggregation.weights.use_confidence` | `true` | Enables confidence scaling in adaptive weights |
| `aggregation.weights.use_entropy` | `true` | Enables entropy penalty in adaptive weights |
| `aggregation.weights.use_explainability` | `true` | Enables explanation-score boosting |
| `aggregation.risk.low_threshold` | `0.3` | Below this → LOW |
| `aggregation.risk.medium_threshold` | `0.6` | Below this → MEDIUM; at or above → HIGH |
| `aggregation.risk.uncertainty_penalty` | `0.2` | Fraction by which entropy attenuates risk score |
| `aggregation.fusion.graph_influence_cap` | `0.1` | Max graph signal added per section |
| `aggregation.fusion.explanation_blend` | `0.5` | Blend ratio of explanation score into section score |
| `aggregation.batch_max_workers` | `1` | Thread-pool workers for `run_batch` |
| `aggregation.strict_mode` | `false` | If true, FeatureMapper raises on non-numeric values |
| `aggregation.enable_explanations` | `true` | Gates the ScoreExplainer call |
| `aggregation.enable_risk` | `true` | Gates risk assessment |
| `aggregation.monitoring.enabled` | `true` | Gates AggregationMetrics updates |
| `aggregation.uncertainty.p95_threshold` | `0.8` | Alert threshold for entropy p95 |
| `aggregation.uncertainty.p99_threshold` | `0.95` | Alert threshold for entropy p99 |
| `explainability.aggregation_weights.shap` | `0.35` | Per-method weight for explainability fusion (see CFG-3) |

### Feature flags

| Flag | When true |
|---|---|
| `enable_explanations` | Runs ScoreExplainer (either IG or profile mode) |
| `enable_risk` | Runs `assess_truthlens_risks` and populates `risks` block |
| `monitoring.enabled` | Calls `self.metrics.update(flat_scores)` per article |
| `uncertainty.track_percentiles` | Computes entropy p95/p99 and attaches to `analysis_modules.uncertainty` |
| `weights.allow_dynamic_adjustment` | Sets `WeightManager.frozen=False`; enables runtime `adjust_weight` calls |

---

## 9. Data Integrity & Validation

### Missing values

- **NaN/Inf in probabilities** — `FeatureMapper.extract_task_signals` emits a `WARNING` log and coerces to 0.0 via `np.nan_to_num`.
- **NaN/Inf in scores** — `_safe_unit` in the pipeline maps to 0.0 and clips to `[0, 1]`.
- **Missing task sections** — `TruthLensScoreCalculator` logs a DEBUG message for missing required sections; missing sections contribute 0.0 to composites.
- **Empty source dict** — pipeline warns and returns all-zero scores rather than crashing.

### Duplicate records

No deduplication occurs within the aggregation layer. Input uniqueness is assumed to be enforced by the upstream inference pipeline.

### Outliers

- Feature values outside `[0, 1]` are clipped by `FeatureMapper` at intake.
- `WeightManager` clips scale factors to `scale_clip=(0.5, 2.0)`, bounding weight modulation to ±1 octave.
- `ScoreNormalizer.percentile_clip(low=1, high=99)` is available for offline preprocessing.

### Post-aggregation validation

`AggregationValidator.validate` performs:
1. Range enforcement: each score ∈ `[0, 1]`.
2. Contradiction detection: `credibility > 0.8 AND manipulation > 0.8` → issue logged.

Validation failures are non-blocking (warning logged, pipeline continues).

---

## 10. Optimization & Scalability

| Optimization | Location | Audit ref | Description |
|---|---|---|---|
| Single-pass confidence/entropy | `FeatureMapper.extract_task_signals` | REC-AG-1 | Eliminated triple recomputation of softmax per article |
| Batched IG alpha broadcast | `ScoreExplainer._integrated_gradients` | PERF-AG-2 | Single `alphas × delta` broadcast replaces 32 tensor copies |
| `math.fsum` for micro-vectors | `TruthLensScoreCalculator._aggregate` | PERF-AG-3 | Avoids `np.array` construction overhead for 1–3 feature vectors |
| Cached `_DEFAULT_TRUTHLENS_CONFIG` | `risk_assessment.py` | PERF-AG-4 | RiskConfig allocated once at module load, not per article |
| ThreadPoolExecutor batch fan-out | `AggregationPipeline.run_batch` | PERF-AG-5 | `batch_max_workers > 1` enables parallel article processing |
| No per-article normalization | `AggregationPipeline.run` | NORM-AG-1 | Removed inline `fit_transform` that distorted single-feature sections |
| No per-article calibration | `AggregationPipeline.run` | CRIT-AG-6 | Calibration moved to logit boundary in `FeatureMapper._extract_probs` |
| Group renormalization once | `WeightManager.get_adaptive_weights` | CRIT-AG-10 | Renorm before smoothing so the α parameter is interpretable |

**Memory footprint:**
- `AggregationMetrics.history` grows unboundedly in long-running processes. Call `metrics.reset()` periodically or cap upstream if memory is a concern.
- `ScoreNormalizer.stats["sorted"]` stores the full sorted array for quantile normalization. Use minmax/zscore for large populations.

**Distributed execution:** Not implemented. The pipeline is CPU/single-process by default. `run_batch` with `max_workers > 1` uses Python threads (GIL-limited for pure Python but effective when model inference releases the GIL, e.g. with PyTorch).

---

## 11. Extensibility Guide

### Adding a new section / task

1. Add the task → raw key mapping to `DEFAULT_FEATURE_MAP` in `feature_mapper.py`.
2. Assign the task to a weight group in `TASK_TO_GROUP` (`aggregation_config.py`).
3. Add the task's weight key to the corresponding `WEIGHT_GROUPS` tuple.
4. Add a default weight to `DEFAULT_WEIGHTS` in `weight_manager.py`.
5. If the section contributes to `manipulation_risk` or `credibility_score`, update the corresponding formula in `truthlens_score_calculator.py`.
6. Optionally add section keywords to `SECTION_KEYWORDS` in `score_explainer.py`.

### Adding a new calibration method

1. Subclass `BaseCalibrator` in `calibration.py`.
2. Implement `fit(logits, labels)` and `transform(logits)`.
3. Register the new method string in `get_calibrator`.
4. Add the method to the `Literal` in `CalibrationConfig`.

### Adding a new weight group

1. Add the group name and member keys to `WEIGHT_GROUPS` in `aggregation_config.py`.
2. Update `TASK_TO_GROUP` so each task maps to the new group.
3. Add default weights to `DEFAULT_WEIGHTS`.
4. Update composite score formulas in `TruthLensScoreCalculator` if the group should feed a composite.

### Adding a new drift metric

1. Implement the function in `aggregation_metrics.py` following the `_normalize`/`_safe_array` pattern.
2. Register the method in `compute_distribution_shift`.
3. Add the method string to `DriftConfig.method` Literal in `aggregation_config.py`.

### Changing risk thresholds per key

Pass a `RiskConfig` with a populated `per_key` dict:

```python
risk_cfg = RiskConfig(
    per_key={"manipulation_risk": RiskThresholds(low=0.4, medium=0.7)}
)
```

---

## 12. Common Pitfalls / Risks

| Risk | Root cause | Mitigation |
|---|---|---|
| All-zero section scores | Upstream task names don't match `DEFAULT_FEATURE_MAP` keys | Check debug log for "missing sections"; verify task key names against `DEFAULT_FEATURE_MAP` |
| Weight group sums to zero | All weights for a group set to 0 in override | `WeightManager._validate_weights` raises `ValueError` at startup |
| Double confidence attenuation | Passing `confidence`/`entropy` to both `WeightManager.get_adaptive_weights` and `TruthLensScoreCalculator.compute_scores` | The calculator explicitly ignores these kwargs (CRIT-AG-7); pass them only to the weight manager |
| Calibrator silently bypassed | Calibrator not fitted before inference | Check `calibrator.fitted`; if False, `PassThroughCalibrator` is used (clip only) |
| Contradictory risk labels | `credibility_score > 0.8 AND manipulation_risk > 0.8` | Validator flags it; review upstream models for conflicting outputs |
| NaN propagation | Model produces NaN logits or probabilities | `FeatureMapper` warns and coerces to 0.0; search for upstream model bugs |
| Entropy p95 threshold alerts | High model uncertainty on out-of-distribution text | Check `analysis_modules.uncertainty.exceeds_p95_threshold` in output |
| Stale `AggregationMetrics` history | History never reset in long-running service | Call `pipeline.metrics.reset()` at batch boundaries or implement size cap |
| Per-article normalization regression | Re-introducing `normalize=True` in `FeatureMapper` | Any per-article `fit_transform` will collapse single-feature sections to 0.0 |
| Graph signal key not found | Upstream graph module changed its output key | Add new key to `_GRAPH_SIGNAL_KEYS` tuple in `truthlens_score_calculator.py` |

---

## 13. Example Usage

### Minimal: model outputs → aggregated scores

```python
from src.aggregation import AggregationPipeline, load_aggregation_config

config = load_aggregation_config("config/config.yaml")
pipeline = AggregationPipeline(config=config)

model_outputs = {
    "bias": {
        "probabilities": [0.2, 0.8],   # [not_biased, biased]
        "logits": [-1.4, 1.4],
    },
    "emotion": {
        "probabilities": [0.6, 0.4],
    },
    "narrative": {
        "probabilities": [0.7, 0.3],
    },
    "discourse": {
        "probabilities": [0.5, 0.5],
    },
    "graph": {
        "probabilities": [0.4, 0.6],
    },
    "ideology": {
        "probabilities": [0.45, 0.55],
    },
    "analysis": {
        "probabilities": [0.55, 0.45],
    },
}

result = pipeline.run(model_outputs=model_outputs, text="The government claimed …")

print(result["scores"]["manipulation_risk"])   # e.g. 0.62
print(result["scores"]["credibility_score"])   # e.g. 0.41
print(result["scores"]["final_score"])         # e.g. 0.48
print(result["risks"]["manipulation_risk"])    # {"level": "HIGH", "score": 0.62}
```

### Pre-built profile (BiasProfileBuilder output)

```python
profile = {
    "bias":      {"probability": 0.75, "intensity": 0.60},
    "emotion":   {"probability": 0.40},
    "narrative": {"probability": 0.55, "score": 0.50},
    "discourse": {"probability": 0.65},
    "graph":     {"consistency": 0.70},
    "ideology":  {"probability": 0.45, "score": 0.40},
    "analysis":  {"probability": 0.50},
}

result = pipeline.run(profile=profile)
```

### Batch processing

```python
inputs = [
    {"model_outputs": mo1, "text": "Article 1 …"},
    {"model_outputs": mo2, "text": "Article 2 …"},
]

results = pipeline.run_batch(inputs, max_workers=4)
```

### Drift monitoring (comparing yesterday vs today)

```python
from src.aggregation import AggregationMetrics
from src.aggregation.aggregation_metrics import compute_distribution_shift

yesterday_scores = [0.3, 0.55, 0.4, 0.7, ...]   # manipulation_risk values
today_scores     = [0.6, 0.8, 0.75, 0.9, ...]

shift = compute_distribution_shift(
    reference=yesterday_scores,
    current=today_scores,
)
print(shift)  # {"kl": 0.12, "js": 0.08, "psi": 0.23}
# psi > 0.2 indicates significant distribution shift
```

### Accessing rolling metrics

```python
# After processing many articles:
summary = pipeline.metrics.summarize()
# Returns per-key {stats: {mean, std, min, max, median, p95, p99}, histogram: {counts, bin_edges}}
print(summary["final_score"]["stats"]["mean"])
```
