# `src/features` — TruthLens Feature Engineering Module

**Canonical reference documentation for the hand-engineered feature layer.**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Folder Architecture](#2-folder-architecture)
3. [End-to-End Pipeline](#3-end-to-end-pipeline)
4. [File-by-File Deep Dive](#4-file-by-file-deep-dive)
5. [Feature Definitions Table](#5-feature-definitions-table)
6. [Feature Selection Logic](#6-feature-selection-logic)
7. [Data Contracts](#7-data-contracts)
8. [Leakage & Bias Prevention](#8-leakage--bias-prevention)
9. [Config Integration](#9-config-integration)
10. [Optimization & Efficiency](#10-optimization--efficiency)
11. [Extensibility Guide](#11-extensibility-guide)
12. [Common Pitfalls / Risks](#12-common-pitfalls--risks)
13. [Example Usage](#13-example-usage)

---

## 1. Overview

`src/features/` is the **hand-engineered signal layer** of the TruthLens misinformation-detection system. It transforms raw article text into a fixed-width numerical feature vector that downstream ML model heads consume for classification, calibration, and explanation.

The module covers **nine semantic domains**:

| Domain | What it captures |
|---|---|
| **Bias** | Loaded language, subjectivity, uncertainty hedging, polarity |
| **Framing** | Economic / moral / security / human-interest / conflict frames |
| **Ideology** | Left / right leaning, polarisation, group references |
| **Discourse** | Causal, contrast, additive, sequential, evidential connectors |
| **Argument** | Claim, premise, evidence, counter-argument, rhetorical questions |
| **Emotion** | 11-label distribution (EMOTION-11), intensity, trajectory, polarity |
| **Narrative** | Hero / villain / victim roles, conflict arc, crisis, polarisation |
| **Propaganda** | Name-calling, fear appeal, exaggeration, glittering generalities, us-vs-them |
| **Graph** | Entity co-occurrence and interaction graph topology |
| **Text** | Lexical richness, syntactic complexity, semantic embedding stats |

### Key design invariants

- **All feature values are `float` in `[0.0, MAX_CLIP]`** (or unbounded raw magnitudes explicitly marked `_safe_unbounded`).
- **`NaN` / `±inf` are replaced with `0.0` at the extractor boundary** — the downstream model never sees non-finite values.
- **Every extractor returns a `dict[str, float]`** — no tensors, no lists, no optional types at the boundary.
- **Scaling is forbidden inside individual extractors.** Per-row per-extractor magic divisors were removed in the audit; all normalisation happens in `FeatureScalingPipeline` on the training corpus.
- **Empty input → fixed-key zero dict**, never an empty `{}` (where a consistent shape matters for downstream schema validation).

---

## 2. Folder Architecture

```
src/features/
│
├── __init__.py                     # Public re-exports (FeatureConfig, bootstrap_feature_registry, …)
├── feature_config.py               # FeatureConfig dataclass — single source of truth for runtime config
├── feature_schema.py               # ALL_FEATURES master list + per-domain lists + FEATURE_SECTIONS map
├── feature_schema_validator.py     # FeatureSchemaValidator — validates, imputes, and aligns feature dicts
├── feature_statistics.py           # FeatureStatistics — variance, skewness, constant detection
├── feature_bootstrap.py            # bootstrap_feature_registry() — 19-module import orchestrator
├── feature_pruning.py              # FeaturePruner — variance + correlation pruning
├── feature_report.py               # FeatureReport — human-readable summary generator
├── dataset_feature_generator.py    # DatasetFeatureGenerator — parallel batch generation
├── runtime_config.py               # Thread-safe mutable runtime flags
│
├── base/                           # Shared primitives used by all extractors
│   ├── base_feature.py             # BaseFeature, FeatureContext (with LRU bounded context cache)
│   ├── feature_registry.py         # FeatureRegistry (thread-safe, freeze-able)
│   ├── feature_config.py           # (base-layer alias, re-exports FeatureConfig)
│   ├── lexicon_loader.py           # load_lexicon / load_lexicon_set / load_lexicon_dict (lazy, thread-safe)
│   ├── lexicon_matcher.py          # LexiconMatcher / WeightedLexiconMatcher (vectorised np.isin)
│   ├── matrix_build.py             # Feature matrix utilities
│   ├── numerics.py                 # EPS, MAX_CLIP, normalized_entropy
│   ├── segmentation.py             # split_sentences, heuristic_entities (canonical — shared by 4 extractors)
│   ├── spacy_doc.py                # ensure_spacy_doc / set_spacy_doc (per-context Doc cache)
│   ├── spacy_loader.py             # get_shared_nlp() (singleton spaCy model)
│   ├── text_signals.py             # get_text_signals() (shared caps + exclamation + question signals)
│   ├── tokenization.py             # ensure_tokens_word, ensure_tokens_word_counter, tokenize_words
│   └── __init__.py
│
├── analysis/                       # Bridge adapters to src.analysis.* analyzers
│   ├── analysis_adapter_features.py  # 4 adapters: argument mining, discourse coherence, framing, ideological
│   └── __init__.py
│
├── bias/                           # Bias / framing / ideology extractors
│   ├── bias_features.py            # BiasFeaturesV2 (9 features)
│   ├── bias_lexicon_features.py    # BiasLexiconFeatures
│   ├── bias_lexicon.py             # Seed lexicon (inline, deprecated by lexicon_loader)
│   ├── framing_features.py         # FramingFeatures (11 features)
│   ├── ideological_features.py     # IdeologicalFeatures (9 features)
│   └── __init__.py
│
├── cache/                          # Two-tier (memory LRU + disk pickle) feature cache
│   ├── cache_manager.py            # CacheManager — key generation, hit/miss counters, batch API
│   ├── feature_cache.py            # FeatureCache — atomic pickle write, TTL+size prune
│   └── __init__.py
│
├── discourse/                      # Discourse / argument structure
│   ├── discourse_features.py       # DiscourseFeatures (7 features)
│   ├── argument_structure_features.py  # ArgumentStructureFeatures (7 features)
│   └── __init__.py
│
├── emotion/                        # Emotion signal layer (hybrid lexicon + transformer)
│   ├── emotion_schema.py           # EMOTION_LABELS (11), EMOTION_TERMS, polarity groups
│   ├── emotion_features.py         # EmotionFeatures — per-label distribution + polarity
│   ├── emotion_intensity_features.py  # EmotionIntensityFeatures — hybrid DistilRoBERTa + lexicon
│   ├── emotion_trajectory_features.py # EmotionTrajectoryFeatures — per-sentence arc analysis
│   ├── emotion_target_features.py  # EmotionTargetFeatures — entity-targeted emotion
│   ├── emotion_lexicon_features.py # EmotionLexiconFeatures
│   ├── emotion_lexicon.py          # Seed lexicon
│   └── __init__.py
│
├── fusion/                         # Post-extraction aggregation, scaling, selection
│   ├── feature_fusion.py           # FeatureFusion — per-extractor indicator flags + merge
│   ├── feature_merger.py           # merge_features() — dict union with conflict logging
│   ├── feature_reduction.py        # VarianceThresholdSelector, CorrelationSelector, TopKSelector, FeatureSelectionPipeline
│   ├── feature_scaling.py          # FeatureScalingPipeline (StandardScaler / MinMaxScaler / RobustScaler)
│   ├── feature_selection.py        # Backward-compat shim → feature_reduction
│   └── __init__.py
│
├── graph/                          # Graph topology features
│   ├── entity_graph_features.py    # EntityGraphFeatures (7 features, spaCy NER + heuristic fallback)
│   ├── interaction_graph_features.py  # InteractionGraphFeatures
│   └── __init__.py
│
├── importance/                     # Post-hoc feature importance
│   ├── shap_importance.py          # ShapImportance (shim → src.evaluation.importance)
│   ├── permutation_importance.py   # PermutationImportance
│   ├── feature_ablation.py         # FeatureAblation
│   └── __init__.py
│
├── narrative/                      # Narrative arc + conflict features
│   ├── narrative_features.py       # NarrativeFeatures (12 features)
│   ├── narrative_role_features.py  # NarrativeRoleFeatures
│   ├── narrative_frame_features.py # NarrativeFrameFeatures
│   ├── conflict_features.py        # ConflictFeatures
│   └── __init__.py
│
├── pipelines/                      # High-level orchestrators
│   ├── feature_pipeline.py         # FeaturePipeline + partition_feature_sections()
│   ├── batch_feature_pipeline.py   # BatchFeaturePipeline (parallel worker dispatch)
│   ├── feature_engineering_pipeline.py  # FeatureEngineeringPipeline (7-stage: extract→validate→prune→stats→scale→select→report)
│   ├── feature_schema.py           # Pipeline-local schema helpers
│   └── __init__.py
│
├── propaganda/                     # Propaganda technique detection
│   ├── propaganda_features.py      # PropagandaFeatures (12 features)
│   ├── manipulation_patterns.py    # ManipulationPatterns
│   ├── propaganda_lexicon_features.py # PropagandaLexiconFeatures
│   └── __init__.py
│
├── text/                           # Surface-form text features
│   ├── lexical_features.py         # LexicalFeatures (9 features — TTR, hapax, entropy, Yule's K)
│   ├── semantic_features.py        # SemanticFeatures (8 features — embedding stats + availability flag)
│   ├── syntactic_features.py       # SyntacticFeatures (8 features — POS entropy, dep depth, spaCy batch)
│   ├── token_features.py           # TokenFeatures
│   └── __init__.py
│
└── utils/
    └── tfidf_engineering.py        # TFIDFEngineering (corpus-level TF-IDF feature generator)
```

---

## 3. End-to-End Pipeline

```
Raw text (str)
    │
    ▼
FeatureContext(text=..., metadata=..., embeddings=...)
    │
    ├─── ensure_tokens_word()        ← Unicode word tokenisation, cached on ctx.tokens_word
    │                                   (runs once per context at the top of FeatureFusion.extract)
    │
    ▼
FeatureFusion.extract(context)  or  BatchFeaturePipeline.process(contexts)
    │
    ├─── For each registered extractor:
    │        BaseFeature.safe_extract(context)
    │            ├── check enabled flag
    │            ├── lazy initialize() on first call
    │            ├── call extract(context)   ← override per-extractor
    │            ├── _validate_output()       ← all keys str, all values finite float
    │            └── _fallback() on error (fail_silent=True by default)
    │
    ├─── merge_features(outputs)     ← flat dict union, last-writer-wins with a warning on collision
    │
    ▼
Flat feature dict  {str: float}
    │
    ▼  (inside FeatureEngineeringPipeline.process)
    ├── FeatureSchemaValidator.validate_batch()   ← impute missing keys, clip, dtype-check
    ├── FeaturePruner.transform()                 ← drop zero-variance + high-correlation columns
    ├── FeatureStatistics (logging only)
    ├── FeatureScalingPipeline.transform()        ← StandardScaler / MinMax / Robust (fit on train set)
    ├── FeatureSelectionPipeline.transform()      ← TopK / variance / correlation selectors
    └── FeatureReport (optional, save to disk)
    │
    ▼
Final feature vector  (dict or np.ndarray, depending on return_vector flag)
    │
    ▼
Downstream ML heads (bias head, emotion head, narrative head, …)
```

### Shared caches inside a single FeatureContext

| Cache slot | Type | What lives there |
|---|---|---|
| `ctx.tokens_word` | `List[str]` | Lowercased Unicode word tokens |
| `ctx.cache["lex_entropy"]` | `float` | Per-sample LRU (bounded, 256 entries default) |
| `ctx.cache["spacy_doc"]` | `spacy.Doc` | Parsed spaCy Doc, seeded by SyntacticFeatures.extract_batch |
| `ctx.cache["analysis_*"]` | `dict` | Raw analyzer output, keyed per adapter |
| `ctx.shared` | `dict` | Batch-wide shared data (set explicitly; avoid cross-sample leakage — see §8) |

---

## 4. File-by-File Deep Dive

### `feature_config.py` — `FeatureConfig`

Single-source-of-truth dataclass for the entire pipeline. Fields are read from environment variables at construction time and pushed into `runtime_config` via `apply_to_runtime()`. New code should create a `FeatureConfig`, configure it, call `apply_to_runtime()`, then call `bootstrap_feature_registry(config=cfg)` — never mutate `runtime_config` directly.

**Key fields:**

| Field | Env var | Default |
|---|---|---|
| `emotion_model` | `TRUTHLENS_EMOTION_MODEL` | `j-hartmann/emotion-english-distilroberta-base` |
| `transformer_enabled` | `TRUTHLENS_DISABLE_TRANSFORMER` | `True` |
| `transformer_max_batch` | `TRUTHLENS_TRANSFORMER_MAX_BATCH` | `64` |
| `transformer_chunk_length` | `TRUTHLENS_TRANSFORMER_CHUNK` | `256` |
| `transformer_chunk_stride` | `TRUTHLENS_TRANSFORMER_STRIDE` | `64` |
| `spacy_model` | `TRUTHLENS_SPACY_MODEL` | `en_core_web_sm` |
| `cache_max_memory_items` | `TRUTHLENS_CACHE_MAX_ITEMS` | `10 000` |
| `cache_max_memory_bytes` | `TRUTHLENS_CACHE_MAX_BYTES` | `512 MB` |
| `feature_context_cache_size` | `TRUTHLENS_CONTEXT_CACHE_SIZE` | `256` |
| `analysis_adapters_strict` | `TRUTHLENS_ANALYSIS_ADAPTERS_STRICT` | `False` |
| `torch_thread_cap` | `TRUTHLENS_TORCH_THREADS` | `4` |

---

### `feature_schema.py` — Master Feature List

Canonical authority for every feature name that the pipeline may emit. All per-domain lists must stay in lock-step with the `extract()` keys of the corresponding extractor.

**Domain lists (live, post-audit):**

| List constant | Size | Domain |
|---|---|---|
| `BIAS_FEATURES` | 9 | Bias |
| `FRAMING_FEATURES` | 10 | Framing |
| `IDEOLOGICAL_FEATURES` | 8 | Ideology |
| `DISCOURSE_FEATURES` | 7 | Discourse |
| `ARGUMENT_FEATURES` | 7 | Argument |
| `EMOTION_FEATURES` | 12 (11 labels + intensity) | Emotion |
| `NARRATIVE_FEATURES` | 9 | Narrative roles |
| `PROPAGANDA_FEATURES` | 11 | Propaganda |
| `CONFLICT_FEATURES` | 9 | Conflict |
| `GRAPH_FEATURES` | 11 | Graph topology |
| `GRAPH_PIPELINE_FEATURES` | 4 | Pipeline-derived graph |
| `LEXICAL_FEATURES` | 5 | Lexical richness |
| `SEMANTIC_FEATURES` | 5 | Embedding stats |
| `SYNTACTIC_FEATURES` | 7 | Syntax / POS |
| `TOKEN_FEATURES` | 6 | Token statistics |

`ALL_FEATURES` = union of all above lists, alphabetically sorted. `FEATURE_SECTIONS` maps section names (`"bias"`, `"emotion"`, etc.) to their sub-lists for `partition_feature_sections()`.

> **Important:** `EMOTION_FEATURES` is constructed by importing `EMOTION_LABELS` from `emotion_schema.py` so this file can never drift from the live 11-label set. The previous inline 20-name list was the root cause of a schema/extractor misalignment bug.

---

### `feature_schema_validator.py` — `FeatureSchemaValidator`

Validates and normalises feature dicts against `ALL_FEATURES`. Responsibilities:
- Drops unknown keys (keys not in schema)
- Imputes missing keys with `fill_value` (default `0.0`) or a per-feature mean fitted on the training set
- Clips all values to `[0.0, 1.0]` (configurable)
- Converts non-finite values to `fill_value`

---

### `feature_statistics.py` — `FeatureStatistics`

Post-extraction analytics. Used inside `FeatureEngineeringPipeline` and the `/features/stats` endpoint.

| Method | Description |
|---|---|
| `dataset_summary(features)` | Count samples, features, mean/min/max variance |
| `compute_variance(features)` | Per-feature variance across the batch |
| `compute_skewness(features)` | Per-feature Fisher-Pearson skewness |
| `detect_constant_features(features)` | Return names where variance ≈ 0 |

---

### `feature_bootstrap.py` — `bootstrap_feature_registry`

Imports exactly 19 feature modules in a deterministic order, triggering the `@register_feature` decorator on each extractor class. Deduplication is handled by `dict.fromkeys`. Returns the final registered list.

**`FEATURE_MODULES` order (abridged):**

```python
FEATURE_MODULES = [
    "src.features.text.lexical_features",
    "src.features.text.semantic_features",
    "src.features.text.syntactic_features",
    "src.features.text.token_features",
    "src.features.bias.bias_features",
    "src.features.bias.framing_features",
    "src.features.bias.ideological_features",
    "src.features.discourse.discourse_features",
    "src.features.discourse.argument_structure_features",
    "src.features.emotion.emotion_features",
    "src.features.emotion.emotion_intensity_features",
    "src.features.emotion.emotion_trajectory_features",
    "src.features.narrative.narrative_features",
    "src.features.propaganda.propaganda_features",
    "src.features.graph.entity_graph_features",
    "src.features.graph.interaction_graph_features",
    "src.features.analysis.analysis_adapter_features",
    ...
]
```

---

### `feature_pruning.py` — `FeaturePruner`

Two-stage column pruning fitted on the training matrix:

1. **Zero-variance filter** — drops any column whose variance across the training set is `< variance_threshold` (default `1e-6`).
2. **Correlation filter** — builds a pairwise Pearson correlation matrix and greedily drops the column from each high-correlation pair (`> correlation_threshold`, default `0.95`), retaining the one with higher variance.

Pruning runs **before** `FeatureStatistics` in `FeatureEngineeringPipeline` (audit fix §1.12) so the O(N²) correlation work is bounded by the post-prune column count.

---

### `feature_report.py` — `FeatureReport`

Generates a structured summary (JSON or human-readable text) covering feature counts per domain, value distributions, zero-rate, and missing-key rate. Optionally saved to disk during training runs.

---

### `dataset_feature_generator.py` — `DatasetFeatureGenerator`

Orchestrates large-scale offline feature generation. Wraps `FeatureEngineeringPipeline` with:
- Thread-pool parallel execution
- Cache-manager integration (`CacheManager.get_or_compute_batch`)
- Progress tracking and CSV/Parquet output

---

### `runtime_config.py` — Thread-safe mutable flags

Module-level registry for flags that may be toggled at runtime (e.g. `transformer_enabled`, `max_transformer_batch`). All reads go through accessor functions:

```python
runtime_config.transformer_enabled()   -> bool
runtime_config.max_transformer_batch() -> int
runtime_config.transformer_chunk_length() -> int
runtime_config.transformer_chunk_stride() -> int
runtime_config.analysis_adapters_strict() -> bool
```

A `configure(**kwargs)` function updates the registry atomically. Prefer `FeatureConfig.apply_to_runtime()` rather than calling `configure()` directly.

---

### `base/base_feature.py` — `BaseFeature` + `FeatureContext`

**`FeatureContext`** — the request object passed to all extractors.

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Raw input text |
| `metadata` | `dict` | Request metadata (tokenizer_id, source URL, …) |
| `tokens` | `List[str]` | Legacy token list (kept for compat; prefer `tokens_word`) |
| `tokens_word` | `List[str]` | Canonical lowercased Unicode word tokens |
| `tokens_bpe` | `List[int]` | Subword token IDs (reserved for future HF-aligned features) |
| `embeddings` | `Any` | Embedding vector for `SemanticFeatures` |
| `cache` | `_BoundedContextCache` | Per-sample LRU cache (max 256 entries, configurable) |
| `shared` | `dict` | Batch-wide shared state |

**`BaseFeature`** — abstract base for all extractors.

| Method | Behaviour |
|---|---|
| `extract(ctx)` | Abstract — override in subclass |
| `extract_batch(contexts)` | Default: `[extract(c) for c in contexts]` — override for vectorised speedup |
| `safe_extract(ctx)` | Validates output, replaces NaN/inf, returns fallback on error |
| `safe_extract_batch(contexts)` | Per-sample validation; length contract with `contexts` enforced |
| `initialize()` | Called once on first `safe_extract`; load heavy resources here |
| `teardown()` | Lifecycle hook for resource cleanup |
| `_fallback()` | Returns `{}` by default; override to return a zero-key dict |

---

### `base/feature_registry.py` — `FeatureRegistry`

Thread-safe class-level registry. Decorated with `@register_feature` decorator.

| Method | Description |
|---|---|
| `register(cls, override=False)` | Register an extractor class; raises on duplicate unless `override=True` |
| `get_feature(name)` | Retrieve class by name |
| `create_feature(name, **kwargs)` | Instantiate by name |
| `list_features()` | Sorted list of all registered names |
| `list_by_group(group)` | Filter by group tag |
| `auto_discover(package)` | Walk and import all modules in a package |
| `freeze()` | Prevent further registrations (used at startup) |
| `clear_registry()` | Wipe registry (test hook only) |

---

### `base/lexicon_loader.py`

Lazy, thread-safe loader for JSON lexicon files in `src/config/lexicons/`.

| Function | Returns |
|---|---|
| `load_lexicon(name)` | Full `{category: [terms]}` dict |
| `load_lexicon_set(name, key)` | Lowercased `set[str]` for one category |
| `load_lexicon_dict(name, key)` | `{term: weight}` dict (list → uniform weight) |

Missing files and missing keys produce a warning and an empty result — never a crash. `_FILE_CACHE` provides one-disk-read-per-file-per-process behaviour.

---

### `base/lexicon_matcher.py`

Vectorised lexicon matching built on NumPy.

| Class / Function | Description |
|---|---|
| `LexiconMatcher` | Stores lexicon as a sorted numpy array; `count_in_tokens(arr)` uses `np.isin` |
| `WeightedLexiconMatcher` | Stores `{term: weight}`; `negation_aware_sum(arr, neg_mask)` applies negation window |
| `to_token_array(tokens)` | Convert `List[str]` to `np.ndarray[str]` once per extractor call |
| `compute_negation_mask(arr, negations, window)` | Boolean mask — tokens within `window` of a negator |

---

### `cache/cache_manager.py` — `CacheManager`

Two-tier cache: in-process LRU (`LRUCache`) + on-disk pickle (`FeatureCache`).

**Cache key** includes:
- `CACHE_VERSION` string (bump to invalidate all disk entries)
- Feature-set fingerprint (SHA-16 of sorted registered feature names — changes if any extractor is toggled)
- Lexicon fingerprint (SHA-16 of source file contents — changes if any lexicon JSON is edited)
- Tokenizer ID from `context.metadata`
- `context.text` and `context.tokens`

**`LRUCache`** enforces both an item-count cap (`max_items`) and a byte-budget cap (`max_bytes`, default 512 MB), evicting the least-recently-used entries first.

**Stats counters** (Prometheus-ready): `mem_hits`, `mem_misses`, `disk_hits`, `disk_misses`, `computes`, `disk_write_failures`.

---

### `emotion/emotion_schema.py` — Canonical Emotion Labels

```python
EMOTION_LABELS = [
    "neutral",
    "admiration", "approval", "gratitude",
    "annoyance", "amusement", "curiosity", "disapproval",
    "love", "optimism", "anger",
]  # 11 labels — EMOTION-11
```

`_LEGACY_EMOTION_LABELS` (9 removed names) is retained for audit/debug only and must never appear in live feature names. `EMOTION_TERMS` maps each live label to a weighted lexicon dict. `validate_schema()` runs on import to auto-stub any missing lexicon entries.

---

### `emotion/emotion_intensity_features.py` — Hybrid Emotion Model

The most computationally significant extractor. Implements a **hybrid lexicon + transformer** scoring strategy:

```
alpha = 0.7 if transformer available else 0.0
score = alpha * transformer_probs + (1 - alpha) * lexicon_probs
```

The transformer (`j-hartmann/emotion-english-distilroberta-base`) is loaded **lazily** on first call with a module-level lock. Long documents use **sliding-window chunking** (chunk=256, stride=64) with per-window softmax averaging. Batches are chunked into `max_batch`-size slices to prevent OOM.

**`extract_batch` optimisation:** one `tokenizer()` call for the full batch, one `model(**inputs)` call per chunk-batch, one `.cpu()` transfer for all windows.

---

### `fusion/feature_fusion.py` — `FeatureFusion`

Aggregates outputs from all registered extractors for a single context or a batch.

- Calls `safe_extract` / `safe_extract_batch` on each extractor in order.
- Appends a `{feature_name}_extracted: 1.0 / 0.0` indicator per extractor so the downstream model can mask zero-fills caused by extractor failures.
- Merges all dicts with `merge_features()`.
- Optionally returns a sorted numpy vector (`return_vector=True`).

---

### `pipelines/feature_engineering_pipeline.py` — 7-Stage Orchestrator

```
Stage 1: pipeline.batch_extract(contexts)         ← FeatureFusion over all extractors
Stage 2: validator.validate_batch(features)        ← schema alignment + imputation
Stage 3: pruner.transform(features)               ← variance + correlation pruning
Stage 4: FeatureStatistics (logging, no mutation)
Stage 5: scaler.transform(features)               ← StandardScaler / MinMax / Robust
Stage 6: selector.transform(features)             ← TopK / variance / correlation
Stage 7: FeatureReport (optional save to disk)
```

All stages are optional (each is `None`-guarded). `fit=True` triggers fitting of stateful stages (pruner, scaler, selector) on the current batch — used during training, not inference.

---

## 5. Feature Definitions Table

### 5.1 Bias Features (`BiasFeaturesV2`, group: `bias`)

| Feature key | Description | Range |
|---|---|---|
| `bias_loaded` | Normalised share of loaded-language tokens | [0, 1] |
| `bias_subjective` | Normalised share of subjective-tone tokens | [0, 1] |
| `bias_uncertainty` | Normalised share of uncertainty hedges | [0, 1] |
| `bias_polarization` | Normalised share of polarising terms | [0, 1] |
| `bias_evaluative` | Normalised share of evaluative adjectives | [0, 1] |
| `bias_intensity` | Mean raw ratio across all 5 bias categories | [0, 1] |
| `bias_diversity` | Normalised entropy of bias distribution | [0, 1] |
| `bias_caps_ratio` | Proportion of uppercase tokens (NER-masked) | [0, 1] |
| `bias_exclamation_density` | `!` count / token count (headline-weighted) | [0, 1] |

Negation-aware: tokens within a 3-token window of `{not, no, never, n't}` are down-weighted by `WeightedLexiconMatcher.negation_aware_sum`.

---

### 5.2 Framing Features (`FramingFeatures`, group: `framing`)

| Feature key | Description | Range |
|---|---|---|
| `frame_economic` | Normalised economic-frame token share | [0, 1] |
| `frame_moral` | Normalised moral-frame token share | [0, 1] |
| `frame_security` | Normalised security-frame token share | [0, 1] |
| `frame_human` | Normalised human-interest frame share | [0, 1] |
| `frame_conflict` | Normalised conflict-frame token share | [0, 1] |
| `frame_phrase_score` | Multi-word frame-phrase hits / token count | [0, 1] |
| `frame_quote_density` | Quote-mark count / token count | [0, 1] |
| `frame_intensity` | Mean raw ratio across all 5 frame categories | [0, 1] |
| `frame_diversity` | Proportion of non-zero frame categories | [0, 1] |
| `frame_entropy` | Normalised entropy of frame distribution | [0, 1] |
| `frame_dominance` | Max normalised frame share | [0, 1] |

---

### 5.3 Ideology Features (`IdeologicalFeatures`, group: `ideology`)

| Feature key | Description | Range |
|---|---|---|
| `ideology_left` | Normalised left-leaning token share | [0, 1] |
| `ideology_right` | Normalised right-leaning token share | [0, 1] |
| `ideology_balance` | `1 - |left - right|` | [0, 1] |
| `ideology_entropy` | Normalised entropy of left/right distribution | [0, 1] |
| `ideology_polarization` | Polarising-term token ratio | [0, 1] |
| `ideology_group_reference` | Group-reference-term token ratio | [0, 1] |
| `ideology_phrase_score` | Multi-word ideology phrase hits / token count | [0, 1] |
| `ideology_intensity` | Mean raw left/right ratio | [0, 1] |
| `ideology_signal_strength` | `0.6*(left+right) + 0.4*polarization` | [0, 1] |

---

### 5.4 Discourse Features (`DiscourseFeatures`, group: `discourse`)

| Feature key | Description | Range |
|---|---|---|
| `discourse_causal_ratio` | Normalised causal connector share | [0, 1] |
| `discourse_contrast_ratio` | Normalised contrast connector share | [0, 1] |
| `discourse_additive_ratio` | Normalised additive connector share | [0, 1] |
| `discourse_sequential_ratio` | Normalised sequential connector share | [0, 1] |
| `discourse_evidential_ratio` | Normalised evidential connector share | [0, 1] |
| `discourse_marker_density` | L2 norm of raw connector counts | [0, 1] |
| `discourse_diversity` | Normalised entropy of connector distribution | [0, 1] |

---

### 5.5 Argument Features (`ArgumentStructureFeatures`, group: `argument`)

| Feature key | Description | Range |
|---|---|---|
| `argument_claim_ratio` | Normalised claim-marker share | [0, 1] |
| `argument_premise_ratio` | Normalised premise-marker share | [0, 1] |
| `argument_evidence_ratio` | Normalised evidence-marker share | [0, 1] |
| `argument_counterargument_ratio` | Normalised counter-arg marker share | [0, 1] |
| `argument_structure_density` | L2 norm of raw argument marker counts | [0, 1] |
| `argument_structure_diversity` | `1 - std(normalised distribution)` | [0, 1] |
| `argument_rhetorical_question_ratio` | `(? count + interrogative hits) / tokens` | [0, 1] |

---

### 5.6 Emotion Features

**`EmotionFeatures`** (group: `emotion`) — per-label lexicon distribution:

| Feature key | Description |
|---|---|
| `emotion_neutral` | Normalised share of neutral-emotion tokens |
| `emotion_admiration` | … |
| `emotion_approval` | … |
| `emotion_gratitude` | … |
| `emotion_annoyance` | … |
| `emotion_amusement` | … |
| `emotion_curiosity` | … |
| `emotion_disapproval` | … |
| `emotion_love` | … |
| `emotion_optimism` | … |
| `emotion_anger` | … |
| `emotion_coverage` | `total_emotion_hits / token_count` |
| `emotion_intensity` | L2 norm of per-label distribution |
| `emotion_entropy` | Normalised distribution entropy |
| `emotion_polarity` | `((pos - neg) + 1) / 2` — remapped to [0, 1] |

**`EmotionIntensityFeatures`** (group: `emotion`) — hybrid model stats:

| Feature key | Description |
|---|---|
| `emotion_intensity_max` | Max score across 11 labels |
| `emotion_intensity_mean` | Mean score |
| `emotion_intensity_std` | Std dev |
| `emotion_intensity_range` | Max − min |
| `emotion_intensity_l2` | L2 norm |
| `emotion_intensity_entropy` | Distribution entropy |
| `emotion_coverage` | Emotion-term hit rate |
| `emotion_transformer_available` | `1.0` if transformer loaded; `0.0` if lexicon-only |

**`EmotionTrajectoryFeatures`** (group: `emotion`) — per-sentence arc:

| Feature key | Description |
|---|---|
| `emotion_traj_mean` | Mean per-sentence intensity |
| `emotion_traj_std` | Std dev of trajectory |
| `emotion_traj_slope` | Linear slope of trajectory (normalised to [0, 1]) |
| `emotion_traj_peak_position` | Position of peak intensity (0=start, 1=end) |
| `emotion_traj_volatility` | Mean absolute delta between adjacent sentences |
| `emotion_traj_range` | Max − min intensity |
| `emotion_traj_shift_mean` | Mean L2 distance between adjacent sentence emotion vectors |
| `emotion_traj_entropy_mean` | Mean within-sentence emotion entropy |
| `emotion_traj_single_sentence` | `1.0` if document was a single sentence (trajectory stats are degenerate) |

---

### 5.7 Narrative Features (`NarrativeFeatures`, group: `narrative`)

| Feature key | Description | Range |
|---|---|---|
| `narrative_hero` | Normalised hero-term share | [0, 1] |
| `narrative_villain` | Normalised villain-term share | [0, 1] |
| `narrative_victim` | Normalised victim-term share | [0, 1] |
| `narrative_conflict` | Normalised conflict-context share | [0, 1] |
| `narrative_resolution` | Normalised resolution-context share | [0, 1] |
| `narrative_crisis` | Normalised crisis-context share | [0, 1] |
| `narrative_polarization` | Normalised polarisation-term share | [0, 1] |
| `narrative_intensity` | L2 norm of context-term vector | [0, 1] |
| `narrative_role_entropy` | Normalised entropy of role distribution | [0, 1] |
| `narrative_context_entropy` | Normalised entropy of context distribution | [0, 1] |
| `narrative_progression` | `resolution / (resolution + conflict + ε)` | [0, 1] |
| `narrative_rhetoric` | `(! + ?) count / token count` | [0, 1] |

---

### 5.8 Propaganda Features (`PropagandaFeatures`, group: `propaganda`)

| Feature key | Description | Range |
|---|---|---|
| `propaganda_name_calling` | Name-calling token share | [0, 1] |
| `propaganda_fear` | Fear-appeal token share | [0, 1] |
| `propaganda_exaggeration` | Exaggeration token share | [0, 1] |
| `propaganda_glitter` | Glittering-generalities share | [0, 1] |
| `propaganda_us_vs_them` | Us-vs-them token share | [0, 1] |
| `propaganda_authority` | Authority-appeal token share | [0, 1] |
| `propaganda_intensifier` | Intensifier token share | [0, 1] |
| `propaganda_intensity` | L2 norm of technique vector | [0, 1] |
| `propaganda_entropy` | Normalised distribution entropy | [0, 1] |
| `propaganda_diversity` | Proportion of non-zero technique slots | [0, 1] |
| `propaganda_rhetoric` | Exclamation + question density (NER-masked, headline-weighted) | [0, 1] |
| `propaganda_caps_ratio` | Uppercase token ratio (NER-masked) | [0, 1] |

---

### 5.9 Graph Features (`EntityGraphFeatures`, group: `graph`)

| Feature key | Description | Range |
|---|---|---|
| `graph_nodes_log` | `log1p(entity_count)` | [0, ∞) raw |
| `graph_edges_log` | `log1p(edge_count)` | [0, ∞) raw |
| `graph_density` | `edges / max_edges` (clamped to [0, 1]) | [0, 1] |
| `graph_sparsity` | `1 - density` | [0, 1] |
| `graph_degree_norm` | `avg_degree / node_count` | [0, 1] |
| `graph_entropy` | Entropy over density/sparsity pair | [0, 1] |
| `graph_intensity` | L2 norm of `[density, degree_norm]` | [0, 1] |

Spacy NER is used when available (seeded from `ctx.cache["spacy_doc"]`). Falls back to `heuristic_entities()` (capitalised token groups) when spaCy is absent.

---

### 5.10 Text Features

**`LexicalFeatures`** (group: `lexical`):

| Feature key | Description | Range |
|---|---|---|
| `lex_vocab_ttr` | Type-token ratio | [0, 1] |
| `lex_vocab_cttr` | Corrected TTR | [0, 1] |
| `lex_hapax_ratio` | Hapax legomena / token count | [0, 1] |
| `lex_dislegomena_ratio` | Twice-occurring types / tokens | [0, 1] |
| `lex_entropy` | Normalised token distribution entropy | [0, 1] |
| `lex_simpson_diversity` | `1 - Σp²` | [0, 1] |
| `lex_yule_k` | `(Σf² - N) / N²` | [0, 1] |
| `lex_avg_word_length` | Mean character length (unbounded) | [0, ∞) |
| `lex_std_word_length` | Std dev of character length (unbounded) | [0, ∞) |

**`SemanticFeatures`** (group: `semantic`):

| Feature key | Description |
|---|---|
| `sem_norm` | L2 norm of embedding vector |
| `sem_mean` | Mean of normalised embedding |
| `sem_std` | Std dev of normalised embedding |
| `sem_entropy` | Entropy of absolute value distribution |
| `sem_anisotropy` | Variance of normalised embedding |
| `sem_sparsity` | Proportion of near-zero dimensions |
| `sem_peakiness` | Max absolute value of normalised embedding |
| `sem_available` | `1.0` if embedding present; `0.0` if extractor fell back to zeros |

**`SyntacticFeatures`** (group: `syntactic`):

| Feature key | Description | Range |
|---|---|---|
| `syn_pos_entropy` | Entropy of NOUN/VERB/ADJ/ADV distribution | [0, 1] |
| `syn_sentence_avg_len` | Mean content-token count per sentence (unbounded) | [0, ∞) |
| `syn_sentence_dispersion` | `std / mean` sentence length | [0, 1] |
| `syn_sentence_entropy` | Entropy of sentence-length distribution | [0, 1] |
| `syn_complexity` | Mean dependency-tree depth (unbounded, O(N) algorithm) | [0, ∞) |
| `syn_coordination` | `conj` dep count / token count | [0, 1] |
| `syn_subordination` | `{ccomp,advcl,relcl}` dep count / tokens | [0, 1] |
| `syn_spacy_available` | `1.0` if spaCy parsed; `0.0` if regex fallback | binary |

---

## 6. Feature Selection Logic

Feature selection is a two-step post-extraction process inside `FeatureEngineeringPipeline`.

### Step 1 — `FeaturePruner` (before scaling)

Removes statistically uninformative columns to reduce the downstream model's input dimensionality.

| Selector | Criterion | Default threshold |
|---|---|---|
| `VarianceThresholdSelector` | Drop columns with training variance `< threshold` | `1e-6` |
| `CorrelationSelector` | Drop one column from each pair with Pearson `|r| > threshold` | `0.95` |

The retained set is determined by fitting on the training split only. The pruner serialises the retained column list for deterministic application at inference time.

### Step 2 — `FeatureSelectionPipeline` (after scaling)

A composable chain of selectors from `feature_reduction.py`:

| Selector | Description |
|---|---|
| `VarianceThresholdSelector` | Can be re-applied post-scaling |
| `CorrelationSelector` | Applied on scaled values |
| `TopKSelector` | Keep the K highest-importance features (importance scores provided externally, e.g. from SHAP) |
| `CompositeSelector` | Chains multiple selectors sequentially |

All selectors follow a `fit(features) → transform(features)` API compatible with `sklearn`.

---

## 7. Data Contracts

### Input contract

Every extractor's `extract(context)` receives a `FeatureContext` with:
- `context.text`: a non-empty Python `str`
- `context.metadata`: optional `dict` with string keys and JSON-serialisable values

`safe_extract` enforces this: passing a non-string `context.text` raises `TypeError` before `extract` is even called.

### Output contract

Every `extract()` must return `dict[str, float]`:
- Keys must be strings matching `re.match(r'^[a-z][a-z0-9_]*$')`
- Values must be finite Python floats (NaN/inf are replaced by `_validate_output`)
- Batch contract: `extract_batch(contexts)` must return a list of the **same length** as `contexts`, with `{}` for failed samples

### Schema contract

Features returned by extractors are validated by `FeatureSchemaValidator` against `ALL_FEATURES`. Any key not in the schema is dropped with a `DEBUG`-level log. Any key in the schema but absent from the extractor output is imputed with `0.0` (or a training-mean if available).

### Cache contract

Cache keys incorporate:
- `CACHE_VERSION` (currently `"v4"`)
- Feature-set fingerprint
- Lexicon fingerprint
- `tokenizer_id` from metadata

Changing any of these automatically invalidates stale disk entries without a manual cache wipe.

---

## 8. Leakage & Bias Prevention

| Risk | Where it was found | Mitigation in place |
|---|---|---|
| **Cross-sample leakage via `ctx.shared`** | `LexicalFeatures` was writing `lex_entropy` to `ctx.shared` (batch-wide) | Fixed: writes to `ctx.cache` (per-sample LRU) |
| **Training-set mean imputed at inference with stale stats** | `FeatureSchemaValidator` fill mean could be from wrong split | Use `fill_value=0.0` unless explicitly providing training means |
| **Emotion one-hot leaking label identity** | `emotion_dominant_<label>` was a one-hot column directly encoding the gold label | Removed (audit task 3) — per-label scalars carry same signal without one-hot encoding |
| **Lexicon fingerprint drift** | Editing `bias.json` silently served stale cache hits | Lexicon fingerprint (SHA-16 of source files) baked into every cache key |
| **Feature-set fingerprint drift** | Enabling/disabling an extractor served stale cache | Feature-set fingerprint (SHA-16 of sorted registry names) baked into every cache key |
| **Scaling constants fitted on test set** | Previous per-extractor `/50.0` divisors implied a corpus assumption | All scaling removed from extractors; `FeatureScalingPipeline` fitted on training split only |
| **Empty-text leakage** | Some extractors returned `{}` (missing keys) while others returned zeros | Standardised: every extractor returns a fixed-key zero dict for empty inputs |
| **NER-masked caps counting proper nouns** | `bias_caps_ratio` counted capitalised proper nouns, biasing NER-heavy articles | `get_text_signals()` excludes spaCy NER span tokens from the caps count |

---

## 9. Config Integration

All runtime behaviour flows through a single path:

```
environment variables
       │
       ▼
FeatureConfig(...)       ← construction reads env vars once
       │
  apply_to_runtime()
       │
       ▼
runtime_config.*         ← consumed by individual extractors at call time
```

**Key integration points:**

| Extractor | Config field consumed |
|---|---|
| `EmotionIntensityFeatures` | `transformer_enabled`, `max_transformer_batch`, `transformer_chunk_length`, `transformer_chunk_stride` |
| `SyntacticFeatures` | `TRUTHLENS_SPACY_MODEL` (via `get_shared_nlp`) |
| `EntityGraphFeatures` | `TRUTHLENS_SPACY_MODEL` (shared Doc via `ensure_spacy_doc`) |
| `_BaseAnalysisFeature` | `analysis_adapters_strict` |
| `CacheManager` | `cache_max_memory_items`, `cache_max_memory_bytes` |
| `FeatureContext.cache` | `TRUTHLENS_CONTEXT_CACHE_SIZE` |

---

## 10. Optimization & Efficiency

### Tokenisation: once per request

`ensure_tokens_word(context)` is called **once at the top of `FeatureFusion.extract`** and stores the result on `context.tokens_word`. All 15+ extractors that need word tokens read this cached list — no extractor re-tokenises the same string.

### Lexicon matching: vectorised `np.isin`

`LexiconMatcher.count_in_tokens(arr)` runs a single `np.isin` against a sorted numpy array of terms. On 20-document batches this is ~15× faster than the previous `Counter + sum(counter.get(w, 0) for w in lexicon)` Python loop.

### spaCy: one parse per request + `nlp.pipe` for batches

`ensure_spacy_doc` / `set_spacy_doc` store the parsed `Doc` in `ctx.cache["spacy_doc"]`. `SyntacticFeatures.extract_batch` calls `nlp.pipe(texts, batch_size=64)` for the batch and seeds the cache for every context, so `EntityGraphFeatures` (and any future graph extractor) inherits the parse without re-parsing.

### Dependency depths: O(N) memoised walk

`_memoized_dependency_depths` caches each token's tree depth by `token.i` and re-uses cached ancestors when ascending. The previous naïve walk was O(N²) per document.

### Emotion transformer: lazy load + sliding-window batch

The transformer model is loaded exactly once (behind a `threading.Lock`) on first call. Long documents use sliding-window chunking with per-window softmax averaging, preventing the O(L) truncation-induced signal loss. CUDA autocast (`float16`) provides a free ~2× speedup on GPU.

### Analysis adapters: bounded flattening

`_numeric_output` caps recursion at `MAX_DEPTH=3`, fanout at `MAX_FANOUT=32`, and total features at `MAX_FEATURES_PER_ADAPTER=64` to prevent unbounded column expansion from debug blobs returned by external analyzers.

### Cache: two-tier, memory + disk, fingerprinted

Memory LRU with both item-count and byte-budget eviction, backed by atomic pickle writes to disk. A single process restart that changes the lexicons or registered feature set will auto-invalidate stale entries without any manual cache clear.

---

## 11. Extensibility Guide

### Adding a new feature extractor

1. **Create the file** in the appropriate subfolder (e.g. `src/features/bias/my_new_feature.py`).

2. **Subclass `BaseFeature`**:
   ```python
   from dataclasses import dataclass
   from src.features.base.base_feature import BaseFeature, FeatureContext
   from src.features.base.feature_registry import register_feature

   @dataclass
   @register_feature
   class MyNewFeature(BaseFeature):
       name: str = "my_new_feature"
       group: str = "bias"         # must match a section name in FEATURE_SECTIONS

       def extract(self, context: FeatureContext) -> dict:
           # ... your logic ...
           return {"my_feature_foo": 0.5, "my_feature_bar": 0.1}
   ```

3. **Add the feature keys** to `feature_schema.py`:
   ```python
   BIAS_FEATURES = [..., "my_feature_foo", "my_feature_bar"]
   ```

4. **Register the module** in `feature_bootstrap.py` `FEATURE_MODULES` list.

5. **Run `assert_schema_consistency()`** at startup to catch name drift early.

### Adding a new emotion label

1. Append the label to `EMOTION_LABELS` in `emotion_schema.py`.
2. Add an entry to `EMOTION_TERMS` with at least one word.
3. Re-run `validate_schema()` to confirm no duplicates.
4. `EMOTION_FEATURES` in `feature_schema.py` will automatically expand (it is constructed from `EMOTION_LABELS`).
5. Bump `CACHE_VERSION` in `feature_cache.py` to invalidate stale on-disk cache entries.

### Adding a new lexicon

1. Create `src/config/lexicons/<name>.json`:
   ```json
   {
     "_doc": "human-readable description",
     "category_a": ["term1", "term2"],
     "category_b": {"weighted_term": 1.5}
   }
   ```
2. In your extractor, load with `load_lexicon_set("name", "category_a")` or `load_lexicon_dict(...)`.
3. The lexicon fingerprint in `CacheManager` will automatically invalidate stale cache entries when the file changes.

### Replacing the scaling strategy

Instantiate `FeatureScalingPipeline` with the desired `method` (`"standard"`, `"minmax"`, `"robust"`) and pass it to `FeatureEngineeringPipeline`. The pipeline will call `scaler.fit(features)` during training and `scaler.transform(features)` at inference.

---

## 12. Common Pitfalls / Risks

| Pitfall | Symptom | Fix |
|---|---|---|
| **Editing `EMOTION_LABELS` without bumping `CACHE_VERSION`** | Old disk cache returns 20-column vectors to an 11-column model head | Bump `CACHE_VERSION` in `feature_cache.py` whenever the schema changes |
| **Writing per-sample data to `ctx.shared`** | Feature values silently overwrite between samples in a batch | Always write derived per-sample values to `ctx.cache`, not `ctx.shared` |
| **Returning `{}` instead of a zero dict for empty input** | `FeatureSchemaValidator` imputes zeros but logs a warning per missing key | Override `_fallback()` and `_empty()` to return fixed-key zero dicts |
| **Adding `@register_feature` twice on the same class** | `ValueError: Feature already registered` at startup | `FeatureRegistry.register` raises on duplicate unless `override=True` |
| **Calling `FeatureConfig.apply_to_runtime()` after extractors are first called** | Transformer loads with wrong chunk size; spaCy loads wrong model | Always apply config before the first `safe_extract` call |
| **Scaling inside an extractor** | Corpus-specific divisors saturate or under-represent on other corpora | Remove divisors; let `FeatureScalingPipeline` handle population normalisation |
| **Disabling an extractor mid-run without clearing the memory LRU** | `_extracted` indicators become inconsistent between cache hits and live runs | `CacheManager` auto-invalidates on feature-set fingerprint change; but if you disable between `CacheManager` instantiations and skip constructing a fresh one, call `cache.clear()` manually |
| **Long articles silently truncated by transformer** | `emotion_intensity_*` features reflect only the first 512 tokens | Sliding-window chunking is already on — ensure `TRUTHLENS_TRANSFORMER_CHUNK=256` and `TRUTHLENS_TRANSFORMER_STRIDE=64` are set |
| **`FeaturePruner` fitted on test set** | Models overfit to test-set statistics; evaluation metrics are optimistic | Always pass `fit=True` only on training split; call `pruner.transform()` with `fit=False` at inference |
| **Analysis adapter returning a nested debug blob** | `_numeric_output` truncates to `MAX_FEATURES_PER_ADAPTER=64` silently | Check `DEBUG` log for "Analyzer flatten truncated" messages; clean up adapter output shape |

---

## 13. Example Usage

### Single-document feature extraction

```python
from src.features.feature_config import FeatureConfig
from src.features.feature_bootstrap import bootstrap_feature_registry
from src.features.base.base_feature import FeatureContext
from src.features.fusion.feature_fusion import FeatureFusion
from src.features.base.feature_registry import FeatureRegistry

# 1. Configure and boot
cfg = FeatureConfig(transformer_enabled=False)  # lexicon-only for speed
cfg.apply_to_runtime()
bootstrap_feature_registry(config=cfg)

# 2. Build the fusion layer from all registered extractors
features_list = [FeatureRegistry.create_feature(name) for name in FeatureRegistry.list_features()]
fusion = FeatureFusion(features=features_list)

# 3. Build the context and extract
ctx = FeatureContext(
    text="Scientists warn that climate change is accelerating beyond predictions.",
    metadata={"source": "reuters", "tokenizer_id": "default"},
)
feature_dict = fusion.extract(ctx)

print(f"Extracted {len(feature_dict)} features")
# → Extracted 97 features (exact count varies with active extractors)
```

### Batch extraction with the full engineering pipeline

```python
from src.features.pipelines.feature_engineering_pipeline import FeatureEngineeringPipeline
from src.features.pipelines.feature_pipeline import FeaturePipeline
from src.features.feature_schema_validator import FeatureSchemaValidator
from src.features.feature_pruning import FeaturePruner
from src.features.fusion.feature_scaling import FeatureScalingPipeline

pipeline = FeatureEngineeringPipeline(
    pipeline=FeaturePipeline(extractors=features_list),
    validator=FeatureSchemaValidator(),
    pruner=FeaturePruner(variance_threshold=1e-6, correlation_threshold=0.95),
    scaler=FeatureScalingPipeline(method="standard"),
)

train_contexts = [FeatureContext(text=t) for t in training_texts]

# Fit on training set
train_features = pipeline.process(train_contexts, labels=train_labels, fit=True)

# Transform at inference (no fit)
test_contexts = [FeatureContext(text=t) for t in test_texts]
test_features = pipeline.process(test_contexts, fit=False)
```

### Using the cache for repeated inference

```python
from src.features.cache.cache_manager import CacheManager
from pathlib import Path

cache_manager = CacheManager(
    base_cache_dir=Path("cache/features"),
    max_memory_items=5_000,
    max_memory_bytes=256 * 1024 * 1024,  # 256 MB
)

def compute_features_cached(ctx: FeatureContext):
    return cache_manager.get_or_compute(
        namespace="production",
        context=ctx,
        compute_fn=lambda c: fusion.extract(c),
    )

# First call: computes and caches
features = compute_features_cached(ctx)

# Second call: returns memory-cached copy in ~1 µs
features = compute_features_cached(ctx)

# View cache stats
print(cache_manager.stats())
# → {'mem_hits': 1, 'mem_misses': 1, 'disk_hits': 0, 'disk_misses': 1, 'computes': 1, ...}
```

### Checking the schema

```python
from src.features.feature_schema import ALL_FEATURES, FEATURE_SECTIONS, get_feature_sections

print(f"Total schema features: {len(ALL_FEATURES)}")
for section, names in get_feature_sections().items():
    print(f"  {section:12s}: {len(names)} features")
```

### Runtime toggle (e.g. disable transformer in tests)

```python
from src.features import runtime_config

# Disable transformer for a test
runtime_config.configure(transformer_enabled=False)

# Re-enable
runtime_config.configure(transformer_enabled=True)
```

### Pruning cache after lexicon update

```python
# After editing src/config/lexicons/bias.json:
# The lexicon fingerprint will change automatically on the next CacheManager instantiation.
# To manually prune stale entries from an existing manager:
results = cache_manager.prune_all(max_age_days=0)  # max_age_days=0 removes everything
print(results)
```
