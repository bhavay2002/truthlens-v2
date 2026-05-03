# `src/data_processing/` — Data Layer Reference

> **Note on path.** The repository's actual data layer lives in `src/data_processing/`, not `src/data/` (the latter contains only stale `__pycache__` artefacts). All paths in this document refer to `src/data_processing/`.

---

## 1. Overview

The `src/data_processing/` package is the **single, contract-governed entry point** between raw on-disk files and PyTorch `DataLoader` objects consumed by the trainer.

**Responsibilities (in pipeline order):**

| # | Stage | Module |
|---|---|---|
| 1 | Path resolution + file existence checks | `data_resolver` |
| 2 | Format-aware file loading (CSV / JSON / JSONL / Parquet) | `data_loader` |
| 3 | Schema validation (per-task, derived from contracts) | `data_validator` |
| 4 | Text + label cleaning | `data_cleaning` |
| 5 | Multi-task + label-health validation | (delegated to `src/analysis/*`) |
| 6 | **Pre-augmentation leakage check** on raw splits | `leakage_checker` |
| 7 | Stratified, leak-aware augmentation (train only) | `data_augmentation` |
| 8 | Atomic Parquet cache write | `data_cache` |
| 9 | Dataset profiling | `data_profiler` + `class_balance` |
| 10 | Pre-tokenized PyTorch `Dataset` build | `dataset` + `dataset_factory` |
| 11 | `DataLoader` build with weighted samplers + correct collate | `dataloader_factory` + `samplers` + `collate` |

**Two design invariants the whole package enforces:**

1. **Contracts are the single source of truth.** Every label/text column name comes from `data_contracts.CONTRACTS`; cleaning, validation, sampling, dataset construction, and augmentation can never disagree about column layouts.
2. **Fail loudly, never silently degrade.** Schema mismatches, NaN labels, out-of-range multilabel values, leakage, and cache corruption all raise — they do not get patched and warned over.

---

## 2. Folder Architecture

```
src/data_processing/
├── __init__.py             → Re-exports the public surface
├── data_contracts.py       → Per-task schema (text col, label cols, type, num_classes)
├── data_resolver.py        → Resolve relative paths against $DATA_DIR; verify existence
├── data_loader.py          → CSV / JSON(L) / Parquet → pandas.DataFrame
├── data_validator.py       → Per-task schema validator (uses CONTRACTS)
├── data_cleaning.py        → Vectorised text normalisation, dedup, length filtering
├── data_augmentation.py    → Task-aware, label-coupled, leak-aware augmentation (train only)
├── leakage_checker.py      → Exact + opt-in near-dup train/val/test overlap detector
├── data_cache.py           → Atomic Parquet cache, content-hashed key, prune helpers
├── data_profiler.py        → Per-split text/duplicate/balance/memory profile
├── class_balance.py        → Inverse-frequency weight computation per task
├── samplers.py             → Contract-driven WeightedRandomSampler (single + multilabel)
├── dataset.py              → BaseTextDataset / ClassificationDataset / MultiLabelDataset
├── dataset_factory.py      → build_dataset(task=…) → correct Dataset subclass via contract
├── collate.py              → pad_token_id-aware collate (RoBERTa pad=1, BERT pad=0)
├── dataloader_factory.py   → DataLoader builder + DataLoaderConfig (env-tuned defaults)
└── data_pipeline.py        → run_data_pipeline(...) — orchestrator that wires all of the above
```

---

## 3. End-to-End Data Flow

```
                      ┌────────────────────────────┐
                      │  data_config (yaml/dict)   │
                      └─────────────┬──────────────┘
                                    ▼
                          data_resolver
                       (resolves $DATA_DIR + checks)
                                    │
                                    ▼
                       ┌──────  CACHE LOOKUP ──────┐
                       │ get_cache_key(            │
                       │   data_config,            │
                       │   file fingerprints,      │
                       │   tokenizer + cleaning +  │
                       │   augmentation + max_len) │
                       └──────────┬────────────────┘
                                  │
                       ┌──────────┴──────────┐
                       ▼                     ▼
                 CACHE HIT               CACHE MISS
                       │                     │
                       │                     ▼
                       │           per task / per split:
                       │              load_dataframe
                       │                     │
                       │                     ▼
                       │             validate_dataframe
                       │                     │
                       │                     ▼
                       │              clean_for_task
                       │                     │
                       │                     ▼
                       │      validate_multitask_dataframe
                       │           + analyze_labels
                       │                     │
                       │                     ▼
                       │      check_leakage_all_tasks  ← on RAW splits
                       │                     │
                       │                     ▼
                       │     augment_dataset (train only,
                       │     stratified + leak-prefiltered)
                       │                     │
                       │                     ▼
                       │      check_leakage_all_tasks  ← defence in depth
                       │                     │
                       │                     ▼
                       │             save_cached_datasets (atomic)
                       │                     │
                       │                     ▼
                       │             profile_dataframe
                       └──────────┬──────────┘
                                  │
                                  ▼
              (optional, if build_dataloaders=True)
                                  │
                                  ▼
                          build_all_datasets
                       (tokenize ONCE up-front;
                        store flat int32 ids array)
                                  │
                                  ▼
                          build_all_dataloaders
                  (sampler if train, pad_token_id-aware collate,
                   pin_memory iff CUDA, persistent_workers)
                                  │
                                  ▼
                Dict[task → Dict[split → DataLoader]]
```

**Critical ordering decisions (and why):**

- **Leakage check runs on RAW splits, before augmentation.** If augmentation runs first, an op that mutates a train row into a near-duplicate of a val/test row would be invisible to the leakage check. The pipeline runs the cheap exact-match check both **before** augmentation (catches input-data leakage) and **after** augmentation (catches anything the per-row pre-filter missed).
- **Per-row pre-filter inside `augment_dataset`.** Each candidate is hashed with the same normalisation as `leakage_checker._normalize` and rejected if it collides with any held-out row. Caller passes `held_out_dfs=[val, test]`.
- **Cache key includes tokenizer + max_length + cleaning + augmentation config.** Changing any of those invalidates stale cache entries — the cache is never silently re-served against new tokenization rules.
- **Cache hit *still* runs leakage check.** Defence against a poisoned cache.

---

## 4. File-by-File Deep Dive

### File: `data_contracts.py`

**Purpose**
The single source of truth for the schema of every task. Cleaning, validation, sampling, dataset construction, augmentation, and the analysis layer all read from this table — no other module hard-codes column names or num_classes.

**Key Definitions**

```python
DEFAULT_MAX_LENGTH: int = 512   # Global default (CFG-D6 — used to be duplicated 4×)

@dataclass(frozen=True)
class DataContract:
    task: str
    task_type: str              # "classification" | "multilabel"
    text_column: str
    label_columns: List[str]
    num_classes: Optional[int]  # classification only
    optional_columns: Optional[List[str]] = None
```

**Registered tasks**

| Task | Type | Labels | num_classes |
|---|---|---|---|
| `bias` | classification | `bias_label` | 2 |
| `ideology` | classification | `ideology_label` | 3 |
| `propaganda` | classification | `propaganda_label` | 2 |
| `frame` | multilabel | `CO`, `EC`, `HI`, `MO`, `RE` | — |
| `narrative` | multilabel | `hero`, `villain`, `victim` | — |
| `emotion` | multilabel | `emotion_0` … `emotion_19` | — |

**Helpers**: `get_contract(task)`, `list_tasks()`, `get_required_columns(task)`, `get_optional_columns(task)`, `is_classification(task)`, `is_multilabel(task)`, `get_num_classes(task)`, `describe_contract(task)`.

**Dependencies**: standard library only.

---

### File: `data_resolver.py`

**Purpose**
Resolve relative paths from a `data_config` dict against an environment-configurable base directory and guarantee every required split file exists before the pipeline does any work.

**Functions**

- `resolve_data_config(config, *, env_var="DATA_DIR", strict=True, required_splits=("train","val","test")) -> Dict[task, Dict[split, Path]]`
  - Reads `$DATA_DIR` once.
  - For every (task, split) in `config`, joins the relative path with `$DATA_DIR`, calls `.resolve()`, and (when `strict=True`) raises `FileNotFoundError` immediately.
  - Raises `ValueError` if any task's config isn't a dict or if a required split is absent.
- `resolve_path(path, *, env_var="DATA_DIR", strict=True) -> Path` — single-path version.

**Edge cases**: empty `$DATA_DIR` falls back to relative-to-CWD resolution; `pretty_print_config` was deliberately removed (was a `print`-based debug helper).

**Dependencies**: stdlib only.

---

### File: `data_loader.py`

**Purpose**
Format-aware loader for raw on-disk files. One entry point, three private loaders, one column-presence guard.

**Functions**

- `load_dataframe(path, *, usecols=None, dtype=None) -> pd.DataFrame`
  Dispatches on file suffix:
  - `.csv`  → `load_csv`
  - `.json`, `.jsonl` → `load_json`
  - `.parquet` → `load_parquet`
  - anything else → `ValueError`.
  Raises `FileNotFoundError` for missing paths.
- `load_csv(path, *, usecols, dtype, encoding="utf-8", na_values)` — `pd.read_csv` with explicit `low_memory=False`. On `UnicodeDecodeError` it logs a warning and retries with `latin-1` (encoding fallback is **logged loudly**, never silent).
- `load_json(path, *, usecols)` — sniffs first non-whitespace byte to decide between JSON-array (`lines=False`) and JSONL (`lines=True`). Fixes `ValueError: Trailing data` that the previous hardcoded `lines=True` produced for `.json` files (CRIT-D4). `usecols` is honoured post-load (pandas has no native column-projection arg for JSON).
- `load_parquet(path, *, usecols)` — forwards `usecols` to `pd.read_parquet(columns=...)` (3-5× memory cut for narrative/emotion frames with large `*_entities` blobs — CRIT-D3).
- `enforce_required_columns(df, required_cols)` — raises `ValueError` listing the missing columns.

**Removed (intentional, see comments)**: `compute_md5`, `load_csv_in_chunks`. The cache layer uses SHA-256 in `data_cache._file_fingerprint`; chunked CSV loading was unused (callers can use `pd.read_csv(chunksize=…)` directly when needed).

**Dependencies**: pandas.

---

### File: `data_validator.py`

**Purpose**
Per-task schema validator with **schemas derived from `CONTRACTS`** at import time. Guarantees the validator can never disagree with the dataset factory.

**Key Types**

```python
@dataclass(frozen=True)
class DataValidatorConfig:
    strict: bool = True
    check_text: bool = True
    min_text_len: int = 3
    max_text_len: int = 10000
    enforce_label_range: bool = True
    enforce_binary_multilabel: bool = True
    sample_errors: int = 5

@dataclass
class ValidationReport:
    rows: int
    columns: int
    missing_columns: List[str]
    invalid_text_rows: int
    invalid_label_rows: Dict[str, int]
    label_value_violations: Dict[str, int]
    notes: Dict[str, Any]
```

**Function**

- `validate_dataframe(df, *, task, config=None) -> ValidationReport` — runs four checks:
  1. **Required columns present** (text col + every label col from contract).
  2. **Text validity**: NaN rejected; length must be in `[min_text_len, max_text_len]`.
  3. **Classification labels**: NaN flagged; range enforced from `_CLASSIFICATION_RANGES` (`bias`/`propaganda` ∈ {0,1}, `ideology` ∈ {0,1,2}); falls back to `(0, num_classes-1)`.
  4. **Multilabel**: NaN flagged; values must be exactly `{0, 1}` when `enforce_binary_multilabel=True`.

  In **strict** mode any violation raises `ValueError`; otherwise it logs a warning. Warnings always log per-task statistics regardless of mode.

**Dependencies**: pandas, `data_contracts`.

---

### File: `data_cleaning.py`

**Purpose**
Reproducible, vectorised text cleaning + duplicate/length filtering + optional label fill.

**Config**

```python
@dataclass
class DataCleaningConfig:
    drop_duplicates: bool = True
    drop_empty_text: bool = True
    normalize_whitespace: bool = True
    lowercase: bool = False           # most NLP tasks keep case
    strip_urls: bool = False          # off by default (URLs can be signal)
    strip_html: bool = True
    min_text_len: int = 3
    max_text_len: int = 20000
    normalize_unicode: bool = False   # NFKC
    remove_emojis: bool = False
    expand_contractions: bool = False
    fill_missing_labels: bool = False
    label_fill_value: int = 0
    log_stats: bool = True
```

**Functions**

- `clean_dataframe(df, *, config, label_cols=None) -> pd.DataFrame` — applies, in this exact order (the `_clean_text` per-string fallback uses the same order so both paths are byte-equivalent):
  1. Capture `nan_mask` **before** `astype(str)` (otherwise `NaN` becomes the literal `"nan"` and survives the empty-text filter).
  2. NFKC unicode normalisation (vectorised via `s.str.normalize`).
  3. HTML strip (`<.*?>`).
  4. URL strip (`https?://\S+|www\.\S+`).
  5. Emoji strip (scoped Unicode ranges — emoticons, pictographs, transport, dingbats, regional indicators, skin tones, ZWJ; deliberately scoped so it doesn't strip math/currency).
  6. Contraction expansion (small ASCII-safe table; `cant→cannot`, `'re→ are`, `gonna→going to`, etc.). Deterministic — no model-driven expansion that would shift cache keys.
  7. Whitespace collapse + `lower()` (optional) + `strip()`.
  8. **Length filter** with **per-bucket logging** — `(NaN)`, `(<min)`, `(>max)` are reported separately (audit §9: silent halving of training corpora).
  9. **Lowercase-aware deduplication** — duplicates dropped on `s.str.lower()` to stay consistent with `leakage_checker._normalize` (avoids false-positive leakage between `"Foo"` and `"foo"` post-clean).
  10. Optional `fillna(label_fill_value)` for label columns.
  11. `reset_index(drop=True)`.
- `clean_for_task(df, task, *, config=None) -> pd.DataFrame` — pulls `label_cols` from `CONTRACTS`. Unknown task → warns and skips label-aware steps (typo defence).

**Performance**: PERF-D1 — every step uses `s.str.replace(..., regex=True)` so the work stays in the C-level pandas/regex engine (~5-10× faster than `s.map(_clean_text)` on 100 k rows).

**Dependencies**: pandas, `re`, `unicodedata`, `data_contracts`.

---

### File: `data_augmentation.py`

**Purpose**
Task-aware, label-coupled, leak-aware augmentation — only applied to the **train** split.

**Config**

```python
@dataclass
class AugmentationConfig:
    multiplier: float = 1.5
    enable_heavy_ops: bool = False    # MLM (RoBERTa fill-mask) + sentence-transformer validation
    similarity_threshold: float = 0.75
    random_seed: int = 42
```

**Operation routing**

| Task | Default ops |
|---|---|
| `bias` | `synonym_replacement`, `random_deletion` |
| `ideology` | `ideology_frame_shift` |
| `propaganda` | `propaganda_injection` |
| `frame` | `random_swap` |
| `narrative` | `narrative_reframe` |
| `emotion` | `emotion_amplify` |

When `enable_heavy_ops=True`, `contextual_replacement` (RoBERTa `<mask>` fill) is appended.

**Label-coupling (CRIT-D6)**: ops that *insert* a domain marker (`propaganda_injection`, `bias_injection`, `emotion_amplify`) **only fire on positive rows**. Otherwise we'd teach the model "`Clearly,` …" is irrelevant to the propaganda label. Determination via `_is_positive(label)`:
- single-label: `int(label) == 1`
- multilabel: `any(label_i == 1)`
- missing: refuse (label-safe).

**Stratified sampling (CRIT-D7)**: `_stratified_weights` builds inverse-frequency weights over per-row label signatures so rare classes are over-sampled. Without this, augmentation on a 95/5 dataset stays 95/5 and `balancing.method: oversample` is a no-op.

**Leak pre-filter (CRIT-D5)**: each candidate's text is hashed with `_leak_key` (matching `leakage_checker._normalize` + `sha256`) and rejected if it collides with any row in `held_out_dfs` (val + test). Up to 5 retries per slot; if all collide the slot is dropped (better to under-augment than poison). Rejected counts are logged.

**Lazy resources**: `_ensure_nltk` (lazy `nltk.download` for `wordnet` + `stopwords` + `omw-1.4`; quiet failures), `_get_mlm` / `_get_embedder` cached singletons. **No module-level `random.seed`, no module-level `nltk.download`** — both were removed (they polluted global RNG / silently network-called at import).

**Per-call RNG**: every call uses a fresh `random.Random(config.random_seed)` so augmentation is reproducible without touching the global RNG.

**Functions**: `augment_text(text, *, task, config, rng, label) -> str`, `augment_dataset(df, *, task, text_column="text", config, held_out_dfs=None) -> pd.DataFrame`.

---

### File: `leakage_checker.py`

**Purpose**
Detect train/val/test text overlap. Exact-match path is the production check; near-duplicate path is opt-in and capped.

**Config / Result**

```python
@dataclass
class LeakageConfig:
    strict: bool = True
    check_near_duplicates: bool = False
    near_duplicate_threshold: float = 0.9
    sample_size: int = 10000
    report_examples: int = 5

@dataclass
class LeakageReport:
    train_val_overlap: int
    train_test_overlap: int
    val_test_overlap: int
    examples: Dict[str, List[str]]
```

**Functions**

- `_normalize(text) -> str` — `str(text).strip().lower()`; matches the hash basis used by `data_augmentation._leak_key` and `data_cleaning` dedup.
- `_hashes(series) -> set` — SHA-256 over normalised text; **empty strings are filtered before hashing** (otherwise they collapse into one bucket and report bogus overlap).
- `check_leakage_splits(train, val, test, *, config) -> LeakageReport` — set intersection over `_hashes(...)`; in strict mode any non-zero overlap raises `RuntimeError`.
- `check_leakage_all_tasks(datasets, *, config)` — runs above per task; skips with a warning if a split is missing.
- `check_near_duplicates(df1, df2, threshold=0.9, *, max_pairs=10_000_000, random_state=0) -> int` — `difflib.SequenceMatcher` over text pairs. **Hard cap** at 10 M pairs (≈30 s); exceeds the cap → even subsample to `√max_pairs` per side and warns loudly. Recommended replacement for large splits: MinHash + LSH.

---

### File: `data_cache.py`

**Purpose**
Atomic Parquet cache for cleaned/augmented dataframes, keyed on a content-aware hash that auto-invalidates on tokenizer / cleaning / augmentation / source-file changes.

**Cache version**

```python
_BASE_VERSION = "v4"
CACHE_VERSION = f"{_BASE_VERSION}-{md5(getsource(_file_fingerprint) + getsource(get_cache_key))[:8]}"
```

Editing either function automatically bumps `CACHE_VERSION` so old caches are invalidated without a manual coordination PR (CACHE-D2).

**Key Functions**

- `_file_fingerprint(path) -> {size, sha}` (CRIT-D2):
  - Files `≤ 2 MB` → SHA-256 of full content.
  - Larger files → SHA-256 of first 1 MB + last 1 MB (head+tail). Closes the blind spot where 1 MB < size ≤ 2 MB files with identical heads but different tails would collide.
- `get_cache_key(data_config, file_paths, *, extra=None) -> str` — SHA-256 of:
  ```python
  {"config": data_config, "files": files_sha, "version": CACHE_VERSION, "extra": extra}
  ```
  `data_pipeline` populates `extra` with `{tokenizer, max_length, cleaning, augmentation, validation flags}`.
- `save_cached_datasets(datasets, cache_key)` (CACHE-D5):
  1. Stage every `parquet` + `meta.json` in `{cache_key}.tmp/`.
  2. `f.flush()` + `os.fsync()` on `meta.json` (fsync errors logged on tmpfs).
  3. `os.replace(tmp, final)` — atomic dir swap. If `final` already exists it's `rmtree`'d first.
- `load_cached_datasets(cache_key)`:
  - `meta.json` is the **commit marker**. If missing, the directory is from a crashed save and is invalidated (returns `None`).
  - Per-file Parquet read errors → invalidate the whole cache entry.
- `prune_cache(*, max_bytes=None, max_age_days=None) -> {scanned, removed_age, removed_bytes, bytes_freed}` (CACHE-D4) — LRU eviction by mtime when over the byte cap; age-based eviction. Skips in-flight `*.tmp` dirs. OS errors are logged, never raised.

**Lazy settings**: `_get_cache_dir` reads `src.config.settings_loader.load_settings` only on first call so importing `data_cache` doesn't require the data CSVs to exist.

---

### File: `data_profiler.py`

**Purpose**
Cheap per-task stats: text length, duplicates, empty rows, class balance, memory footprint.

**Result**

```python
@dataclass
class DataProfile:
    rows: int
    columns: int
    avg_text_len: Optional[float]
    min_text_len: Optional[int]
    max_text_len: Optional[int]
    duplicate_rows: Optional[int]
    empty_text_rows: Optional[int]
    class_balance: Optional[Dict[str, Any]]   # delegates to class_balance.analyze_task_balance
    memory_usage_mb: Optional[float]
```

**Functions**

- `profile_dataframe(df, *, task=None, config=None) -> DataProfile` — optionally subsamples to `config.sample_size`; computes text-length stats over `df["text"].astype(str).str.len()`; `df.memory_usage(deep=True).sum() / MiB`.
- `profile_all_datasets(datasets, *, split="train", config=None) -> Dict[task, DataProfile]`.
- `profile_to_dict(profile)` — dataclass → dict for serialisation.

---

### File: `class_balance.py`

**Purpose**
Per-task class-balance analysis + class-weight computation (driven by contracts).

**Functions**

- `analyze_classification(df, label_col, *, config) -> ClassBalanceReport` — `value_counts → distribution`; flags `imbalance_detected` when `min(distribution) < imbalance_threshold` (default 0.2); optionally returns inverse-frequency weights (normalised to sum to 1).
- `analyze_multilabel(df, label_cols, *, config) -> ClassBalanceReport` — per-column positive rate; per-column `neg/pos` weight (BCE-pos-weight friendly).
- `analyze_task_balance(df, task)` — contract-driven dispatch.

**Used by**: `data_profiler` (reads `report.distribution` and `report.imbalance_detected`).

---

### File: `samplers.py`

**Purpose**
Build per-task `WeightedRandomSampler` for train DataLoaders.

**Functions**

- `build_classification_sampler(labels, *, normalize=True)` — inverse-frequency weights (`total / class_counts`, optionally normalised). `np.maximum(class_counts, 1)` avoids div-by-zero on absent classes. Returns `WeightedRandomSampler(weights, num_samples=n, replacement=True)`.
- `build_multilabel_sampler(label_matrix, *, epsilon=1.0)` — Laplace-smoothed per-column positive counts (`epsilon=1.0` prevents the 1e6 weight blowup that the previous `1e-6` produced for zero-positive columns). Per-row weight = `Σ (label * label_weight)`; rows with zero positives get the mean of `label_weights` as fallback.
- `build_sampler(*, task, df, use_weighted=True)` — contract-driven dispatch; raises `KeyError` listing missing label cols.

---

### File: `dataset.py`

**Purpose**
Three pre-tokenized PyTorch `Dataset` classes. **Tokenization happens once in `__init__`** — no per-sample `tokenizer(...)` calls.

**Classes**

- `BaseTextDataset(Dataset)` — common pre-tokenization layer.
  - `__init__(df, tokenizer, *, text_col="text", max_length=512, return_offsets_mapping=False, log_truncation=True)`.
  - Calls the tokenizer **once** with `truncation=True, padding=False, return_attention_mask=True, return_length=True` (and `return_offsets_mapping=…`).
  - `return_offsets_mapping=True` requires a fast tokenizer (`is_fast=True`); raises `ValueError` if not.
  - **Flat storage layout (PERF-D2)**: instead of a list of Python lists (∼25 M Python ints / GC pressure / fork-copy storm on 100 k × 256), stores:
    - `_ids_flat: int32` array (1-D)
    - `_attn_flat: int8` array (1-D)
    - `_offsets: int64` cumulative-sum array — `offsets[i]:offsets[i+1]` is the slice for row i.
    - Optional `_om_flat: int64[total, 2]` for offset_mapping.
    - ~3-5× lower RSS, ~30 % faster `__getitem__`, shared by reference across DataLoader worker forks.
  - **Truncation diagnostics**: prefers HuggingFace `encodings[i].overflowing` (canonical signal); falls back to `length >= max_length` heuristic for slow tokenizers (TOK-D2 — old heuristic over-counted samples that fit exactly).
  - `_encoded_inputs(idx)` — slices the flat arrays, casts to `int64`, wraps in `torch.from_numpy`. The `.astype(int64, copy=True) → from_numpy` chain yields one fresh tensor with no double-copy.

- `ClassificationDataset(BaseTextDataset)`
  - Extra args: `label_col`, `num_classes`, `task_name`.
  - Vectorises labels once into `int64` numpy array.
  - Raises on NaN labels (no silent fill).
  - `__getitem__ → {input_ids, attention_mask, [offset_mapping], labels: long, task: str}`.

- `MultiLabelDataset(BaseTextDataset)`
  - Extra args: `label_cols`, `task_name`.
  - Vectorises labels once into `float32` matrix `(N, len(label_cols))`.
  - **Soft labels in [0, 1] are accepted** (BCE-with-logits handles them); values outside `[0, 1]` raise `ValueError` reporting how many entries are bad. Removes the inconsistency where `data_validator` rejected fractional values but `MultiLabelDataset` quietly accepted them (audit §9).
  - `__getitem__ → {input_ids, attention_mask, [offset_mapping], labels: float[L], task: str}`.

---

### File: `dataset_factory.py`

**Purpose**
Single contract-driven entry point for building any `Dataset`. Callers pass a task name; the factory resolves the contract and instantiates the right subclass with the right columns.

**Config**

```python
@dataclass(frozen=True)
class DatasetBuildConfig:
    max_length: int = DEFAULT_MAX_LENGTH
    return_offsets_mapping: bool = False
    log_truncation: bool = True
```

**Functions**

- `build_dataset(*, task, df, tokenizer, max_length=512, return_offsets_mapping=False, log_truncation=True, config=None)` — when `config` is given, its fields take precedence over the loose kwargs (UNUSED-D1 — loose kwargs kept for back-compat). Routes to `ClassificationDataset` or `MultiLabelDataset` based on `contract.task_type`.
- `build_all_datasets(*, datasets, tokenizer, …) -> Dict[task, Dict[split, Dataset]]` — bulk build over all tasks/splits.
- `validate_dataset_compatibility(task, df)` — raises `ValueError` listing every contract-required column missing from `df`.

---

### File: `collate.py`

**Purpose**
Pad-aware collate function. **`pad_token_id` is not hardcoded** — RoBERTa uses 1, BERT uses 0.

**Functions**

- `_collate(batch, *, pad_token_id, safety_check=True) -> Dict[str, Tensor]` — pads `input_ids` with `pad_token_id`, `attention_mask` with `0`, optional `offset_mapping` with `0`, stacks `labels`, propagates `task` (and asserts all rows share the same task when `safety_check=True`).
- `build_collate_fn(*, pad_token_id, safety_check=True)` — `partial(_collate, …)` closure used by `dataloader_factory`.
- `collate_fn(batch)` / `fast_collate_fn(batch)` — **deprecated** legacy helpers (`pad_token_id=0`). They fire a one-shot `DeprecationWarning` per process because RoBERTa-family models silently train on garbage padding when wired to these (UNUSED-D4).

---

### File: `dataloader_factory.py`

**Purpose**
Build PyTorch `DataLoader`s with sensible, env-aware defaults; bind the correct collate `pad_token_id`.

**Config**

```python
@dataclass
class DataLoaderConfig:
    batch_size: int = 16
    num_workers: int = -1                  # -1 → auto = min(8, os.cpu_count())
    pin_memory: bool = True                # gated on torch.cuda.is_available() at build time
    use_sampler: bool = True               # mutually exclusive with shuffle on train
    drop_last: bool = False
    persistent_workers: bool = True
    prefetch_factor: int = 4
    safety_check_collate: bool = True
    shuffle: bool = True                   # honoured only when no sampler is in play

    @classmethod
    def from_yaml_data(cls, data_section): ...   # accepts dict OR DataConfig dataclass; warns on unknown keys
    def resolved_num_workers(self) -> int: ...
    def resolved_pin_memory(self) -> bool: ...
```

`_default_num_workers()` was bumped from `min(4, cpu//2)` → `min(8, cpu)` so 12-/16-core boxes don't bottleneck on the data path (GPU-D5).

**Functions**

- `build_dataloader(*, task, dataset, df, split, config, tokenizer=None) -> DataLoader`:
  - Train + `use_sampler` → `samplers.build_sampler(task, df)`; otherwise honour `config.shuffle` (samplers and shuffle are mutually exclusive at DataLoader's API level).
  - `pad_id = tokenizer.pad_token_id or dataset.pad_token_id or 0` — feeds `build_collate_fn(pad_token_id=pad_id)`.
  - `pin_memory` only when CUDA is available; `persistent_workers` + `prefetch_factor` only when `num_workers > 0`.
  - `drop_last` only on train.
  - Logs `task | split | size | workers | pin | pad_id`.
- `build_all_dataloaders(*, datasets, raw_dfs, config, tokenizer)` — bulk build over the dict-of-dicts.

---

### File: `data_pipeline.py`

**Purpose**
The orchestrator. Wires every other module into one `run_data_pipeline(...)` call.

**Config**

```python
class DataPipelineConfig:
    enable_cleaning: bool = True
    enable_validation: bool = True
    enable_augmentation: bool = False
    enable_profiling: bool = True
    enable_leakage_check: bool = True
    enable_multitask_validation: bool = True
    enable_label_analysis: bool = True
    enable_cache: bool = True
    force_rebuild: bool = False
    max_length: int = DEFAULT_MAX_LENGTH
    return_offsets_mapping: bool = False
    cleaning_config: Optional[DataCleaningConfig] = None
    augmentation_config: Optional[AugmentationConfig] = None
    dataloader_config: Optional[DataLoaderConfig] = None
```

**Entry point**

```python
def run_data_pipeline(
    *,
    data_config: Dict[str, Dict[str, str]],
    tokenizer=None,
    build_dataloaders: bool = False,
    config: Optional[DataPipelineConfig] = None,
):
```

**Algorithm**:
1. `resolve_data_config(data_config)`.
2. `cache_key = get_cache_key(data_config, resolved_paths, extra=_cache_extra(config, tokenizer))` — `_cache_extra` pulls in `max_length`, cleaning config, augmentation config, validation flags, and tokenizer fingerprint (`name_or_path`, `vocab_size`, class).
3. **Cache hit** → load + run leakage check (poisoned-cache defence).
4. **Cache miss** → `_build_raw_datasets`:
   - For each (task, split): `load_dataframe` → `validate_dataframe` → `clean_for_task` → `validate_multitask_dataframe` + `analyze_labels` (delegated to `src/analysis`).
   - Leakage check on raw splits.
   - Augmentation on train (with `held_out_dfs=[val, test]` for the per-row leak pre-filter).
   - Re-run leakage check (defence in depth).
   - `save_cached_datasets`.
   - `profile_dataframe` per task.
5. If `build_dataloaders=False` → return `Dict[task, Dict[split, DataFrame]]`.
6. Else require `tokenizer`, `build_all_datasets(...)`, `build_all_dataloaders(...)`, return `Dict[task, Dict[split, DataLoader]]`.

`_to_plain` flattens dataclass / `__dict__` configs into JSON-stable dicts so the cache-key extra is deterministic.

**Cross-package dependencies**: `src.analysis.label_analysis`, `src.analysis.multitask_validator`, `src.config.settings_loader` (via `data_cache`).

---

## 5. Data Contracts & Schemas

### Input formats

| Suffix | Reader | Notes |
|---|---|---|
| `.csv` | `pd.read_csv(..., low_memory=False)` | UTF-8 with `latin-1` fallback (logged) |
| `.json` | `pd.read_json(..., lines=False)` | Sniffed if first non-WS byte is `[` |
| `.jsonl` | `pd.read_json(..., lines=True)` | `usecols` honoured post-load |
| `.parquet` | `pd.read_parquet(columns=…)` | Native column projection |

### Required columns per task

Every input dataframe **must** contain `text` (UTF-8, length ≥ 3 chars after cleaning) plus the label columns from the contract:

| Task | Label columns | Allowed values |
|---|---|---|
| `bias` | `bias_label` | `{0, 1}` |
| `ideology` | `ideology_label` | `{0, 1, 2}` |
| `propaganda` | `propaganda_label` | `{0, 1}` |
| `frame` | `CO`, `EC`, `HI`, `MO`, `RE` | `{0, 1}` (or `[0, 1]` soft) |
| `narrative` | `hero`, `villain`, `victim` | `{0, 1}` (or `[0, 1]` soft) |
| `emotion` | `emotion_0` … `emotion_19` | `{0, 1}` (or `[0, 1]` soft) |

**Soft multilabel values** in `[0, 1]` are accepted by `MultiLabelDataset` (BCE-with-logits supports them); anything outside that range raises.

### Output of the pipeline

- `build_dataloaders=False` → `Dict[task, Dict[split, pd.DataFrame]]` (cleaned + validated + optionally augmented).
- `build_dataloaders=True` → `Dict[task, Dict[split, torch.utils.data.DataLoader]]`.

Each batch yielded by a DataLoader is a dict:

```python
{
    "input_ids":      LongTensor[B, T],
    "attention_mask": LongTensor[B, T],
    "offset_mapping": LongTensor[B, T, 2],   # only if return_offsets_mapping=True
    "labels":         LongTensor[B] | FloatTensor[B, L],
    "task":           str,                   # same value for every sample (enforced by collate)
}
```

---

## 6. Config Integration

**Environment variables**

| Var | Read by | Effect |
|---|---|---|
| `DATA_DIR` | `data_resolver.resolve_data_config` | Base directory prepended to all per-task split paths. |

**`config.yaml::data` block** is consumed by `DataLoaderConfig.from_yaml_data(...)`:

| YAML key | DataLoaderConfig field | Default |
|---|---|---|
| `batch_size` | `batch_size` | 16 |
| `num_workers` | `num_workers` (`-1` = auto) | -1 |
| `pin_memory` | `pin_memory` | `True` (gated on CUDA) |
| `use_sampler` | `use_sampler` | `True` |
| `drop_last` | `drop_last` | `False` |
| `persistent_workers` | `persistent_workers` | `True` |
| `prefetch_factor` | `prefetch_factor` | 4 |
| `safety_check_collate` | `safety_check_collate` | `True` |
| `shuffle` | `shuffle` (only when no sampler) | `True` |

Unknown YAML keys under `data:` are dropped with a warning, so a stale block doesn't crash the run (CFG-D1).

**Cache directory** comes from `src.config.settings_loader.load_settings().paths.cache_dir / "data"` (resolved lazily on first call).

---

## 7. Error Handling & Validation

| Concern | Where | Behaviour |
|---|---|---|
| Missing input file | `data_resolver` | `FileNotFoundError` (strict mode default) |
| Missing required split | `data_resolver` | `ValueError` listing the missing split |
| Unsupported file format | `data_loader.load_dataframe` | `ValueError(f"Unsupported file format: {suffix}")` |
| Encoding decode failure | `data_loader.load_csv` | Warn + retry with `latin-1` |
| Missing required column | `data_validator` (strict) / `dataset_factory.validate_dataset_compatibility` / `samplers.build_sampler` | `ValueError` / `KeyError` listing the missing columns |
| NaN labels (single) | `ClassificationDataset.__init__` | `ValueError` |
| NaN labels (multi) | `MultiLabelDataset.__init__` | `ValueError` |
| Out-of-range multilabel | `MultiLabelDataset.__init__` | `ValueError` reporting bad-entry count |
| Out-of-range classification label | `data_validator` (strict) | `ValueError` |
| Train/val/test leakage | `leakage_checker` (strict) | `RuntimeError` with per-pair counts |
| Augmentation collides with held-out | `data_augmentation.augment_dataset` | Reject + retry up to 5×; log rejection count; drop slot if all attempts collide |
| Cache `meta.json` missing | `data_cache.load_cached_datasets` | Treat as crashed save → `None` (rebuild) |
| Cache parquet corruption | `data_cache.load_cached_datasets` | Warn + invalidate whole entry |
| Mixed-task batch | `collate._collate` | `RuntimeError("Mixed-task batch detected")` |
| Empty batch | `collate._collate` | `ValueError("Empty batch")` |
| `return_offsets_mapping=True` on slow tokenizer | `BaseTextDataset.__init__` | `ValueError` |

**Logging**: every module uses `logger = logging.getLogger(__name__)`. Stage-level `INFO` lines (rows in/out, frames cached, samplers built) and warnings on any silent-degradation surface (encoding fallback, fsync skip, near-dup subsample, unknown YAML keys).

---

## 8. Optimization & Efficiency

| Module | Optimization |
|---|---|
| `data_loader` | Native `usecols` / `columns=` projection on Parquet (3-5× memory cut on entity-blob frames). |
| `data_cleaning` | Vectorised `pandas.Series.str.replace(..., regex=True)` chain (PERF-D1: ~5-10× faster than per-row `.map`). |
| `data_cache` | SHA-256 head+tail for files > 2 MB (bounded I/O); atomic dir replace (no partial reads); LRU prune to keep disk bounded. |
| `dataset` | One-shot tokenization + flat `int32`/`int8` arrays (PERF-D2: ~3-5× lower RSS, fast slicing, fork-friendly). |
| `dataloader_factory` | `pin_memory` gated on `torch.cuda.is_available()`; `persistent_workers` + `prefetch_factor` reduce worker respawn cost; `num_workers` defaults to `min(8, cpu)`. |
| `leakage_checker` | Hash-set intersection for the production path; `SequenceMatcher` near-dup capped at 10 M pairs (subsample + warn above cap). |
| `data_augmentation` | Lazy `nltk.download` + lazy heavy-model singletons; per-call RNG (no global mutation). |

**No GPU usage** in this package — all tensor allocation happens on the CPU side. The trainer moves batches to device.

---

## 9. Extensibility Guide

### Adding a new task
1. **Register a contract** in `data_contracts.CONTRACTS` (text col, label cols, type, `num_classes`).
2. **Add a classification range** in `data_validator._CLASSIFICATION_RANGES` if the task is classification and not in the default `(0, num_classes-1)` band.
3. **Map a default augmentation op** in `data_augmentation.TASK_OPS` (label-coupled ops should call `_is_positive(label)` first if they insert markers).
4. Done — `clean_for_task`, `validate_dataframe`, `build_dataset`, `build_sampler`, and `build_all_dataloaders` are all contract-driven and require **no** changes.

### Adding a new file format
Add a private `load_<fmt>` function and a suffix branch to `data_loader.load_dataframe`. Keep `usecols` semantics identical.

### Adding a new cleaning step
1. Add a flag to `DataCleaningConfig`.
2. Add a vectorised branch to `clean_dataframe` (and the matching scalar branch to `_clean_text` so both code paths produce identical output).
3. **Ordering matters** — keep the `_cache_extra` invariant by inserting it in both code paths in the same position.
4. Mention the change in `_cache_extra` indirectly via the dataclass field — `_to_plain(config.cleaning_config)` will pick it up automatically and invalidate stale caches.

### Adding a new augmentation op
1. Implement `def my_op(text, rng, *, label=None) -> str`.
2. If it inserts a domain marker, gate on `_is_positive(label)` (CRIT-D6 — never inject a positive marker into a negative row).
3. Register in `TASK_OPS[task]`.

### Bumping cache invalidation manually
Increment `_BASE_VERSION` in `data_cache`. Source-level changes to `_file_fingerprint` / `get_cache_key` already auto-invalidate via the logic fingerprint.

---

## 10. Common Pitfalls / Risks

| Pitfall | Mitigation in current code |
|---|---|
| Augmentation creating near-duplicates of val/test | Per-row pre-filter in `augment_dataset` + post-augmentation leakage re-check. |
| Augmentation inserting positive markers on negative rows | `_is_positive(label)` gate on `propaganda_injection`, `bias_injection`, `emotion_amplify`. |
| `NaN` text becoming the literal `"nan"` and surviving filters | `nan_mask` captured **before** `astype(str)` in `clean_dataframe`. |
| Case-sensitive dedup conflicting with case-insensitive leakage check | `clean_dataframe` dedups on `s.str.lower()` to match `leakage_checker._normalize`. |
| Crashed cache save returning a partial dataset | Atomic dir replace + `meta.json` commit marker. |
| Tokenizer mismatch reusing a stale cache | Cache key includes tokenizer name + vocab size + class. |
| Wrong pad token id silently corrupting RoBERTa training | `build_collate_fn(pad_token_id=tokenizer.pad_token_id)`; legacy `pad_token_id=0` helpers warn loudly. |
| Soft multilabel values bypassing the validator | `MultiLabelDataset` itself rejects values outside `[0, 1]` regardless of validator strictness. |
| Pure-ratio multilabel sampler weights blowing up on zero-positive columns | `epsilon=1.0` Laplace smoothing + fallback weight for zero-positive rows. |
| Near-dup leakage check melting on 10k × 10k splits | Hard 10 M-pair cap with even subsample + loud warning. |
| `nltk.download` at import polluting CI | Lazy `_ensure_nltk` on first call. |
| Global RNG mutation from augmentation | Per-call `random.Random(seed)`. |
| JSON-array file with `.json` suffix raising `Trailing data` | First-byte sniff in `load_json`. |

---

## 11. Example Usage

### Full pipeline → DataLoaders

```python
from transformers import AutoTokenizer
from src.data_processing import (
    run_data_pipeline,
    DataPipelineConfig,
    DataLoaderConfig,
)
from src.data_processing.data_cleaning import DataCleaningConfig
from src.data_processing.data_augmentation import AugmentationConfig

data_config = {
    "bias": {
        "train": "bias/train.csv",
        "val":   "bias/val.csv",
        "test":  "bias/test.csv",
    },
    "emotion": {
        "train": "emotion/train.parquet",
        "val":   "emotion/val.parquet",
        "test":  "emotion/test.parquet",
    },
}

tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)

config = DataPipelineConfig(
    enable_augmentation=True,
    augmentation_config=AugmentationConfig(multiplier=2.0, random_seed=42),
    cleaning_config=DataCleaningConfig(strip_html=True, normalize_unicode=True),
    dataloader_config=DataLoaderConfig(batch_size=32, num_workers=-1),
    max_length=256,
)

loaders = run_data_pipeline(
    data_config=data_config,
    tokenizer=tokenizer,
    build_dataloaders=True,
    config=config,
)

for task, splits in loaders.items():
    train_loader = splits["train"]
    for batch in train_loader:
        # batch keys: input_ids, attention_mask, labels, task
        ...
        break
```

### Just the cleaned dataframes (no tokenization)

```python
dfs = run_data_pipeline(data_config=data_config, build_dataloaders=False)
print(dfs["bias"]["train"].head())
```

### Single-task dataset construction (programmatic)

```python
from src.data_processing import build_dataset, DatasetBuildConfig

ds = build_dataset(
    task="emotion",
    df=dfs["emotion"]["train"],
    tokenizer=tokenizer,
    config=DatasetBuildConfig(max_length=256, return_offsets_mapping=True),
)
sample = ds[0]
# {"input_ids": LongTensor, "attention_mask": LongTensor,
#  "offset_mapping": LongTensor, "labels": FloatTensor[20], "task": "emotion"}
```

### Cache hygiene (e.g. at process startup)

```python
from src.data_processing.data_cache import prune_cache

stats = prune_cache(max_bytes=10 * 1024**3, max_age_days=14)
# {"scanned": 17, "removed_age": 3, "removed_bytes": 2, "bytes_freed": 5_241_098_240}
```
