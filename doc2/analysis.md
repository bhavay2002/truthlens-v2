# `src/analysis/` — Feature Analysis Layer

This document covers `src/analysis/`, the feature-engineering layer that sits
between raw text (or the `src/data_processing/` cleaned corpus) and the
multi-task transformer heads. It is the only layer that produces TruthLens'
hand-crafted features (~150 numeric signals across 14 analyzer groups plus the
propaganda pattern detector and bias profile).

Every claim here was read out of the code in this directory at the time of
writing. Audit tags (`CRIT-A*`, `PERF-A*`, `NUM-A*`, `F*`, "Section N") are
preserved verbatim where the source uses them, so future audits can grep this
file and the code with the same vocabulary.

---

## 1. Overview

The analysis layer turns one piece of text into one **`FullAnalysisOutput`** —
a typed pydantic record whose 15 nested sections collectively expose:

* 14 deterministic, lexicon + spaCy-driven analyzer outputs (rhetoric, argument,
  context omission, discourse coherence, emotion targets, framing, information
  density, information omission, ideology, narrative roles, narrative conflict,
  narrative propagation, narrative temporal, source attribution).
* 1 second-order **propaganda pattern** vector that *consumes* the upstream
  features and re-projects them through pattern templates (manipulation,
  polarization, etc.).
* A **bias profile** (weighted, normalized aggregation of the 14 sections plus
  the propaganda block) that the API surfaces as the headline summary.

The layer is structured around three independent guarantees:

1. **Determinism** — for any text the same vector comes out, byte for byte.
   Section ordering is locked (`feature_keys.py`), value ranges are clipped to
   `[0, 1]` (`output_models.FeatureModel`), and entropy/normalization helpers
   guard against zero-mass divisions (`safe_normalized_entropy`, `_safe`).
2. **Single-pass NLP** — every analyzer that needs spaCy reads from one cached
   `Doc` per text via `FeatureContext` + `spacy_loader.get_doc(...)`. Tokens,
   counts, lower-cased text and punctuation counts are each computed once per
   context (`PERF-A1/A2`).
3. **Schema-bound I/O** — analyzers only emit keys that exist in
   `feature_keys.py`. The pipeline validates analyzer outputs against the
   schema (`feature_schema.validate_schema_integrity`), and `make_vector`
   converts a feature dict to a fixed-order numpy array. Anything that drifts
   crashes loudly at startup, never at inference time.

The pipeline is wired from a registry (`build_default_registry()` in
`analysis_registry.py`), so the orchestrator does not hard-code analyzer
imports — it iterates the registry and calls each analyzer through
`BaseAnalyzer.__call__`.

---

## 2. Folder architecture

```
src/analysis/
├── __init__.py                      # empty (module is import-by-path)
│
├── analysis_pipeline.py             # AnalysisPipeline: registry-driven runner
├── analysis_registry.py             # AnalyzerRegistry + build_default_registry()
├── analysis_config.py               # AnalysisConfig dataclass (toggles, thresholds)
├── orchestrator.py                  # AnalysisOrchestrator: pipeline + bias + patterns
├── batch_processor.py               # BatchProcessor: many-doc wrapper, no cross-cache
├── integration_runner.py            # End-to-end debug harness (CLI/REPL)
│
├── base_analyzer.py                 # BaseAnalyzer ABC + __call__ wrapper
├── feature_context.py               # FeatureContext: lazy spaCy + shared cache
│
├── feature_keys.py                  # 15 ordered key tuples (single source of truth)
├── feature_schema.py                # Lockable registry, make_vector, integrity check
├── feature_merger.py                # FeatureMerger: dict[name → features] → models
├── output_models.py                 # pydantic FeatureModel + FullAnalysisOutput
│
├── spacy_config.py                  # spaCy model name / device / disable maps
├── spacy_loader.py                  # Thread-safe multi-model cache, get_doc()
├── _text_features.py                # Shared text utilities (term_ratio, entropy,…)
├── preprocessing.py                 # Text normalization + segmentation
│
├── emotion_lexicon.py               # Plutchik/NRC-derived emotion taxonomy
│
├── rhetorical_device_detector.py    # order=1
├── argument_mining.py               # order=2
├── context_omission_detector.py     # order=3
├── discourse_coherence_analyzer.py  # order=4
├── emotion_target_analysis.py       # order=5
├── framing_analysis.py              # order=6
├── information_density_analyzer.py  # order=7
├── information_omission_detector.py # order=8
├── ideological_language_detector.py # order=9
├── narrative_role_extractor.py      # order=10
├── narrative_conflict.py            # order=11
├── narrative_propagation.py         # order=12
├── narrative_temporal_analyzer.py   # order=13
├── source_attribution_analyzer.py   # order=14
│
├── propaganda_pattern_detector.py   # second-order pattern vector
├── bias_profile_builder.py          # final weighted aggregation + profile vector
│
├── label_analysis.py                # offline label-distribution audit (training)
└── multitask_validator.py           # offline multi-task label cleaner (training)
```

Three things to internalize from the layout:

* **Inference path** is a small core: `analysis_pipeline.py` drives the 14
  analyzers via the registry; `orchestrator.py` adds the propaganda block and
  the bias profile on top.
* **`label_analysis.py` and `multitask_validator.py` are training-side**, not
  inference. They never touch `FeatureContext` or spaCy; they consume a
  pandas dataframe of labels.
* **`__init__.py` is empty.** Everything is imported by absolute path
  (`from src.analysis.X import Y`) — no public re-exports. This keeps the
  registry the only "list of analyzers" you ever need to maintain.

---

## 3. End-to-end flow (one text → one FullAnalysisOutput)

```
                    ┌──────────────────────────────────────────────┐
text  ───────────►  │  AnalysisOrchestrator.analyze(text)          │
                    │                                              │
                    │  1. AnalysisPipeline.analyze(text)           │
                    │     a. preprocess (preprocessing.py)         │
                    │     b. FeatureContext(text=clean,            │
                    │        cache={}, shared={})                  │
                    │     c. ctx.ensure_tokens()  (lazy)           │
                    │     d. for analyzer in registry (order 1→14):│
                    │          BaseAnalyzer.__call__(ctx)          │
                    │            ├ _validate_context (ensures      │
                    │            │   tokens; CRIT-A6 / Section 4)  │
                    │            ├ cache hit? → return             │
                    │            ├ analyze(ctx)  ──── calls        │
                    │            │     get_doc(ctx, task=…)        │
                    │            │     uses ctx.safe_counts(),     │
                    │            │     ctx.text_lower, etc.        │
                    │            ├ _post_validate (clip [0,1],     │
                    │            │   strip unknown keys)           │
                    │            └ cache store                     │
                    │     e. FeatureMerger.merge(features_by_name) │
                    │        → FullAnalysisOutput (pydantic)       │
                    │                                              │
                    │  2. PropagandaPatternDetector.detect(        │
                    │       full_output, ctx)  ── consumes         │
                    │       upstream feature dicts only            │
                    │                                              │
                    │  3. BiasProfileBuilder.build(full_output)    │
                    │     → BiasProfile (per-section scores +      │
                    │        global score + softmax over           │
                    │        manipulation/polarization/…)          │
                    └──────────────────────────────────────────────┘
                                       │
                                       ▼
                    FullAnalysisOutput.to_vector()  → np.ndarray (fixed order)
                    FullAnalysisOutput.dict()       → JSON for /analyze
                    BiasProfile                     → API "headline" payload
```

Batch path (`BatchProcessor.process(texts)`) wraps step 1 in a `nlp.pipe(...)`
loop so one `Doc` per text is built once and seeded into the
`FeatureContext.cache` (see `FeatureContext.from_doc`, `CRIT-A7`, `CRIT-A2`,
`CRIT-A8`). Each text gets its own context and shared dict so no cache or
phrase-hit memo leaks between documents.

---

## 4. File-by-file deep dive

### Pipeline core

#### `analysis_pipeline.py`
* **`AnalysisPipeline(config, registry)`** — the deterministic runner.
  - `__init__` validates config against the registry (`CRIT-A3`): every name
    in `config.enabled_analyzers` and `config.disabled_analyzers` must exist;
    otherwise raises at startup, never silently runs nothing.
  - `analyze(text)` builds a `FeatureContext`, iterates registry entries by
    `order`, calls each analyzer via `BaseAnalyzer.__call__(ctx)` (so
    validation / caching / clipping happen inside the wrapper), and feeds the
    name → features dict into `FeatureMerger`.
  - Handles analyzer-specific kwargs by inspecting `analyze`'s signature
    (`CRIT-A5`) — analyzers that accept `hero_entities` / `villain_entities` /
    `victim_entities` (`narrative_conflict`, `narrative_propagation`) receive
    them only when the orchestrator supplies them; everything else is called
    bare. No reflection at the analyzer side, no surprise kwargs.
  - Batch hook (`analyze_batch`) honors `config.n_process` (`PERF-A5`) by
    delegating to `BatchProcessor` rather than re-implementing `nlp.pipe`.

#### `analysis_registry.py`
* **`AnalyzerRegistry`** — name → (analyzer, order) map. `register(name, …,
  order=…)` rejects duplicate names; `entries()` returns a list sorted by
  `order`.
* **`build_default_registry()`** is the canonical wiring (orders 1-14, see
  Section 2 / Section 3). The order is *not* alphabetical — it is dependency
  shaped: rhetoric and argument first (cheap, self-contained), narrative
  analyzers later (need spaCy syntax + ents already cached on the context),
  source attribution last (uses NER pipe).

#### `analysis_config.py`
* `AnalysisConfig` dataclass: `enabled_analyzers`, `disabled_analyzers`,
  `n_process`, `safe_mode`, `cache_features`, plus per-section thresholds and
  weight overrides for `BiasProfileBuilder`.
* Helpers: `is_enabled(name)`, `merge(other)`, `from_dict(d)`. There is no
  `from_yaml` here — config files are loaded by `config/` and passed in.

#### `orchestrator.py`
* **`AnalysisOrchestrator`** stitches the three layers:
  1. `pipeline.analyze(text)` → `FullAnalysisOutput`,
  2. `PropagandaPatternDetector.detect(...)` filling
     `output.propaganda_pattern`,
  3. `BiasProfileBuilder.build(output)` → `BiasProfile`.
* It is the file that owns *cross-analyzer wiring*: it pulls
  `narrative_role_extractor`'s hero/villain/victim sets out of the merged
  output and threads them to the `narrative_conflict` and
  `narrative_propagation` analyzers (`CRIT-A5`-friendly path: kwargs only
  flow when those analyzers declare them).

#### `batch_processor.py`
* **`BatchProcessor`** wraps `AnalysisPipeline` for many documents.
  - Pre-warms a single spaCy pipe via `get_task_nlp(...)` and uses
    `nlp.pipe(texts, n_process=..., batch_size=...)` so spaCy parses only
    once per text (`PERF-A5`).
  - For each `(text, doc)` pair it constructs a fresh `FeatureContext` via
    `FeatureContext.from_doc(text, doc, safe=True)` so the parser cache is
    reused but every other per-doc cache is empty (`CRIT-A2`, `CRIT-A8`).
    No cross-document feature dict, phrase-hit memo, or `shared` slot leaks
    between texts.
  - Honors `n_process=1` under multiprocess workers (PERF guard against
    nested fork bombs); pinned by `torch.set_num_threads(1)` in the loader.

#### `integration_runner.py`
* CLI/REPL utility: `run(text)` constructs a default config + orchestrator,
  prints the merged dict and the bias profile. Used for ad-hoc debugging; not
  on any inference path.

### Foundations

#### `base_analyzer.py`
* **`BaseAnalyzer` (ABC)** — the contract:
  - `name: str`, `expected_keys: set[str]`, `analyze(ctx, **kwargs) -> dict`.
  - `__call__(ctx, **kwargs)` is the wrapper every analyzer is invoked
    through:
    * `_validate_context(ctx)` — calls `ctx.ensure_tokens()` so `safe_counts`
      and `safe_n_tokens` are warm before `analyze()` runs (`CRIT-A6` /
      `Section 4`).
    * Cache lookup on `ctx.cache[(self.name, frozenset(kwargs.items()))]`.
    * `analyze(ctx, **kwargs)` — actual computation.
    * `_post_validate(features)` — clips numeric values into `[0, 1]`,
      drops keys outside `expected_keys`, fills any missing `expected_key`
      with `0.0`. This is the line of defense against analyzers shipping
      malformed dicts (`F16`).
    * Cache store, then return.
  - On exception, falls back to `{k: 0.0 for k in expected_keys}` and logs
    once — the pipeline never aborts because one analyzer broke on one text.

#### `feature_context.py`
* **`FeatureContext`** — the per-text scratchpad.
  - Fields: `text`, `text_lower` (lazy), `cache` (per-analyzer memo),
    `shared` (free-form cross-analyzer bag), `_doc_by_task` (task → spaCy
    `Doc`), `_token_counts`, `_n_tokens`, `_punct_count_cache`,
    `_phrase_hit_cache`.
  - **Lazy tokenization**: `ensure_tokens()` runs the cheap whitespace
    tokenizer + lemma fallback exactly once and populates `_token_counts`
    and `_n_tokens`. `safe_counts()` / `safe_n_tokens()` are accessors that
    return `{}` / `0` if `ensure_tokens` somehow wasn't called (defensive
    branch, `Section 4`).
  - **`from_doc(text, doc, safe=True)`** constructor used by
    `BatchProcessor`: seeds the spaCy doc into the cache only under the
    `"syntax"` and `"ner"` slots (`CRIT-A7`). It explicitly does **not**
    seed `"fast"` because the fast pipe disables the parser/NER and the
    cached doc would lie about what's available.
  - **`punct_count(symbol)`** — caches `text.count(symbol)` per symbol
    (`PERF-A1`). Multiple analyzers ask for `!` and `?`; this pays each
    cost once.
  - **`cached_phrase_match_count(ctx, phrases)`** (in `_text_features.py`)
    keys by `id(phrases)` on `ctx._phrase_hit_cache` (`PERF-A2`); same
    lexicon set re-used across analyzers hits the cache.

#### `spacy_config.py`
* Static config:
  - `MODEL_NAME` (default `en_core_web_sm` unless overridden).
  - `DEVICE` (auto-detected, GPU if available).
  - `TASK_DISABLE_MAP` — the source of truth for what each task pipe disables
    (Section 6):
    * `"syntax"` — disables `ner` (faster parser-only pipe).
    * `"ner"`    — disables `parser` (faster NER-only pipe).
    * `"fast"`   — disables both `parser` and `ner` (tokenizer + lemma only).
* Used by `spacy_loader` to construct each task pipe exactly once per
  process.

#### `spacy_loader.py`
* Thread-safe multi-model cache: `get_task_nlp(task)` returns the cached
  `Language` for that task, building it under a `threading.Lock` if absent.
* `get_doc(ctx, task=...)` returns the cached `Doc` from
  `ctx._doc_by_task[task]`, building it once via the task pipe if missing.
* GPU initialization is performed exactly once at module import (under a
  guard) so `prefer_gpu()` isn't re-entered per request.
* Under multiprocess workers, `torch.set_num_threads(1)` is set so spaCy
  + torch don't oversubscribe cores (Section 7 / `PERF-A5`).

#### `_text_features.py`
* Shared utilities every analyzer leans on:
  - `term_ratio(counts, n_tokens, lexicon)` — count-based density (single
    tokens only).
  - `phrase_match_count(text_lower, phrases)` — substring scan; tolerant of
    multi-word phrases and underscores via `normalize_lexicon_terms`.
  - `cached_phrase_match_count(ctx, phrases)` — `PERF-A2` cached version,
    keyed by `id(phrases)`.
  - `normalize_lexicon_terms(terms)` — lowercase, replace `_` with space,
    drop empties.
  - `safe_normalized_entropy(values)` — `NUM-A1`. Returns `0.0` for empty,
    zero-mass, or single-bin distributions; otherwise computes
    `H / log(n)`. This is the only entropy the pipeline should call.

#### `preprocessing.py`
* `preprocess(text)` — strip control characters, collapse whitespace, fix
  smart-quote variants, drop trailing junk. Pure function, no analyzer
  hooks.
* `segment_sentences(text)` — fallback sentence splitter for callers that
  don't have a spaCy doc handy (analyzers that *do* have one use
  `doc.sents`).

### Schema and serialization

#### `feature_keys.py`
* The single source of truth for **what features exist and in what order**.
* 15 ordered tuples (one per analyzer + `PROPAGANDA_PATTERN_KEYS`); plus
  `ALL_FEATURE_KEYS` (a dict from section name to its tuple).
* Order is locked here — `to_vector()` and `make_vector(...)` use these
  tuples directly. No alphabetic re-sorting downstream.

#### `feature_schema.py`
* `make_vector(features, keys)` — convert a `dict[str, float]` into a
  numpy vector in the exact order of `keys`, missing keys filled with
  `0.0`, NaNs/inf clipped through `_safe`. Modes: `safe` (default — fill +
  clip), `clip` (just clip, raise on missing), `strict` (raise on
  missing or out-of-range).
* `LockableSchemaRegistry` — a wrapper around `ALL_FEATURE_KEYS` that
  locks at startup (`lock()`) and refuses any mutation thereafter.
* `validate_schema_integrity()` — invariant check executed once at startup:
  no duplicate keys, no overlap across sections, every analyzer's
  `expected_keys` is a subset of its registered tuple. Raises if anything
  drifts.

#### `output_models.py`
* `FeatureModel` (pydantic `BaseModel`) — every section model inherits
  from this. The base validator clips every float to `[0, 1]` and drops
  keys not declared on the subclass.
* One subclass per analyzer (e.g. `RhetoricalDeviceFeatures`,
  `NarrativeConflictFeatures`, …) listing the exact fields from
  `feature_keys.py`.
* **`FullAnalysisOutput`** — composite model holding all 15 sections.
  - `to_dict()` — vanilla pydantic dict.
  - `to_vector()` — concatenates each section's `make_vector(...)` in the
    order declared in `ALL_FEATURE_KEYS`. Sections appended later in
    history are appended at the *end* of the vector (`F3`) so existing
    downstream consumers never get keys reshuffled.

#### `feature_merger.py`
* `FeatureMerger.merge(features_by_name)` — turns the
  `{"rhetorical": {...}, "argument": {...}, ...}` dict produced by the
  pipeline into a fully populated `FullAnalysisOutput`. Missing analyzers
  are filled with their zero-default model. Unknown analyzer names are
  logged and discarded — never raised — so adding a new analyzer doesn't
  hard-crash an old merger.

### The 14 analyzers

Every analyzer here is a `BaseAnalyzer` with a fixed `expected_keys` set
matching one tuple in `feature_keys.py`. Outputs are clipped to `[0, 1]`
by `_safe`/`_post_validate`. Common pattern: token-level signal +
phrase-level signal fused as `0.7 * token + 0.3 * phrase` (anti
double-counting), then optionally per-section relative normalization,
then derived intensity / diversity metrics.

#### 1. `rhetorical_device_detector.py` (`RhetoricalDeviceDetector`)
Lexicons: exaggeration, loaded language, emotional appeal, fear appeal,
intensifiers; pattern lexicons: scapegoat, false dilemma; punctuation
regex `[!?]+`. Output: 7 normalized device scores + intensifier ratio +
scapegoating + false dilemma + `rhetoric_punctuation_score` (counts run
length so `!!!` weighs more than `!`) + `rhetoric_intensity` (mean of
raws) + `rhetoric_diversity` (entropy of normalized distribution).

#### 2. `argument_mining.py` (`ArgumentMiningAnalyzer`)
Lexicons: claim / premise / contrast / support / rebuttal markers. Uses
`get_doc(ctx, task="syntax")` for: VERB density, clause density
(`ccomp` / `xcomp` / `advcl`). Phrase ratios go through
`cached_phrase_match_count` (`PERF-A2`). `argument_complexity` =
`clause_density + verb_density` clipped to `[0,1]`.

#### 3. `context_omission_detector.py` (`ContextOmissionDetector`)
Lexicons: vague references, attribution markers, evidence markers,
uncertainty markers. Quote regex restricted to actual double-quote
glyphs (`F15` — apostrophes were inflating the score on contractions
and possessives). Uses `get_doc(ctx, task="ner")` for entity ratio +
entity-type diversity (`log1p(types)/log1p(20)`). `context_grounding_score`
= weighted aggregate (0.4 evidence + 0.3 entity + 0.2 (1-uncertainty) +
0.1 attribution).

#### 4. `discourse_coherence_analyzer.py` (`DiscourseCoherenceAnalyzer`)
Per-sentence lemma sets (alpha, non-stop) cached on
`ctx.shared["disc_sent_lemmas"]` (`PERF-A7`). Local coherence = mean
adjacent Jaccard; global = first vs last sentence Jaccard; smoothed
`0.7 * local + 0.3 * global`, then `sqrt`. `topic_drift = 1 - coherence`.
`narrative_continuity` (also emitted as `entity_repetition_ratio`,
`F14`) = `sqrt(1 - unique_entities / total_entities)`. Transition ratio
goes through cached phrase matcher.

#### 5. `emotion_target_analysis.py` (`EmotionTargetAnalyzer`)
Builds a spaCy `PhraseMatcher` *eagerly in `__init__`* (`PERF-A3` /
`CRIT-A6`) against `get_task_nlp("ner")`'s vocab. At analyze time it
guards `doc.vocab is self._matcher_vocab` (`CRIT-A4`) — if the runtime
doc was built with a different vocab the matcher silently returns no
hits, so we log once and fall back to token-lemma matching only.
Resolves the *target* of each emotion via NER span first, then
dependency children (`nsubj`, `dobj`, `pobj`, `amod`, `acomp`).
Emits diversity (log-scaled), focus (max target / sum), expression
ratio, type diversity (count / `MAX_EMOTION_TYPES=20`), dominant
emotion strength.

#### 6. `framing_analysis.py` (`FramingAnalyzer`)
Five frames (conflict, economic, moral, human-interest, security)
scored by token + phrase fusion (0.7/0.3). Renormalized so the five
sum to 1 (`F3`-friendly: section is self-normalizing, distribution
shape is the model signal). Adds `frame_dominance_score` (max) and
`frame_diversity_score` (entropy / `log(5)`).

#### 7. `information_density_analyzer.py` (`InformationDensityAnalyzer`)
Six categories: factual, opinion, claim, rhetorical, emotion, modal.
Same fusion + renormalization. Punctuation score uses
`ctx.punct_count(...)` (`PERF-A1`). `information_emotion_ratio =
factual / (factual + emotion)` after normalization. `information_diversity`
goes through `safe_normalized_entropy` (`NUM-A1`).

#### 8. `information_omission_detector.py` (`InformationOmissionDetector`)
Densities for counter-argument, evidence, claim, framing markers,
relative-normalized to a 4-way distribution. Outputs:
* `missing_counterargument_score = 1 - counter_n`
* `one_sided_framing_score = (claim_n + framing_n) / counter_n`,
  squashed into `[0, 1]` via `x / (1 + x)` (bounded logistic form, no
  unbounded ratios).
* `incomplete_evidence_score = 1 - evidence_n`
* `claim_evidence_imbalance = claim_n / (claim_n + evidence_n)`

#### 9. `ideological_language_detector.py` (`IdeologicalLanguageDetector`)
Lexicons: liberty, equality, tradition, anti-elite, plus a phrase
lexicon. Phrase score is added with weight `0.2` to each token raw to
avoid drowning the categorical signal. After renormalization,
`liberty_vs_equality_balance` is mapped from `[-1, 1]` to `[0, 1]`.
`ideology_diversity` = entropy / `log(4)`.

#### 10. `narrative_role_extractor.py` (`NarrativeRoleExtractor`)
Uses `get_doc(ctx, task="syntax")`. Walks tokens; if lemma is a hero
term, assigns the dependency subject as actor and object as victim.
Same for villain. Victim-only terms record their object. Passive
subjects (`nsubjpass`) also count as victims. Entity span resolution
via NER first, falling back to noun/proper-noun lemma. Outputs three
ratios + `hero_vs_villain_balance` mapped from `[-1, 1]` to `[0, 1]`.

#### 11. `narrative_conflict.py` (`NarrativeConflictAnalyzer`)
Accepts optional `hero_entities` / `villain_entities` / `victim_entities`
(orchestrator threads them in via signature inspection, `CRIT-A5`).
Three signal types — `_conflict_verbs(doc)` (verb-fraction), opposition
density, polarization density — relative-normalized. Actor structure
score = `min(hero, villain) + min(villain, victim)` over total mentions.
Punctuation via `ctx.punct_count(...)` (`PERF-A1`).

#### 12. `narrative_propagation.py` (`NarrativePropagationAnalyzer`)
Conflict verbs split into 5 categories (violent, political,
discursive, institutional, coercion). Per-category densities are both
**reported as `_ratio` features directly from `raw`** and
**normalized into a probability distribution `dist`** used only for
`conflict_diversity` (entropy via `safe_normalized_entropy`,
`NUM-A1`). Same actor-structure scoring (hero/villain/victim mins).
Composite `conflict_propagation_intensity = 0.4 * sum(raw) + 0.2 *
opposition + 0.2 * polarization + 0.2 * phrase`.

#### 13. `narrative_temporal_analyzer.py` (`NarrativeTemporalAnalyzer`)
Past / crisis / urgency densities, relative-normalized. Tense
distribution from spaCy `tag_` (VBD/VBN past, VB/VBP/VBZ/VBG present,
`will`/`shall` future). `temporal_contrast_score` = `std(dist) /
sqrt(n_bins-1)/n_bins` — i.e. normalized to the **simplex maximum**
std (`NUM-A5`). For `n=3` the unnormalized std caps at ~0.4714, which
previously made the documented `[0,1]` feature unreachable above
~0.47.

#### 14. `source_attribution_analyzer.py` (`SourceAttributionAnalyzer`)
Lexicons: expert, anonymous, credibility, attribution verbs. Uses
`get_doc(ctx, task="ner")` for `named_source_ratio` (PERSON/ORG count
over doc length). `quotation_ratio` uses paired-quote regex
`\"[^\"]+\"|“[^”]+”` (`F15`) and counts **spans**, not characters,
fixing the prior over-count on stylistic quoting. Balance =
`expert / (expert + anonymous)`, renormalized.

### Second-order layers

#### `propaganda_pattern_detector.py`
* Does **not** consume `ctx`. Operates on the merged feature dicts
  (rhetoric + emotion + narrative_conflict + argument + information).
* Pattern templates are weighted sums over upstream features (e.g.
  manipulation = α·loaded + β·exaggeration + γ·fear-appeal + …) clipped
  to `[0, 1]`.
* Aggregation helpers:
  - `_mean_present(values)` — mean over **present** values (`F10`); if all
    values are zero the score is `0.0` (not `nan`).
  - `safe_normalized_entropy(...)` for diversity-style outputs (`NUM-A1`).
* Emits `PROPAGANDA_PATTERN_KEYS` as the 15th section of `FullAnalysisOutput`.

#### `bias_profile_builder.py`
* **`BiasProfileBuilder(config)`** computes the headline bias profile.
  - `_section_score(model, weights)` — weighted average of a section's
    fields after dropping `_intensity` / `_diversity` *suffix* fields
    (those are meta features, not bias signals).
  - Three normalization modes (per-config `normalization`):
    * `"minmax"` — `(x - min) / (max - min)` over the section's values
      (`NUM-A3`).
    * `"zscore"` — `tanh((x - μ) / σ)` mapped into `[0, 1]` (`NUM-A4`).
    * `"robust"` — `tanh((x - median) / MAD)` mapped into `[0, 1]`.
  - Aggregation: stable softmax over the 6 bias categories
    (manipulation, polarization, framing_bias, source_bias,
    rhetoric_bias, narrative_bias) using `x - x.max()` shift (`NUM-A2`,
    EPS-add dropped — softmax is already safe with the max shift).
  - `global_normalize(...)` is a single-pass linear normalization
    (`PERF-A6`) — older versions iterated min/max twice.
  - **`bias_profile_vector(...)` is deprecated.** Use
    `BiasProfileBuilder.build(...)` and read `BiasProfile` fields.

### Training-side helpers (not on inference path)

#### `label_analysis.py`
* `analyze_labels(df, task_columns, config)` returns a
  `LabelAnalysisResult` (per-task class distributions, imbalance
  flags, rare-class flags, summary). Multi-label columns
  (`list/tuple/set` rows or `"a|b"` strings) are exploded before
  counting.
* `assert_label_health(...)` is the optional hard guard
  (`fail_on_imbalance` / `fail_on_rare`).

#### `multitask_validator.py`
* `validate_multitask_dataframe(df, task_columns, config)` →
  `(clean_df, MultiTaskValidationResult)`. Optionally normalizes
  multi-label string columns to `list[str]`, checks an optional
  `allowed_labels` whitelist, drops rows with `< min_tasks_with_labels`
  labeled tasks. `assert_multitask_health(...)` raises if drop ratio
  exceeds `max_drop_ratio` (default `0.5`).

---

## 5. Metrics and statistical definitions

| Term                           | Definition (as implemented)                                                                 | Where                                            |
| ------------------------------ | ------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| `term_ratio`                   | `Σ counts[t for t in lex if " " not in t] / n_tokens`                                       | `_text_features.term_ratio`                      |
| `phrase_match_count`           | `Σ text_lower.count(p) for p in lex if " " in p`                                            | `_text_features.phrase_match_count`              |
| Token+phrase fusion            | `0.7 * token_score + 0.3 * phrase_score` (anti double-counting)                             | nearly every analyzer's `_score` / `_density`    |
| Section relative normalization | `x_i = raw_i / Σ raw_j` (so a section is a probability over its categories)                 | framing, ideology, info-density, etc.            |
| `safe_normalized_entropy`      | `0` if mass `< EPS` or only one bin; else `H(p) / log(n)`                                   | `_text_features.safe_normalized_entropy`         |
| Jaccard (sentence)             | `|A ∩ B| / |A ∪ B|`, returns `0` if both empty                                              | `discourse_coherence_analyzer._safe_jaccard`     |
| Coherence smoothing            | `sqrt(0.7 * mean_local_jaccard + 0.3 * global_jaccard)`                                     | `discourse_coherence_analyzer.analyze`           |
| `narrative_continuity`         | `sqrt(1 - unique_entities / total_entities)`                                                | `discourse_coherence_analyzer._narrative_continuity` |
| Emotion focus                  | `max(target_weight) / Σ target_weight`                                                      | `emotion_target_analysis.analyze`                |
| Emotion diversity (target)     | `log1p(distinct_targets) / log1p(20)`                                                       | `emotion_target_analysis.analyze`                |
| Frame dominance                | `max(frame_scores)` after renormalization                                                   | `framing_analysis._frame_dominance`              |
| Frame diversity                | `H(frames) / log(5)`                                                                        | `framing_analysis._frame_diversity`              |
| One-sided framing              | `r / (1 + r)` where `r = (claim_n + framing_n) / counter_n` (bounded logistic squash)       | `information_omission_detector.analyze`          |
| Liberty vs equality balance    | `(liberty_n - equality_n + 1) / 2`                                                          | `ideological_language_detector.analyze`          |
| Hero vs villain balance        | `(hero_r - villain_r + 1) / 2`                                                              | `narrative_role_extractor._role_scores`          |
| Conflict propagation intensity | `0.4 * Σ raw + 0.2 * opp + 0.2 * pol + 0.2 * phrase`                                        | `narrative_propagation.analyze`                  |
| Temporal contrast              | `std(dist) / (sqrt(n-1)/n)` (simplex-max normalized, `NUM-A5`)                              | `narrative_temporal_analyzer.analyze`            |
| Past/present/future tense      | Per-tag counts: `VBD/VBN`, `VB/VBP/VBZ/VBG`, `will/shall` — over total verbs                | `narrative_temporal_analyzer._tense_distribution`|
| Quotation ratio                | `len(QUOTE_PATTERN.findall(text)) / n_tokens` where pattern matches paired spans (`F15`)    | `source_attribution_analyzer._quote_density`     |
| Source credibility balance     | `expert_n / (expert_n + anonymous_n)`                                                       | `source_attribution_analyzer.analyze`            |
| Stable softmax                 | `exp(x - max(x)) / Σ exp(x - max(x))` (no manual EPS — `NUM-A2`)                            | `bias_profile_builder._softmax`                  |
| Min-max norm                   | `(x - min) / (max - min)`, identity if range `< EPS` (`NUM-A3`)                             | `bias_profile_builder.global_normalize`          |
| Z-score → bounded              | `tanh((x - μ) / σ)` then `(t + 1) / 2` (`NUM-A4`)                                           | `bias_profile_builder.global_normalize`          |
| Robust → bounded               | `tanh((x - median) / MAD)` then `(t + 1) / 2`                                               | `bias_profile_builder.global_normalize`          |

`EPS` is `1e-8` everywhere it appears (each analyzer redeclares it locally so
modules stay self-contained). Hard clip ceiling `MAX_CLIP / MAX_RATIO_CLIP`
is `1.0` everywhere.

---

## 6. Data assumptions

Inputs and assumptions baked into the analyzers:

* **One text in, one feature record out.** No assumptions about document
  collection, source, language detection, deduplication — those belong to
  `src/data_processing/`. By the time text reaches an analyzer it is
  expected to be plain Unicode prose.
* **English only.** Lexicons are English; spaCy model is `en_core_web_sm`
  by default (`spacy_config.py`). The fallback paths (token-only matching,
  etc.) tolerate non-English input but produce uninformative zeros.
* **Token granularity is whitespace + lemma**, not BPE / WordPiece. This
  is intentional: features here are linguistic, not subword-statistical.
  The transformer tokenization happens entirely in `src/models/`.
* **spaCy task variants** (`spacy_config.TASK_DISABLE_MAP`):
  * `"syntax"` — parser-only (NER disabled). Used by analyzers that need
    `pos_`, `dep_`, `tag_`, `sents`, `lemma_` (argument, discourse,
    narrative_role, narrative_conflict, narrative_temporal).
  * `"ner"` — NER-only (parser disabled). Used by analyzers that only
    need entity spans (context, emotion, source).
  * `"fast"` — both disabled. Tokenizer + lemma only; used by lightweight
    pre-warm paths and never seeded into `from_doc`'s cache (`CRIT-A7`).
* **Empty input is a valid path.** Every analyzer has `_empty_features()`
  / `_empty()` returning the zero-defaulted dict for its keys. The merger
  treats missing analyzers as zeros too. Nothing in this layer raises on
  blank text.
* **Lexicons are expected to be lowercased and de-underscored at
  load-time** (`normalize_lexicon_terms`). Source lexicon constants in
  the analyzer classes are written in human form and normalized in
  `__init__`.
* **No network, no disk writes** during `analyze()`. The only I/O at
  request time is reading the cached spaCy `Doc`.

---

## 7. Output artifacts

The layer emits exactly **one in-memory artifact** per text:

* **`FullAnalysisOutput`** (pydantic) — 15 sections, each a
  `FeatureModel` subclass; values clipped `[0, 1]`.
  * `to_dict()` — JSON-serializable dict for the API.
  * `to_vector()` — fixed-order numpy array; section order matches
    `ALL_FEATURE_KEYS`; new sections appended at the end (`F3`).
  * `dict[section] = FeatureModel`, accessible field-by-field.

The orchestrator additionally returns:

* **`PropagandaPatterns`** (already nested as `output.propaganda_pattern`).
* **`BiasProfile`** — per-section bias scores + global score + softmax
  category distribution. This is the headline payload the API surfaces;
  the underlying `bias_profile_vector(...)` is **deprecated** (use the
  `BiasProfile` fields directly).

There are **no files written** by this layer. Persistence (training
features parquet, label CSVs, etc.) is handled by `src/data_processing/`
and `scripts/` callers — they invoke this layer and write the resulting
vectors themselves.

---

## 8. Config integration

The analyzer layer reads `AnalysisConfig` (`analysis_config.py`); the API /
training entrypoints construct it from the project's YAML config under
`config/`. The fields that actually change behavior:

| Field                              | Effect                                                                      |
| ---------------------------------- | --------------------------------------------------------------------------- |
| `enabled_analyzers: set[str]`      | If non-empty, only these analyzer names run (`CRIT-A3` validates them).     |
| `disabled_analyzers: set[str]`     | Subtracted from registry before running (`CRIT-A3` validates them).         |
| `n_process: int`                   | Forwarded to `nlp.pipe(..., n_process=n)` in `BatchProcessor` (`PERF-A5`).  |
| `safe_mode: bool`                  | Influences `FeatureContext.from_doc(..., safe=True)` seeding (`CRIT-A7`).   |
| `cache_features: bool`             | Toggles per-analyzer cache in `BaseAnalyzer.__call__`.                      |
| `bias_weights: dict[str, float]`   | Passed to `BiasProfileBuilder` for per-section weighting.                   |
| `bias_normalization: str`          | One of `"minmax"`, `"zscore"`, `"robust"` (Section 5).                      |
| `bias_softmax_temperature: float`  | Divides logits before softmax in category aggregation.                      |

Analyzer classes themselves are **not** configurable from YAML —
intentional. Lexicons live next to the analyzer that uses them so
features remain reproducible from a git checkout alone.

`spacy_config.py` exposes `MODEL_NAME` and `DEVICE` as module-level
constants; override via env or by importing and reassigning before
`spacy_loader` is first hit. Once `get_task_nlp` has cached a pipe,
changing the config has no effect for the life of the process.

---

## 9. Validation and sanity guarantees

Hard-fail-at-startup invariants:

* `feature_schema.validate_schema_integrity()` — no duplicate keys, no
  cross-section overlap, every analyzer's `expected_keys` is a subset
  of its registered tuple.
* `LockableSchemaRegistry.lock()` — schema is read-only after startup.
* `AnalysisPipeline.__init__` — every name in `enabled_analyzers` /
  `disabled_analyzers` exists in the registry (`CRIT-A3`).
* `AnalyzerRegistry.register` — duplicate analyzer name raises.
* `EmotionTargetAnalyzer.__init__` — `PhraseMatcher` is built eagerly
  against the canonical NER vocab (`PERF-A3` / `CRIT-A6`); silent
  vocab mismatch at runtime degrades gracefully (`CRIT-A4`).

Per-call defenses:

* `BaseAnalyzer.__call__` always pre-warms tokens (`CRIT-A6` / Section 4),
  then post-validates the analyzer's output: clips floats to `[0, 1]`,
  drops unknown keys, fills missing keys with `0.0` (`F16`).
* Every analyzer redeclares `_safe(value)` to reject non-finite values
  and clip into `[0, 1]`.
* `safe_normalized_entropy` returns `0.0` for empty / zero-mass /
  single-bin distributions (`NUM-A1`).
* Stable softmax uses `x - max(x)` shift, no spurious EPS (`NUM-A2`).
* `BatchProcessor` constructs a fresh `FeatureContext` per text so no
  cache or `shared` slot leaks across documents (`CRIT-A2`, `CRIT-A8`).
* `FeatureContext.from_doc(safe=True)` only seeds `"syntax"` and
  `"ner"` slots — never `"fast"`, which would lie about parser/NER
  availability (`CRIT-A7`).
* Emotion target matcher checks `doc.vocab is self._matcher_vocab` and
  falls back to token-only matching on mismatch (`CRIT-A4`).
* Failure inside an analyzer's `analyze()` returns the zero-defaulted
  dict and logs once — pipeline continues.

Numeric correctness:

* `narrative_temporal.temporal_contrast_score` is normalized to the
  simplex-max std so the documented `[0, 1]` range is reachable
  (`NUM-A5`).
* `BiasProfileBuilder.global_normalize` uses single-pass min/max
  (`PERF-A6`) — earlier versions traversed twice.
* `discourse_coherence`'s `entity_repetition_ratio` is emitted alongside
  the legacy `narrative_continuity` alias to avoid silent
  rename-induced regressions (`F14`).
* `context_omission_detector.QUOTE_PATTERN` excludes apostrophes
  (`F15`); `source_attribution_analyzer.QUOTE_PATTERN` matches paired
  quoted spans, not individual quote chars (`F15`).

---

## 10. Optimization tags

Every audit tag still in code is reproduced here so scans stay coherent.

| Tag      | Where                                                  | Effect                                                                                                    |
| -------- | ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| `CRIT-A2`| `batch_processor.py`                                   | Fresh `FeatureContext` per text — no cache leakage across documents.                                     |
| `CRIT-A3`| `analysis_pipeline.py`                                 | Validate `enabled` / `disabled` analyzer names against the registry at startup.                          |
| `CRIT-A4`| `emotion_target_analysis.py`                           | Vocab-mismatch guard for `PhraseMatcher`; fall back to token-lemma matching.                             |
| `CRIT-A5`| `analysis_pipeline.py`, `orchestrator.py`              | Inspect each analyzer's `analyze()` signature; only forward kwargs it declares.                          |
| `CRIT-A6`| `base_analyzer.py`, `emotion_target_analysis.py`       | `__call__` pre-warms `ctx.ensure_tokens()`; phrase matchers built eagerly.                                |
| `CRIT-A7`| `feature_context.py`, `batch_processor.py`             | `from_doc(safe=True)` only seeds `"syntax"` / `"ner"` cache slots, never `"fast"`.                       |
| `CRIT-A8`| `batch_processor.py`                                   | Per-text `shared` dict — no cross-doc memo bleed.                                                         |
| `PERF-A1`| `feature_context.py`, `info_density`, `narr_conflict`  | `ctx.punct_count(symbol)` cache — `text.count(...)` paid once per (ctx, symbol).                         |
| `PERF-A2`| `_text_features.py`, multiple analyzers                | `cached_phrase_match_count(ctx, phrases)` — `id(phrases)`-keyed memo per ctx.                             |
| `PERF-A3`| `emotion_target_analysis.py`                           | `PhraseMatcher` built once in `__init__`, not per call.                                                   |
| `PERF-A5`| `analysis_pipeline.py`, `batch_processor.py`           | `nlp.pipe(..., n_process=config.n_process)`; `torch.set_num_threads(1)` under multiproc.                  |
| `PERF-A6`| `bias_profile_builder.py`                              | Single-pass min/max in `global_normalize`.                                                                 |
| `PERF-A7`| `discourse_coherence_analyzer.py`                      | Per-sentence lemma sets cached on `ctx.shared["disc_sent_lemmas"]`.                                       |
| `NUM-A1` | `_text_features.safe_normalized_entropy`               | Guards against zero-mass / single-bin distributions.                                                       |
| `NUM-A2` | `bias_profile_builder.py`                              | Stable softmax via `x - max(x)`; dropped manual EPS-add.                                                   |
| `NUM-A3` | `bias_profile_builder.global_normalize` (`minmax`)     | Identity transform when `range < EPS`.                                                                     |
| `NUM-A4` | `bias_profile_builder.global_normalize` (`zscore`)     | `tanh` mapping into `[0, 1]`; identity when σ ≈ 0.                                                        |
| `NUM-A5` | `narrative_temporal_analyzer.py`                       | `temporal_contrast_score` normalized to simplex-max std (`sqrt(n-1)/n`).                                  |
| `F3`     | `output_models.FullAnalysisOutput.to_vector`           | Sections appended later in history concatenated at the end of the vector — never reorder.                 |
| `F10`    | `propaganda_pattern_detector._mean_present`            | Mean over present values; all-zero → `0.0`, never `nan`.                                                  |
| `F14`    | `discourse_coherence_analyzer._narrative_continuity`   | Emit `entity_repetition_ratio` alongside legacy `narrative_continuity`.                                   |
| `F15`    | `context_omission_detector.QUOTE_PATTERN`, `source_attribution_analyzer.QUOTE_PATTERN` | Quote regex fixes (no apostrophes; paired spans).                                          |
| `F16`    | `base_analyzer._post_validate`                         | Clip + drop-unknown + fill-missing on every analyzer's output.                                            |
| Section 4| `feature_context.safe_*`, multiple analyzers           | Defensive `safe_n_tokens()` / `safe_counts()` accessors; never compare `None == 0`.                       |
| Section 6| `spacy_config.TASK_DISABLE_MAP`                        | Source of truth for which spaCy components each task disables.                                            |
| Section 7| `spacy_loader.py`                                      | GPU init once at import; `torch.set_num_threads(1)` under multiprocess.                                    |

---

## 11. Extensibility

Adding a new analyzer:

1. Create `src/analysis/<name>.py` with a `BaseAnalyzer` subclass.
   * Set `name`, `expected_keys = set(<NEW>_KEYS)`, implement
     `analyze(ctx, **optional_kwargs) -> dict`.
   * If you need a spaCy `Doc`, call `get_doc(ctx, task="syntax" | "ner")`.
2. Add a key tuple in `feature_keys.py` and register it in
   `ALL_FEATURE_KEYS`. **Append, do not insert** (`F3`).
3. Add the matching `FeatureModel` subclass to `output_models.py` and a
   field on `FullAnalysisOutput`.
4. Register the analyzer in `build_default_registry()` with a new
   `order=` value.
5. If the analyzer needs cross-analyzer context (entities, etc.), add
   the kwargs in your `analyze()` signature — the pipeline will detect
   them via `CRIT-A5` reflection and route them only when the
   orchestrator threads them in.
6. Run the integration harness:
   `python -m src.analysis.integration_runner "<sample text>"`.
   Schema integrity validation will fail loudly at startup if anything
   drifts.

Adding a new propaganda pattern: edit
`PROPAGANDA_PATTERN_KEYS` in `feature_keys.py` (append) and add the
weighted aggregation in `propaganda_pattern_detector.py`. Anything that
references the upstream feature dicts is fair game — the detector is
schema-bound to those analyzers' outputs by name.

Changing bias normalization: pass
`bias_normalization="minmax" | "zscore" | "robust"` via `AnalysisConfig`.
Adding a new mode requires a branch in
`BiasProfileBuilder.global_normalize` — keep the contract: in `[0, 1]`,
finite, deterministic.

---

## 12. Pitfalls

* **Do not bypass `BaseAnalyzer.__call__`.** Calling `analyzer.analyze(ctx)`
  directly skips `ensure_tokens()` (`CRIT-A6` / Section 4), the
  `[0,1]` clip, the unknown-key drop, and the missing-key fill (`F16`).
  The defensive `safe_n_tokens()` accessor exists precisely so this
  shortcut doesn't blow up — but the result will be schema-non-conformant.
* **Do not seed `"fast"` pipe docs into `FeatureContext`.** Fast pipe
  has no parser and no NER; analyzers that ask for `task="syntax"` or
  `task="ner"` would silently get empty `doc.sents` / `doc.ents`.
  `from_doc(safe=True)` already enforces this (`CRIT-A7`); just don't
  override the cache by hand.
* **Lexicon mutation at runtime poisons the phrase cache.** The phrase
  hit memo is keyed by `id(phrases)` (`PERF-A2`). If you mutate the same
  set after a cache entry has been written, hits will be stale.
  Build new sets via `normalize_lexicon_terms` and reassign instead.
* **`PhraseMatcher` vocab identity is required**, not just equality
  (`CRIT-A4`). If you build a matcher in one process and ship it across
  a fork, vocabs differ and matching silently returns no hits. The
  emotion analyzer guards this by logging once and falling back to
  token-only matching — but other analyzers using `PhraseMatcher` would
  not.
* **`bias_profile_vector(...)` is deprecated.** It still computes a
  vector for backward-compat callers but new code must read fields off
  `BiasProfile`. The vector ordering is not guaranteed past additional
  bias categories.
* **Section ordering in `to_vector()` is append-only (`F3`).** Inserting
  a new section in the middle of `ALL_FEATURE_KEYS` will shift every
  downstream feature index — this **will** break trained models that
  consumed the prior order.
* **`n_process > 1` plus an already-multi-threaded torch is lethal.**
  Worker processes must run with `torch.set_num_threads(1)` (the
  loader sets it). Bypassing the loader and calling `spacy.load`
  directly inside a worker is a classic way to OOM the box.
* **Empty / zero-mass distributions are not errors here.** They produce
  `0.0` everywhere via `safe_normalized_entropy` (`NUM-A1`) and the
  per-analyzer `_safe`. Don't add `EPS` "to be safe" — the helpers
  already handle it correctly, and extra EPS biases your normalized
  distributions away from `0`.

---

## 13. Example usage

### Single text (via the orchestrator)

```python
from src.analysis.analysis_config import AnalysisConfig
from src.analysis.analysis_registry import build_default_registry
from src.analysis.analysis_pipeline import AnalysisPipeline
from src.analysis.orchestrator import AnalysisOrchestrator

cfg       = AnalysisConfig()             # defaults are sensible
registry  = build_default_registry()     # 14 analyzers, ordered
pipeline  = AnalysisPipeline(cfg, registry)
orch      = AnalysisOrchestrator(pipeline)

text   = "President X attacked the new bill, calling it a betrayal of working families."
output = orch.analyze(text)              # FullAnalysisOutput

print(output.framing.dict())             # one section
print(output.propaganda_pattern.dict())  # second-order section
print(output.to_vector().shape)          # fixed-order numpy vector
print(orch.last_bias_profile)            # BiasProfile (or returned tuple — see signature)
```

### Batch (training / bulk inference)

```python
from src.analysis.analysis_config import AnalysisConfig
from src.analysis.analysis_registry import build_default_registry
from src.analysis.analysis_pipeline import AnalysisPipeline
from src.analysis.batch_processor import BatchProcessor

cfg = AnalysisConfig(n_process=4, cache_features=True)
pipeline = AnalysisPipeline(cfg, build_default_registry())
batch = BatchProcessor(pipeline, batch_size=64)

texts   = [...]                          # list[str]
outputs = batch.process(texts)           # list[FullAnalysisOutput]

import numpy as np
X = np.stack([o.to_vector() for o in outputs])
```

### Just one analyzer (debugging)

```python
from src.analysis.feature_context import FeatureContext
from src.analysis.framing_analysis import FramingAnalyzer

ctx  = FeatureContext(text="…", cache={}, shared={})
out  = FramingAnalyzer()(ctx)            # goes through __call__ wrapper
print(out)                               # dict of seven framing keys, [0,1]
```

### Quick smoke run (CLI)

```bash
python -m src.analysis.integration_runner "Officials warned of an imminent crackdown."
```

This prints the merged feature dict plus the bias profile and is the
fastest way to verify a freshly-added analyzer end-to-end before
hooking it into the API.
