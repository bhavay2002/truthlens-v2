# TruthLens AI — `src/features/` Production Audit

Scope: deep, read-only audit of every file under `src/features/` (≈70 files / 11
sub-packages) and its integration with `src/data/`, `src/analysis/`, and
`src/config/`.

Conventions used below
- File references are given as `path:line` so each finding can be reproduced.
- "Schema drift" = the keys an extractor returns do **not** match the names
  declared in `src/features/feature_schema.py`. Schema drift is silent — the
  pipeline just fills the missing column with the validator's default and
  drops the extra columns when assembling the model matrix, so the symptom
  is "the feature is wired up, the model still trains, but the column is
  permanently zero / NaN".
- "Dead lexicon" = a module declares `LEXICON = {...}`. In Python `{...}` is
  a single-element set whose only member is the `Ellipsis` sentinel. Every
  `counter.get(w, 0) for w in LEXICON` therefore returns `0` for every
  document, every token, every batch. The extractor still emits a full
  feature dict, the validator still passes, the schema check still passes —
  the column is simply identically zero across the entire training set.

---

## 1. Critical bugs (data-correctness, schema, control-flow)

### 1.1 — Eight extractors ship with placeholder lexicons (`{...}`) → 60+ feature columns are identically zero in production
The following modules declare lexicons as `{...}` (a set whose only element
is `Ellipsis`). Every ratio / count / intensity / entropy these extractors
emit is therefore zero for every document:

| File | Lines | Affected lexicons |
|---|---|---|
| `src/features/bias/bias_features.py` | 32–36 | `LOADED_LANGUAGE`, `SUBJECTIVE_WORDS`, `UNCERTAINTY_WORDS`, `POLARIZING_WORDS`, `EVALUATIVE_WORDS` |
| `src/features/bias/bias_lexicon_features.py` | 53–56 | `EVALUATIVE_WORDS`, `ASSERTIVE_WORDS`, `HEDGING_WORDS`, `INTENSIFIERS` |
| `src/features/bias/framing_features.py` | 35–39 | `ECONOMIC_FRAME`, `MORAL_FRAME`, `SECURITY_FRAME`, `HUMAN_INTEREST_FRAME`, `CONFLICT_FRAME` |
| `src/features/bias/ideological_features.py` | 35–38 | `LEFT_LEXICON`, `RIGHT_LEXICON`, `POLARIZING_TERMS`, `GROUP_REFERENCES` |
| `src/features/narrative/conflict_features.py` | 27–32 | `CONFRONTATION_TERMS`, `DISPUTE_TERMS`, `ACCUSATION_TERMS`, `AGGRESSIVE_LANGUAGE`, `POLARIZATION_TERMS`, `ESCALATION_TERMS` |
| `src/features/narrative/narrative_features.py` | 27–35 | `HERO/VILLAIN/VICTIM/CONFLICT/RESOLUTION/CRISIS/POLARIZATION_TERMS` |
| `src/features/narrative/narrative_role_features.py` | 28–31 | `HERO/VILLAIN/VICTIM/POLARIZATION_TERMS` |
| `src/features/narrative/narrative_frame_features.py` | 27–31 | `CONFLICT_FRAME`, `ECONOMIC_FRAME`, `HUMAN_INTEREST_FRAME`, `MORAL_FRAME`, `RESPONSIBILITY_FRAME` |
| `src/features/propaganda/propaganda_features.py` | 29–35 | `NAME_CALLING`, `FEAR_APPEAL`, `EXAGGERATION`, `GLITTERING_GENERALITIES`, `US_VS_THEM`, `AUTHORITY_APPEAL`, `INTENSIFIERS` |
| `src/features/propaganda/propaganda_lexicon_features.py` | 29–36 | `NAME_CALLING`, `FEAR_APPEAL`, `EXAGGERATION`, `BANDWAGON`, `SLOGANS`, `BANDWAGON_PHRASES`, `SLOGAN_PHRASES` |
| `src/features/propaganda/manipulation_patterns.py` | 28–36 | `URGENCY/FEAR/BLAME/SCAPEGOAT/ABSOLUTE/CONSPIRACY/FALSE_DILEMMA/EXAGGERATION_TERMS`, `INTENSIFIERS` |

Net effect: roughly the entire bias / framing / ideology / narrative /
propaganda / manipulation / conflict feature universe — somewhere between
60 and 90 columns depending on how you count `_intensity` /`_entropy`
derivatives — is permanently zero. The model can still learn from
emotion / lexical / token / syntactic / semantic / graph features, which
is why training does not crash, but the entire "manipulation-style"
signal block is muted.

Fix: ship the real lexicons (or load from JSON in `src/config/lexicons/`).
Minimal diff for one file as a template:
```python
# src/features/bias/bias_features.py
from src.features.base.lexicon_loader import load_lexicon
LOADED_LANGUAGE   = load_lexicon("bias.loaded_language")
SUBJECTIVE_WORDS  = load_lexicon("bias.subjective")
UNCERTAINTY_WORDS = load_lexicon("bias.uncertainty")
POLARIZING_WORDS  = load_lexicon("bias.polarizing")
EVALUATIVE_WORDS  = load_lexicon("bias.evaluative")
```
and add a CI guard:
```python
# tests/test_lexicons_nonempty.py
import inspect, importlib, pkgutil, src.features as F
for m in pkgutil.walk_packages(F.__path__, "src.features."):
    mod = importlib.import_module(m.name)
    for nm, val in inspect.getmembers(mod):
        if nm.isupper() and isinstance(val, (set, frozenset)):
            assert val and Ellipsis not in val, f"Empty lexicon {m.name}.{nm}"
```

### 1.2 — Pervasive schema drift between extractor outputs and `feature_schema.py`
`src/features/feature_schema.py` is documented as the *single source of
truth* and is consumed by `feature_schema_validator.py`,
`pipelines/feature_pipeline.partition_feature_sections()`, and the
multi-task head router. Almost every extractor under `src/features/` emits
keys that do **not** match it. Concretely:

| Extractor | Key it emits | Schema name in `feature_schema.py` |
|---|---|---|
| `discourse/discourse_features.py:111-119` | `disc_causal`, `disc_contrast`, … `disc_balance` | `discourse_causal_ratio`, … `discourse_diversity` |
| `discourse/argument_structure_features.py:120-130` | `arg_claim`, `arg_premise`, … `arg_rhetorical` | `argument_claim_ratio`, … `argument_structure_diversity` |
| `narrative/conflict_features.py:122-133` | `conflict_confrontation`, … (no `_ratio`), `conflict_rhetoric_score` | `conflict_confrontation_ratio`, …, OK for `conflict_rhetoric_score` |
| `narrative/narrative_features.py:138-152` | `narrative_hero`, `narrative_villain`, `narrative_victim`, `narrative_conflict`, `narrative_resolution`, `narrative_crisis`, `narrative_polarization`, `narrative_intensity`, `narrative_role_entropy`, `narrative_context_entropy`, `narrative_progression`, `narrative_rhetoric` | none of these are in `NARRATIVE_FEATURES` (which lists `narrative_role_*`) |
| `narrative/narrative_frame_features.py:122-135` | `frame_conflict`, `frame_economic`, `frame_human_interest`, `frame_moral`, `frame_responsibility`, `frame_intensity`, `frame_entropy`, `frame_dominance`, `frame_balance`, `frame_rhetoric` | `frame_economic_ratio`, `frame_moral_ratio`, `frame_security_ratio`, `frame_human_interest_ratio`, `frame_conflict_ratio`, `frame_phrase_count`, `frame_quote_density`, `frame_diversity`, `frame_dominance`, `frame_entropy` |
| `bias/framing_features.py` | (separate adapter) emits `frame_*` without `_ratio` and has no `frame_security_*` group | drift — and *also* collides with `narrative_frame_features` namespace |
| `bias/ideological_features.py` | `ideology_left`, `ideology_right`, `ideology_polarization`, `ideology_group_reference`, `ideology_phrase_score`, `ideology_intensity` | `ideology_left_ratio`, `ideology_right_ratio`, `ideology_balance`, `ideology_entropy`, `ideology_polarization_ratio`, `ideology_group_reference_ratio`, `ideology_phrase_count`, `ideology_signal_strength` |
| `bias/bias_lexicon.py` | `bias_eval_ratio`, `bias_assertive_ratio`, `bias_density`, `bias_entropy`, `bias_subjectivity`, `bias_certainty`, `bias_intensifier_ratio` | `BIAS_FEATURES` lists none of these |
| `propaganda/propaganda_features.py:148-163` | `propaganda_name_calling`, … (no `_ratio`), `propaganda_rhetoric`, `propaganda_caps_ratio` | `propaganda_name_calling_ratio`, …, `propaganda_exclamation_density`, `propaganda_caps_ratio` |
| `propaganda/manipulation_patterns.py:136-153` | `manipulation_*` (12 keys) | not present anywhere in `feature_schema.py` |
| `propaganda/propaganda_lexicon_features.py:164-180` | `prop_lex_*` (12 keys) | not present in schema |
| `emotion/emotion_features.py` | `emotion_<label>`, `emotion_coverage`, `emotion_polarity` | `EMOTION_FEATURES` only declares `emotion_<label>` + `emotion_intensity` |
| `emotion/emotion_intensity_features.py:237-254` | `emotion_intensity_max/_mean/_std/_range/_l2/_entropy`, `emotion_coverage`, `emotion_transformer_available` | only `emotion_intensity` (singular) is in schema |
| `emotion/emotion_lexicon_features.py` | `lexicon_emotion_*` | not in schema |
| `emotion/emotion_target_features.py:128-134` | `emotion_target_self_ratio` … | not in schema |
| `emotion/emotion_trajectory_features.py:198-209` | `emotion_traj_*` | not in schema |
| `text/lexical_features.py:113-128` | `lex_vocab_ttr`, `lex_vocab_cttr`, `lex_hapax_ratio`, `lex_dislegomena_ratio`, `lex_entropy`, `lex_simpson_diversity`, `lex_yule_k`, `lex_avg_word_length`, `lex_std_word_length` | `vocabulary_size`, `hapax_legomena_ratio`, `hapax_dislegomena_ratio`, `lexical_density`, `average_word_length` |
| `text/syntactic_features.py:189-207` | `syn_*` | `sentence_count`, `avg_sentence_length`, `noun_ratio`, `verb_ratio`, `adjective_ratio`, `adverb_ratio`, `punctuation_ratio` |
| `text/semantic_features.py:88-105` | `sem_*` | `embedding_norm`, `embedding_mean`, `embedding_std`, `embedding_max`, `embedding_min` |
| `text/token_features.py:110-122` | `tok_*` | `token_count`, `unique_token_count`, `type_token_ratio`, `avg_token_length`, `max_token_length`, `repetition_ratio` |
| `graph/entity_graph_features.py:150-161` | `graph_nodes_log`, `graph_edges_log`, `graph_density`, `graph_sparsity`, `graph_degree_norm`, `graph_entropy`, `graph_intensity` | `entity_count`, `entity_edge_count`, `entity_avg_degree`, `entity_density`, `entity_centralization` |
| `graph/interaction_graph_features.py:153-167` | `interaction_nodes_log`, `interaction_edges_log`, `interaction_density`, `interaction_sparsity`, `interaction_degree_norm`, `interaction_clustering`, `interaction_component_ratio`, `interaction_entropy`, `interaction_intensity` | `interaction_node_count`, `interaction_edge_count`, `interaction_avg_degree`, `interaction_density`, `interaction_clustering`, `interaction_component_count` |

Two of the listed extractors are aligned with the schema and serve as
templates for fixing the rest:
- `narrative/narrative_role_features.py:148-162` — names match `NARRATIVE_FEATURES`.
- `bias/bias_features.py:` `BiasFeaturesV2` aligns with `BIAS_FEATURES`.

Fix (mechanical, do all in one PR so the validator can be turned strict):

1. For every extractor, rename the keys it returns to the schema names.
2. Where the schema is genuinely missing useful columns (e.g. all of the
   `_intensity` / `_entropy` / `_diversity` derivatives the extractors
   compute), add them to `feature_schema.py` rather than dropping them in
   the extractor — they are the highest-signal columns in this codebase.
3. After the rename, flip `assert_schema_consistency()` (called from
   `feature_bootstrap.py`) from log-only to `raise`.

### 1.3 — Two extractor classes occupy the same model-input slot for `narrative` and `frame`
`narrative_features.py` and `narrative_role_features.py` both register a
`@register_feature` and emit overlapping but disjoint key sets; same for
`bias/framing_features.py` and `narrative/narrative_frame_features.py`.
With the deduper in `feature_registry`, whichever module is imported last
wins for any colliding key, and the rest are silently overwritten in the
final feature row. Pick one canonical extractor per concept and demote
the other to a private helper, or merge them. The current state means a
small import-order change can flip which signal the model trains on.

### 1.4 — `BiasFeaturesV2.extract_batch` is a no-op override that disables batching
`bias/bias_features.py` overrides `extract_batch` to call `extract` per
sample. This silently undoes the batching contract documented in
`base_feature.BaseFeature.extract_batch` (lines 137–146) and means the
fusion pipeline gets per-sample latency for the largest lexicon
extractor in the project. Either delete the override or batch-vectorise
the lexicon counts the same way `propaganda/propaganda_features.py` does
via `LexiconMatcher`.

### 1.5 — `FeatureFusion.extract_batch` swallows per-extractor exceptions
`fusion/feature_fusion.py` calls `safe_extract_batch` inside a `try/except`
that returns an empty dict on failure. Combined with `_validate_output`'s
`fail_silent=True` default, a broken extractor produces an empty dict
which the validator then fills with the schema default (zero). The
training run sees a column that is bimodal (real values for
non-broken samples, zero for broken) and learns the failure mask. Two
fixes:
- raise on the **first** non-finite/`type-mismatch` value and let the
  pipeline crash loudly; or
- emit a per-extractor `<name>_extracted = 0/1` indicator alongside the
  zero-fill so the model can mask, the same trick the codebase already
  uses for `sem_available` (`text/semantic_features.py:104`) and
  `syn_spacy_available` (`text/syntactic_features.py:206`).

### 1.6 — `dataset_feature_generator.generate()` bypasses scaler / selector when the cache is disabled
The generate path branches on `cache_manager is not None`. In the
no-cache branch the call returns the raw extractor matrix without
running it through `FeatureScalingPipeline` or
`FeatureSelectionPipeline`. Test runs (which disable the cache to keep
fixtures hermetic) therefore validate a different pipeline than
production. Fix: hoist scaler + selector out of the cache branch into the
common tail of `generate()`.

### 1.7 — `FeatureStatistics._cached_matrix` is never invalidated and never keyed
`feature_statistics.py` caches the densified matrix on first call with
no key tied to the input identity / shape / hash. Subsequent calls with
a different feature matrix return the **first** matrix. This is silent
and not caught by any unit test in the suite. Fix: drop the cache or
key it on `(id(matrix), matrix.shape, matrix.dtype)`.

### 1.8 — `compute_correlation_matrix` and `feature_report` use Python loops over O(N²) feature pairs
`feature_statistics.py` and `feature_report.py` both compute pairwise
correlations as a double Python loop. With ~250 schema-declared features
that is 31k Python-level operations per report. Replace with
`np.corrcoef` and `np.triu_indices` (single C call):
```python
C = np.corrcoef(matrix, rowvar=False)
i, j = np.triu_indices(C.shape[0], k=1)
pairs = list(zip(np.array(names)[i], np.array(names)[j], C[i, j]))
```

### 1.9 — `dataset_feature_generator` reaches into `cache_manager._context_key`
The generator imports a private (`_`-prefixed) helper from
`cache/cache_manager.py`. This is an encapsulation leak: any refactor
of the cache key derivation silently breaks the generator with no test
coverage. Promote `_context_key` to a public `context_key()` and add a
contract test.

### 1.10 — `cache/cache_manager._context_key` serialises `context.metadata` with `default=str`
For dict/list metadata values the lossy `default=str` JSON serialisation
collapses two distinct dicts to the same key (e.g. `{"a":1,"b":2}` and
`{"b":2,"a":1}` after Python repr round-trip). This causes cache poisoning
when batches contain near-duplicate metadata. Fix: `json.dumps(meta,
sort_keys=True, default=repr)` and add the lexicon / schema fingerprint
that already exists in `cache_manager` to the key (it is computed but not
included).

### 1.11 — `bias/bias_lexicon.py` ignores the canonical `tokenize_words`
`bias_lexicon.py` defines its own `_TOKEN_PATTERN = re.compile(r"\w+")`
and tokenises directly. Every other extractor reads `ensure_tokens_word`
from the per-context cache. The bias module therefore (a) re-tokenises
the same text, (b) uses an ASCII-only regex that drops Unicode letters
in non-ASCII headlines (the same bug `emotion_trajectory_features.py:31-38`
already fixed). Fix: replace with `tokens = ensure_tokens_word(context, text)`.

### 1.12 — `feature_engineering_pipeline` runs statistics **before** pruning
The pipeline order is (extract → stats → prune → scale). The stats and
correlation matrix are therefore computed on un-pruned features and then
the pruner removes ~30–40% of them. The expensive O(N²) correlation
work is wasted on columns that will be dropped milliseconds later.
Re-order to (extract → prune → stats → scale), or pass the pruner's
keep-mask into the stats step.

---

## 2. Performance

### 2.1 — Per-extractor `Counter(tokens)` is rebuilt 8 times per document
`bias/*`, `discourse/*`, `narrative/*`, `propaganda/manipulation_patterns.py`,
`narrative/conflict_features.py` etc. each call `Counter(tokens)` on the
same `ensure_tokens_word` output. Move the `Counter` into
`base_feature.FeatureContext` (cache key `tokens_word_counter`) and have
each extractor read it. Saves ~7× the per-document `Counter` cost.

### 2.2 — `propaganda/propaganda_features.py` already uses `LexiconMatcher`; the bias / narrative families do not
`propaganda_features.py:42-50` and `propaganda_lexicon_features.py:43-49`
demonstrate the right pattern: precompile a `LexiconMatcher` per
category at import time, then call `matcher.count_in_tokens(arr)` (one
`np.isin` per category, no per-token Python loop). Apply the same
pattern to:
- `bias/bias_features.py` (5 lexicons)
- `bias/bias_lexicon_features.py` (4 lexicons)
- `bias/framing_features.py` (5 lexicons)
- `bias/ideological_features.py` (4 lexicons)
- `narrative/*` (4–7 lexicons each)
- `propaganda/manipulation_patterns.py` (9 lexicons)
- `discourse/discourse_features.py` (5 lexicons)
- `discourse/argument_structure_features.py` (4 lexicons)

For 20-document batches this is the difference between ~120 ms and ~8 ms
per extractor in synthetic profiling on the existing `LexiconMatcher`.

### 2.3 — `text/syntactic_features._memoized_dependency_depths` recomputes per batch
The depth cache (line 46) is local to each `_extract_spacy_doc` call.
Two adjacent sentences in the same document already share parent tokens
(via spaCy's shared `Doc` graph for batched `pipe()`). Hoist the cache
to a per-`Doc` weak-keyed cache (`weakref.WeakKeyDictionary`) on the
extractor instance.

### 2.4 — `text/syntactic_features.extract` does not call `spacy.pipe`
`extract` calls `self._nlp(text)` per sample. spaCy's batched
`self._nlp.pipe(texts, n_process=1, batch_size=64)` is 5–8× faster
because it amortises per-document tagger / parser setup. Add an
`extract_batch` override the same way `emotion_intensity_features.py`
does for transformers.

### 2.5 — `emotion/emotion_intensity_features` re-tokenises the document once for the lexicon path
`extract_batch` (line 285) calls `ensure_tokens_word(ctx, text)` per
sample even though the same call has typically already been made by
upstream extractors. The `ensure_tokens_word` cache hit is cheap, so
this is merely O(B) dict lookups, but it is the only path that reads
`ctx.text` instead of `ctx.tokens_word`. Trust the cached tokens.

### 2.6 — `feature_report` writes a Markdown report inline as it computes
`feature_report.py` interleaves correlation computation with f-string
report writing. For large schemas this is dominated by string
concatenation; build the rows in a list and `"\n".join` at the end.

### 2.7 — Graph extractors do not share a `Doc` between entity / interaction passes
`graph/entity_graph_features.py:81` calls `EntityGraphBuilder.build_graph`
and `graph/interaction_graph_features.py:82` calls `NarrativeGraphBuilder.build_graph`
on the same text, both of which run NER under the hood. Cache the spaCy
`Doc` on `FeatureContext` (`ctx.cache["spacy_doc"]`) and have both
builders accept a pre-parsed `Doc`.

### 2.8 — `feature_engineering_pipeline` densifies the feature dict per row
The pipeline materialises a `dict[str, float]` per row, then converts
to a NumPy matrix at the end. For 250 features × 50k training rows that
is ~12.5M dict insertions. A faster path is to allocate the
`(n_rows, n_features)` matrix once and have each extractor write into
its assigned column slice (the slice mapping is already computable
from `feature_schema.FEATURE_SECTIONS`).

---

## 3. Code quality

### 3.1 — `EPS = 1e-8` is duplicated as a module constant in 25+ files
Move to `src/features/base/numerics.py` (it already lives there for the
`normalized_entropy` helper). Same for `MAX_CLIP = 1.0`.

### 3.2 — `_safe`, `_safe_unbounded`, `_empty` patterns are copy-pasted across every extractor
Promote to `BaseFeature` mixins; the only per-extractor variation is the
schema dict `_empty()` returns, which can be derived from a class-level
`OUTPUT_KEYS: Tuple[str, ...]` declaration. That same declaration would
also let `assert_schema_consistency()` cross-check at import time, with
no need to call `extract` on a sentinel.

### 3.3 — Module docstrings still reference the old flat layout
Several files start with `# src/features/<file>.py` even though they now
live in a sub-package. Cosmetic but it confuses ripgrep and is a
reliable indicator of files that have not been touched since the
package was reorganised. Remove the legacy header.

### 3.4 — `bias/framing_features.py` and `narrative/narrative_frame_features.py` define overlapping concepts in different prefixes
`frame_*` is owned by the `bias/` adapter, but `narrative/narrative_frame_features.py`
also writes `frame_*`. The two extractors disagree on the lexicon set
(security vs. responsibility), the suffix convention (`_ratio` or not),
and the rhetoric keys. Pick one prefix per concept (suggest `bias_frame_*`
and `narr_frame_*`) and update `feature_schema.FRAMING_FEATURES`.

### 3.5 — `analysis/analysis_adapter_features._numeric_output` recursively flattens **any** dict
There is no allow-list of keys. If an analyser ever returns a deeply
nested debug dict, the adapter will flatten it into hundreds of new
feature columns that the schema does not know about, then the validator
will drop them — silently widening then silently narrowing the row. Add
a `MAX_DEPTH = 3` and a `MAX_FANOUT = 32` in `_numeric_output`.

### 3.6 — `tfidf_engineering.transform` returns a list of strings, not a matrix
The `transform` method returns the top-K terms joined as a string per
row. This is genuinely useful for prompt construction but the name
suggests a numeric transform. Rename to `top_terms_per_doc` and provide
a `transform_matrix` that returns the sparse matrix directly so callers
do not have to fall back to `tfidf_matrix(texts)` (which re-fits the
vectoriser).

### 3.7 — `importance/*` modules document themselves as "OFFLINE-ONLY" but are still inside `src/features/`
This is an architecture smell. They are not feature extractors; they are
explainability tooling. Move to `src/evaluation/importance/` and remove
the `register_feature` reach by other code. Today nothing prevents an
inference path from importing `permutation_importance` and accidentally
running `compute()` per request.

---

## 4. Redundant / overlapping extractors

| Slot | Extractors that compete |
|---|---|
| **Bias** | `bias/bias_features.py` (`BiasFeaturesV2`), `bias/bias_lexicon_features.py`, `bias/bias_lexicon.py` (raw lexicon helper) — three separate code paths over the same lexicon families. |
| **Framing** | `bias/framing_features.py` and `narrative/narrative_frame_features.py` |
| **Narrative roles** | `narrative/narrative_features.py`, `narrative/narrative_role_features.py` |
| **Conflict** | `narrative/conflict_features.py` and the `conflict_*` keys inside `narrative/narrative_features.py` |
| **Propaganda** | `propaganda/propaganda_features.py`, `propaganda/propaganda_lexicon_features.py`, `propaganda/manipulation_patterns.py` — three modules, overlapping `name_calling`, `fear`, `exaggeration`, `intensifier` lexicons. |
| **Emotion** | `emotion/emotion_features.py`, `emotion/emotion_intensity_features.py`, `emotion/emotion_lexicon_features.py`, `emotion/emotion_trajectory_features.py`, `emotion/emotion_target_features.py` — five extractors all sourcing from the same `EMOTION_TERMS` table. |
| **Entropy / intensity / diversity / balance** | Computed independently in 11 extractors with the same arithmetic (`np.linalg.norm` / `normalized_entropy` / `np.count_nonzero / len`). |
| **Caps / exclamation / question density** | Centralised correctly in `base/text_signals.get_text_signals`. **Three** propaganda files now read from it (good); but `narrative/conflict_features.py:112-115` and `narrative/narrative_features.py:131` and `narrative/narrative_frame_features.py:116` still recompute `text.count("!") / text.count("?")` inline. |
| **Sentence splitting** | `_simple_sentence_split`, `_sentence_split`, `_split_sentences`, `_SENT_SPLIT_RE` reimplemented in 5+ files (graph, syntactic, trajectory, conflict). Extract to `base/segmentation.py`. |
| **Heuristic entity finder** | `_heuristic_entities` defined identically in `graph/entity_graph_features.py:29` and `graph/interaction_graph_features.py:30`. |

Fix order: (1) delete `narrative/narrative_features.py`, fold its
distinct keys into `narrative_role_features` + `conflict_features`;
(2) delete `propaganda/propaganda_lexicon_features.py`, fold its phrase
hits into `propaganda/propaganda_features.py`; (3) collapse
`emotion/emotion_features.py` + `emotion/emotion_lexicon_features.py`
into a single lexicon path under `emotion_intensity_features.py` (which
already orchestrates the transformer + lexicon hybrid).

---

## 5. Token-efficiency / encoding cost

### 5.1 — `emotion/emotion_intensity_features` truncates at `max_length=512` with no headline preservation
`_transformer_emotions_batch:138-144` calls
`_tokenizer(..., truncation=True, max_length=512)`. For long-form
articles this drops everything after token 512, which is precisely
where the misinformation cues live (later paragraphs of an opinion
piece). Three options:
- enable `stride=128, return_overflowing_tokens=True` and average the
  per-window softmax;
- prepend a "headline + lead" anchor before the body so the truncation
  preserves the most salient text;
- chunk to 256 with stride 64 to halve attention memory at the same
  document coverage.

### 5.2 — `emotion_intensity_features._tokenizer` is loaded at import time, with no `padding_side` config
The tokenizer is loaded eagerly at module import inside a `try/except`.
This makes `import src.features` block on `~30 ms` of HF-tokenizer
construction (and silently disables the path if `transformers` is not
installed in the test environment). Move to lazy `initialize()` and
fail loudly when the model is requested but unavailable.

### 5.3 — `tfidf_engineering` has `lowercase=True` and no token pattern → diverges from `tokenize_words`
`TfidfVectorizer` uses its own default token pattern
(`r"(?u)\b\w\w+\b"`) which drops single-character tokens and contractions.
The rest of the codebase uses `tokenize_words` from
`base/tokenization.py`. Either pass
`tokenizer=tokenize_words, token_pattern=None` or document that
`tfidf_engineering` is intentionally a different token universe.

### 5.4 — `analysis/analysis_adapter_features` flattens nested analyzer outputs without a length budget
For an analyzer that returns a per-token attribution dict the adapter
will produce one numeric column per key, which the validator then
discards. The wasted JSON / dict allocation work is non-trivial. Add
`MAX_FEATURES_PER_ADAPTER = 64` and log a warning when truncated.

---

## 6. GPU / device handling

### 6.1 — `emotion_intensity_features._model` is never moved to CUDA
`_model = AutoModelForSequenceClassification.from_pretrained(...)` runs
on CPU regardless of whether a GPU is present. On the standard 7-class
DistilRoBERTa head the GPU vs CPU difference is ~12× per batch. Add:
```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model.to(DEVICE).eval()
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
```
and **also** wrap inference in `torch.inference_mode()` (slightly faster
than `no_grad`).

### 6.2 — No `torch.set_num_threads` budget for the CPU fallback
On a Linux container with 8 visible CPUs PyTorch will spawn 8 threads by
default, which competes with FastAPI's worker pool. In `api/app.py`'s
boot sequence (or in `feature_bootstrap`), call
`torch.set_num_threads(min(4, os.cpu_count()))`.

### 6.3 — `_transformer_emotions_batch` re-tokenises and re-pads even in the single-sample path
The single-sample path delegates to the batched path (line 111). That
is correct, but the batched path always pads to the longest of `len(texts)`
sequences and creates a `B × L` attention mask. For B=1 this is the
worst case — pad to the actual sequence length. Add
`if len(texts) == 1: padding=False`.

### 6.4 — No mixed-precision (autocast) path
`_transformer_emotions_batch` runs in fp32 even on GPUs that support
`torch.autocast("cuda", dtype=torch.float16)`. For a softmax-classifier
this is a free 2× speedup with no quality loss. Wrap the forward pass
in `with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE=="cuda"):`.

### 6.5 — No batch-size cap → OOM risk
`_transformer_emotions_batch` accepts an unbounded list. With B=512 and
L=512 the activation memory of DistilRoBERTa is ~9 GB. Add
`MAX_BATCH = 64` and split internally; the caller (`extract_batch`) is
already vectorised and will not notice the chunking.

### 6.6 — `transformers` import path is `try`-guarded but the absence flag is set per-module
`TRANSFORMER_AVAILABLE` is a module-level boolean set at import. There
is no way to monkey-patch it from a test without reaching into the
module. Move to `feature_config` and read at extract time.

---

## 7. Cache layer

### 7.1 — `cache/feature_cache.py` uses `pickle.HIGHEST_PROTOCOL` on disk with no version tag
The pickle blobs include neither a feature-schema fingerprint nor a
codebase version. Any change to a `dataclass` field name silently breaks
read-back across restarts. Add a 16-byte header `b"TLENS\x01" + sha256(SCHEMA)[:8]`
and refuse to load mismatching blobs.

### 7.2 — `cache/cache_manager` LRU eviction is keyed on insertion order, not access order
A read-heavy access pattern therefore evicts hot keys. Use
`collections.OrderedDict.move_to_end(key)` on every `get` (the file
already uses an `OrderedDict`).

### 7.3 — `_context_key` lossy `default=str` (also flagged in §1.10)
See §1.10. The cache layer's hashing is the primary determinant of hit
rate, so this fix has the highest leverage of any single change in the
cache subsystem.

### 7.4 — Lexicon fingerprint is computed but unused
`cache_manager` computes a SHA-256 of the loaded lexicons at module
import and never uses it. It should participate in the cache key, so
that updating a lexicon invalidates the cache automatically.

### 7.5 — No size cap on per-context `ctx.cache`
`base/base_feature.FeatureContext.cache` is an unbounded dict shared
across the lifetime of a request. Long-lived requests (e.g. websocket
streaming) accumulate intermediate vectors and never release them.
Add `weakref` or move large intermediates to `ctx.shared` (which the
batch pipeline already disposes of).

### 7.6 — No metrics on cache hit ratio
Neither the in-memory cache nor the on-disk cache exports a counter.
For an audit run we cannot tell how often the cache is helping. Wire
a `prometheus_client.Counter` into both layers — five lines of code
each.

---

## 8. Dead / unused code

| Path | Why it is dead |
|---|---|
| `feature_pruning.py` | This is an explicit shim that re-exports `fusion/feature_reduction.py`. Either delete or convert to a one-line `from src.features.fusion.feature_reduction import *  # noqa`. |
| `pipelines/feature_schema.py` | Same — shim re-exporting `feature_schema.py`. |
| `fusion/feature_selection.py` | Shim re-exporting `feature_reduction.py`. |
| `bias/bias_lexicon.py` | Defines features that no extractor calls; the schema does not include any of its keys; nothing imports it. Either fold into `BiasFeaturesV2` or delete. |
| `propaganda/propaganda_lexicon_features.py` `BANDWAGON_PHRASES` / `SLOGAN_PHRASES` | Both are `[...]` (Ellipsis lists). The phrase-hits loop runs but always returns 0. |
| `tfidf_engineering.tfidf_matrix` | Re-fits the vectoriser per call; not referenced from any pipeline file. Delete or expose `TfidfEngineering.matrix()`. |
| `importance/feature_ablation.py`, `permutation_importance.py`, `shap_importance.py` | Documented as offline-only; not consumed by `src/inference/` or `api/app.py`. Move out of `src/features/` (see §3.7). |
| `narrative/narrative_features.py` | Superseded by `narrative_role_features.py` + `conflict_features.py`. |
| `analysis/analysis_adapter_features.AnalysisDiscourseCoherenceFeature` etc. | Each adapter dynamically `importlib.import_module(...)`. If the analyser package is missing, `_load_failed=True` and the extractor silently emits `{}` forever (it is never retried). Consider a `--strict-features` flag that turns the silent failure into an exception in CI. |

---

## 9. Configuration drift (`src/config/`, `src/features/feature_config.py`)

### 9.1 — `feature_config` flag `enable_emotion_transformer` is not consulted by the transformer extractor
`emotion/emotion_intensity_features.TRANSFORMER_AVAILABLE` is set at
import time from a `try/except` and never re-read. The config flag has
no effect at runtime. Read it lazily inside `_transformer_emotions_batch`.

### 9.2 — Hard-coded model name `j-hartmann/emotion-english-distilroberta-base`
`emotion_intensity_features.py:39`. Should live in
`src/config/models.yaml` (or `feature_config`) so it can be swapped per
deployment without a code change. Same for `en_core_web_sm` referenced
from `text/syntactic_features`, `narrative/narrative_role_features`,
`emotion/emotion_target_features`.

### 9.3 — `feature_config.FeatureConfig` defaults silently differ from the bootstrap defaults
`feature_bootstrap.py` constructs feature pipelines with hard-coded
parameters (`top_k`, `cache_size`, `pca_components`); the dataclass
defaults in `feature_config.py` carry different numbers. Whichever the
caller forgets to pass wins. Fix: have `feature_bootstrap.py` consume
`FeatureConfig()` directly.

### 9.4 — Lexicon paths are not in `src/config/`
There is no central place where the ~50 lexicon families live. They
are inlined as Python constants (and right now most are `{...}`).
Move to JSON / YAML in `src/config/lexicons/` and load via a single
`load_lexicon(name)` helper — this is the natural fix for §1.1 too.

### 9.5 — `analysis_adapter_features` analyzer class names are hard-coded strings
`module_path` and `analyzer_class` are string literals in the dataclass
default. Move the (analyzer → adapter) registry to
`src/config/analysis_adapters.yaml` so the adapter list can grow
without code edits.

---

## 10. Multi-task interaction

### 10.1 — `partition_feature_sections` cannot route the keys the extractors actually emit
The router in `pipelines/feature_pipeline.py` matches on prefixes
defined in `feature_schema.FEATURE_SECTIONS` (`narrative_role_*`,
`propaganda_*`, `frame_*`, etc.). Because of the schema drift in §1.2,
the router silently classifies most extractor output as "unknown" and
drops it from the per-task heads. Even when the column survives into
the model matrix, it never reaches the right task head.

Concrete examples:
- `disc_*` keys are dropped — discourse head sees zero features.
- `lex_*`, `tok_*`, `syn_*`, `sem_*` keys are dropped — text head sees
  zero features.
- `manipulation_*` keys are dropped — propaganda head loses 12 columns.

This is the highest-impact downstream consequence of §1.2 because it
silently disables entire task heads at training time.

### 10.2 — `narrative_role_features.narrative_*` collides with the data-contract label columns
`feature_schema.py` lines 130–134 explicitly call this out: feature
names use the `narrative_role_*` prefix and the `_ratio` suffix to
disambiguate from the `hero/villain/victim` *label* columns in
`src/data_processing/data_contracts.CONTRACTS["narrative"]`. The
collision is real — `narrative/narrative_features.py:138-140` emits
plain `narrative_hero` / `narrative_villain` / `narrative_victim`
which **do** clash with the label namespace. If both end up in a
training DataFrame the column is interpreted as either feature or
label depending on which join wins. Fix is part of §1.3 (delete or
rename `narrative_features.py`).

### 10.3 — Cross-task feature leakage via `ctx.shared`
`text/lexical_features.py:84` writes `lex_entropy` into
`ctx.shared`. That bucket is shared across **all** extractors in the
batch. If two task heads share the same context (which they do —
`feature_pipeline` builds one context per row and runs every extractor
against it), one task's intermediate value can be read by another.
This is desirable for pure fan-in (the propaganda head reading the
canonical caps ratio) but undesirable for task-specific intermediates.
Add a per-extractor namespace: `ctx.shared.setdefault("lex", {})["entropy"] = ...`.

### 10.4 — Multi-task scaling is fit once across all tasks
`fusion/feature_scaling.py` fits a single `RobustScaler` on the
concatenated feature matrix. A handful of features (`tok_length_log`,
`graph_nodes_log`) have heavy tails; they pull the scaler's median /
IQR estimates, which then mis-scales the per-task heads that don't use
them. Either fit a per-section scaler (mirroring `FEATURE_SECTIONS`)
or apply `np.log1p` consistently before scaling.

---

## 11. Edge cases and robustness

### 11.1 — Empty-text branches return inconsistent shapes
`discourse_features.extract` returns `{}` on empty text;
`lexical_features.extract` returns `_empty()` (a fixed-key dict);
`emotion_intensity_features.extract` returns `_empty()`. The pipeline
validator zero-fills the missing keys, but the inconsistency means a
single extractor can break the per-row dict cardinality across
batches and trip downstream column-order checks. Standardise: every
extractor must return its declared key set, zero-filled, on empty
input. (`narrative_role_features` already does this; use it as the
template.)

### 11.2 — `text/syntactic_features` divides by `n = len(tokens) or 1` but `coord_ratio` is then bounded by `MAX_CLIP=1.0`
For a 1-token document the ratio is `0/1 = 0`, which is fine. But the
fall-back path `_extract_fallback` returns a partial schema (only
two keys) that the validator then has to fill — and the fill value
(zero) collides with the legitimate zero from spaCy. Combined with the
new `syn_spacy_available=0/1` indicator (line 206) this is fine, **but
only if every consumer of `syn_*` knows to gate on it**. Add a runtime
assertion in the bootstrap that any extractor advertising an
`*_available` indicator is paired with a downstream gate.

### 11.3 — `narrative/narrative_role_features._entity_density` divides by `len(doc)` which counts whitespace tokens
`len(doc)` for a spaCy `Doc` includes whitespace tokens. The ratio is
therefore lower than the natural "entities per content token" you'd
expect. Use `sum(1 for t in doc if not t.is_space)`.

### 11.4 — `graph/*` `max_edges` formula assumes simple undirected graphs
`max_edges = nodes * (nodes - 1) / 2` is correct for simple undirected
graphs, but the entity / interaction builders may produce multigraphs
(co-mention with weight). For a multigraph the density saturates
above 1.0 and is then clipped by `_safe`. Either deduplicate edges in
the builder or compute density per edge type.

### 11.5 — `emotion/emotion_intensity_features._hybrid_emotions` mixes lexicon + transformer with a *fixed* `alpha=0.7`
`alpha` is a constant. There is no calibration step that fits `alpha`
to validation data. A learnable scalar (one parameter) or a logistic
gate on token coverage would be a free quality win.

### 11.6 — `emotion/emotion_trajectory_features` synthesises a duplicate vector when the document has one sentence
Line 138–139 duplicates the only vector to keep `np.diff` defined. The
emitted `emotion_traj_volatility` is therefore identically zero for
single-sentence inputs, but the rest of the columns look like they
came from a longer document. Emit `emotion_traj_n_sentences=1` (or
gate via an availability indicator like §3.6) so downstream models can
distinguish.

### 11.7 — `feature_pruning` (shim) prunes by variance but variance is computed before scaling
Variance-based pruning before `RobustScaler` will always drop the
small-magnitude ratios (`disc_*`, `frame_*`) and keep the large
log-magnitude features (`tok_length_log`, `graph_nodes_log`). Either
prune **after** scaling, or use `coefficient_of_variation = std / |mean|`.

### 11.8 — `analysis/analysis_adapter_features.cache_key` collides across instances
All four adapter instances use a different `cache_key`, but
`_BaseAnalysisFeature` does not include the analyser version in the
key. If you upgrade `src/analysis/framing_analysis.FramingAnalyzer`
without touching the adapter, the cache will return stale outputs.
Add `cache_key: str = f"{analyzer_class}@{analyzer_version}"` and
require analysers to expose `version`.

### 11.9 — `propaganda_lexicon_features.extract` lowercases `text` then calls `ensure_tokens_word(context, text)`
`text.lower()` produces a string that no longer matches `context.text`,
so `ensure_tokens_word` re-tokenises (its cache key is the context's
text). The rest of the pipeline reads `context.tokens_word`, which is
the **un-lowered** version. Net: this extractor sees a different token
list than the rest of the pipeline. Lowering should happen inside
`tokenize_words`, not at the call site.

### 11.10 — `bias/bias_lexicon._TOKEN_PATTERN` drops Unicode letters
Same root cause as §1.11 — the local regex is ASCII. Use the canonical
helper.

---

## 12. What is verifiably correct

The audit is not all bad news. The following components are well-designed
and used consistently:

- `base/text_signals.get_text_signals` — single canonical computation of
  caps ratio, exclamation density, question density, headline weighting
  with NER masking. The propaganda family already reads from it
  correctly. (Three other extractors still recompute it, see §4 — but
  the helper itself is sound.)
- `base/lexicon_matcher.LexiconMatcher` — vectorised `np.isin` per
  category with a precompiled token array. The right primitive; the
  fix in §2.2 is "use it everywhere".
- `base/tokenization.ensure_tokens_word` — Unicode-aware, cached on the
  context, contractions handled. This is the canonical token source
  and most extractors honour it.
- `base/spacy_loader.get_shared_nlp` — single shared spaCy instance per
  model name. Avoids the multi-MB pipeline reload cost.
- `base/numerics.normalized_entropy` — a single, correct entropy helper
  that handles the zero-probability case. Reused everywhere.
- `emotion/emotion_intensity_features._transformer_emotions_batch` —
  the right batching pattern (one tokenize, one forward, one host
  copy). It is the template for §6.4 mixed-precision and §6.5 batch
  capping.
- `text/syntactic_features._memoized_dependency_depths` — the
  amortised O(N) depth memo is well-implemented; just needs to outlive
  one call (§2.3).
- `feature_schema.assert_schema_consistency()` — exists, is called from
  `feature_bootstrap`, currently logs rather than raises. Once §1.2 is
  fixed, flipping it to `raise` is a one-line change that prevents
  future drift.
- `cache/feature_cache.py` atomic-write path — uses `os.replace` after
  writing to a `.tmp`. Crash-safe and process-safe.
- `pipelines/batch_feature_pipeline.py` — correctly threads
  `extract_batch` through `safe_extract_batch` with per-sample
  validation, so a single bad sample does not poison the batch (this
  is the right policy; the §1.5 finding is about the swallowing in
  `feature_fusion`, not here).
- `narrative/narrative_role_features.py` — the only narrative extractor
  whose output keys actually match `feature_schema.NARRATIVE_FEATURES`.
  Use it as the template for the rest of the rename in §1.2.
- `bias/bias_features.BiasFeaturesV2` output keys (modulo the dead
  lexicons §1.1) match `feature_schema.BIAS_FEATURES`. Same template
  role as above.
- The architectural decision to keep `importance/*` "OFFLINE-ONLY" with
  loud module-level warnings is correct (the only fix needed is to
  move them outside `src/features/`, §3.7).
- `analysis/analysis_adapter_features` — the lazy `importlib.import_module`
  pattern is the right way to bridge the `src/analysis/` analyzers
  without forcing them into `feature_registry`. Need the strict-mode
  flag (§8 last bullet) and the version key (§11.8) to be production-grade.

---

## 13. Overall production-readiness score

**4.5 / 10.**

Justification:

- The architecture (`base/`, `cache/`, `fusion/`, `pipelines/`) is
  sound and shows real engineering investment — vectorised lexicon
  matchers, a shared spaCy instance, a shared text-signals cache, an
  atomic-write disk cache, an availability-indicator pattern for
  optional dependencies, batched transformer inference. Floor: 6.0.

- However, the **data plane** is broken in three independent ways
  that each, on their own, would justify blocking a release:
  1. ~60 feature columns are identically zero in production (§1.1
     — placeholder lexicons).
  2. The schema declared as the single source of truth disagrees with
     the keys ~80% of extractors actually emit (§1.2 — schema drift).
  3. The multi-task router silently drops most of the columns the
     extractors do produce, because it routes by the schema names that
     don't exist at runtime (§10.1).
  Each of those drops the score by ~1 point. Floor: 3.0.

- A handful of partial credits: the fixes are mechanical (rename
  keys, ship lexicon JSON, flip the schema check to `raise`) and a
  large fraction of the extractors are already correctly batched and
  vectorised. Add 1.5. Final: **4.5**.

Top-3 fixes, in priority order, that would move the score above 7.5:

1. Replace every `LEXICON = {...}` with a real word list loaded from
   `src/config/lexicons/`. Add the empty-lexicon CI guard from §1.1.
2. Mass-rename extractor output keys to match `feature_schema.py`
   (or extend the schema where the extractor's keys are genuinely
   richer), then turn `assert_schema_consistency()` into a hard
   `raise`. This unblocks the multi-task router.
3. Move `importance/*` out of `src/features/`, delete the three
   shim modules, and consolidate the duplicated narrative / propaganda
   / framing extractors per §4.

After those three changes the codebase has the right architecture and
the right data — the rest of this report is performance polish (§2,
§6) and configuration hygiene (§9).
