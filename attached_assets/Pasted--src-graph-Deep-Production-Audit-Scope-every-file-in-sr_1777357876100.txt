# `src/graph/` — Deep Production Audit

Scope: every file in `src/graph/` plus every cross-module touchpoint with
`src/data`, `src/features`, `src/analysis`, `src/models`, `src/config`.

Tag convention: `G-K*` keying / schema, `G-S*` semantic, `G-T*` token-/entity-
alignment, `G-P*` performance, `G-R*` recomputation / reproducibility, `G-E*`
edge-cases, `G-V*` visualization, `G-D*` dead/duplicated code, `G-CFG*` config,
`G-C*` schema/contract. New findings are introduced below; prior `G-*` fixes
that I re-verified are listed in §10.

Files audited (3 906 LOC total):
`entity_graph.py` (307), `graph_analysis.py` (353), `graph_config.py` (306),
`graph_embeddings.py` (406), `graph_explainer.py` (261), `graph_features.py`
(297), `graph_pipeline.py` (547), `graph_schema.py` (222), `graph_utils.py`
(222), `graph_visualization.py` (278), `narrative_graph_builder.py` (489),
`temporal_graph.py` (218), `__init__.py` (0).

---

## 1. CRITICAL bugs

### G-K2 — `extract_from_graphs` strict-merge always crashes when both graphs are enabled
**File:** `src/graph/graph_features.py` lines 184-228 + `graph_analysis.py` line 271 + `graph_pipeline.py` lines 386-407.

The pipeline calls

```python
features = self.graph_feature_extractor.extract_from_graphs(
    entity_graph=entity_graph,
    narrative_graph=narrative_graph,
    entity_metrics=entity_metrics,        # GraphAnalyzer.analyze(entity_graph).to_dict()
    narrative_metrics=narrative_metrics,  # GraphAnalyzer.analyze(narrative_graph).to_dict()
)
```

`GraphAnalyzer.analyze` is a single generic analyzer — it returns the same
key set for **every** graph it sees: `graph_nodes`, `graph_edges`,
`graph_avg_degree`, `graph_density`, `graph_centralization`,
`graph_clustering`, `graph_entropy`, … (12 keys, all prefixed `graph_`).

Inside `extract_from_graphs` the four blocks appended in order are:

| block | source | sample keys |
|------|--------|-------------|
| 1 | `entity_features` (`EntityGraphFeatures.to_dict()`) | `nodes`, `edges`, `avg_degree`, `density`, `clustering_coeff` |
| 2 | `entity_metrics` | `graph_nodes`, `graph_edges`, `graph_density`, … |
| 3 | `narrative_features` (`NarrativeGraphFeatures.to_dict()`) | `narrative_graph_nodes`, … |
| 4 | `narrative_metrics` | `graph_nodes`, `graph_edges`, `graph_density`, … |

Blocks 2 and 4 share the **complete** key set. `merge_feature_blocks_strict`
(`graph_features.py:102`) raises `ValueError("Duplicate feature key: graph_nodes")`
on the very first key.

Default `config/config.yaml` has `enable_entity_graph: true` and
`enable_narrative_graph: true`, so **every request** with non-empty graphs
hits this exception, gets caught one frame up by
`feature_pipeline._merge_graph_features` (`feature_pipeline.py:179`), the
`graph_merge_failures` counter is incremented, and the model receives an
**all-zero graph signal** for the rest of the run — the exact silent-fallback
mode that prior audit fix `#1.8` was supposed to surface but which this bug
re-creates one layer down. The `_merge_graph_features` warning is logged at
WARNING level only, so this would only be visible in production by counting
that metric.

This is the single highest-impact bug in the layer: every graph feature the
model expects (per `src/features/feature_schema.py` and the fusion model
input width) is silently zero in production.

### G-K1 — `GRAPH_PIPELINE_FEATURES` schema keys don't match the producer
**File:** `src/features/feature_schema.py:205-210` + `src/features/pipelines/feature_pipeline.py:170-177`.

The schema declares the four contract keys the model expects from the graph
pipeline:

```python
GRAPH_PIPELINE_FEATURES = [
    "graph_pipeline_entity_density",
    "graph_pipeline_entity_centralization",
    "graph_pipeline_narrative_flow",
    "graph_pipeline_narrative_coherence",
]
```

The producer (`_merge_graph_features`) actually emits:

```python
features[f"graph_pipeline_entity_{k}"]    # k from entity_metrics
features[f"graph_pipeline_narrative_{k}"] # k from narrative_metrics
```

`entity_metrics` keys are already `graph_*`-prefixed (`graph_density`,
`graph_centralization`, …) so the **actual** keys produced are:

* `graph_pipeline_entity_graph_density`     (double `graph_`)
* `graph_pipeline_entity_graph_centralization`
* `graph_pipeline_narrative_graph_density`
* …

`narrative_flow` and `narrative_coherence` are **not produced anywhere in the
codebase** — neither `compute_graph_metrics`, nor `NarrativeGraphFeatures`,
nor `extract_from_graphs` ever sets them. Even if G-K2 were fixed, all four
expected schema slots would default to 0.0 because the producer/consumer
contract is broken on both sides (prefix doubling **and** non-existent metric
names).

Combined with G-K2 above, the graph layer contributes **zero non-zero feature
values** to the model in the default configuration — even though every
analyzer downstream of it runs to completion and reports success.

---

## 2. Performance bottlenecks

### G-P8 — `run_batch` does not actually batch the narrative graph
**File:** `src/graph/graph_pipeline.py:308-334` + `narrative_graph_builder.py:259-260`.

Docstring claim: *"Narrative / temporal stages still run per-doc — they're
pure Python regex"*. This is **stale** — `_sentence_keywords_spacy` was
introduced (G-S1 / G-T2) and now invokes `self.nlp(text)` once per document
inside the narrative builder. Since `narrative_graph_builder.nlp` is the
shared spaCy instance loaded by `get_shared_nlp`, the model itself is shared,
but the parse is `Language.__call__`, not `Language.pipe` — so for a batch of
N documents the narrative stage does N separate parsing calls instead of one
batched `nlp.pipe` call.

The pipeline already parses each doc once for the entity stage via
`nlp.pipe` and shares the `Doc` via `_entity_graph_from_doc`. The narrative
builder cannot accept a pre-parsed `Doc` (its public surface is text-only),
so the batch optimisation gains roughly **half** of what it claims. For a
batch of 32 articles this is the dominant CPU cost in the graph layer.

Recommended path: add `build_graph_with_doc(doc)` to `NarrativeGraphBuilder`
that mirrors the same factoring `EntityGraphBuilder` already has for the
pipeline, and have `_run_with_doc` pass the shared `doc` to it.

### G-P5 — `_node2vec` allocates an `O(N²)` dense transition matrix
**File:** `src/graph/graph_embeddings.py:264-299`.

```python
mat = np.zeros((len(nodes), len(nodes)))     # N × N float64
for walk in walks:
    for i in range(len(walk) - 1):
        mat[vocab[a], vocab[b]] += 1
vec = np.mean(mat, axis=0)
```

For a 500-node graph that's a 2 MB allocation per request, then averaged to
an N-vector that gets padded/truncated to `embedding_dim=16`. The output is
unrelated to the actual node2vec algorithm (no Skip-Gram, no negative
sampling, no representation learning) — it's a mean transition frequency
fingerprint mislabelled as *"node2vec"*. The dense matrix could be a sparse
counter or — more honestly — the whole `_node2vec` branch should be either
implemented with `gensim.Word2Vec(walks)` or removed and the option dropped
from the config enum. Currently it's misleading callers who think
`embedding_type: "node2vec"` gives them learned embeddings.

### G-P6 — `extract_features(text)` skips the canonical-once optimisation
**File:** `src/graph/graph_features.py:140-154`.

The text-only legacy path runs:

```python
entity_graph = self.entity_builder.build_graph(text)        # canonical (entity builder canonicalizes)
narrative_graph = self.narrative_builder.build_graph(text)  # RAW, asymmetric, possibly with both directions
return self.extract_from_graphs(entity_graph, narrative_graph)
```

`extract_from_graphs` then passes `narrative_graph` straight into
`extract_graph_features(narrative_graph)`, where iteration `for src, nbrs in
graph.items(): for tgt, w in nbrs.items()` counts each directed edge
separately — this is the source of the narrative double-counting in §3
(G-S4). The pipeline path canonicalises at the top (G-P1) and is fine; the
direct path is not.

### G-P7 — Per-scalar block allocation for embedding features
**File:** `src/graph/graph_features.py:201-206`.

```python
for i, val in enumerate(emb):
    blocks.append({f"graph_embedding_{i}": float(val)})
```

For default hybrid (16 floats) this allocates 16 dicts and forces 16
duplicate-key checks inside `merge_feature_blocks_strict`. Trivial perf hit
but cosmetic. Replace with one dict comprehension.

---

## 3. Semantic / numerical correctness issues

### G-S4 — Narrative graph edges and density are systematically 2× inflated
**File:** `src/graph/narrative_graph_builder.py:359-431`.

After the pipeline runs `canonicalize_weighted` on the narrative graph,
every undirected edge is present as both `(a, b)` and `(b, a)` (the
canonical form is symmetric by design). `extract_graph_features` then does:

```python
edges = set()
for src, nbrs in graph.items():
    for tgt, w in nbrs.items():
        if src != tgt:
            edges.add((src, tgt))
            weights.append(w)
e = len(edges)                          # 2 × |E_undirected|
density = e / (n*(n-1))                 # 2 × correct
narrative_graph_avg_degree = mean(degrees)  # already correct since both directions present
```

The entity builder (`entity_graph.py:243`) handles this with
`e = sum(len(v) for v in g.values()) // 2` — narrative builder forgot the
`// 2`. Every emitted `narrative_graph_edges` and `narrative_graph_density`
is double; `narrative_graph_entropy` is inflated by `+log(2) ≈ 0.693`
because the weight list contains every weight twice.

### G-S6 — `_apply_temporal_weight` is a no-op due to subsequent L2 normalisation
**File:** `src/graph/graph_embeddings.py:305-319` + `:380`.

```python
vec = self._apply_temporal_weight(vec, temporal_features)  # vec * scale
...
vec = self._normalize(vec)                                  # vec / ||vec||
```

`_normalize` divides by `np.linalg.norm(vec) + EPS`. Multiplying every
element by a scalar `(1 + drift)` and then L2-normalising **erases** the
scaling — the post-normalisation vector is identical regardless of `drift`.
The temporal-aware embedding feature is silently inert.

Either drop temporal scaling entirely or move it after normalisation, or
incorporate `drift` as a non-uniform feature (e.g., concatenate it as one
extra dim).

### G-S8 — Narrative block all-or-nothing in `extract_feature_vector_from_features`
**File:** `src/graph/graph_features.py:275-287`.

```python
narrative_keys = [
    "narrative_graph_nodes", "narrative_graph_edges",
    "narrative_graph_avg_degree", "narrative_graph_density",
    "narrative_graph_isolated_nodes", "narrative_graph_components",
]
if all(k in features for k in narrative_keys):
    vectors.append(np.array([features[k] for k in narrative_keys], ...))
```

If even one of the six keys is missing (e.g., narrative builder failed for
this doc, was disabled, or strict-merge crashed per G-K2), the **entire
6-element narrative slot is dropped** and the resulting vector is 6 elements
shorter than expected. The downstream model's first dense layer will then
either crash on dim mismatch or — worse — pick up the embedding block as if
it were the narrative block. The vector contract must be fixed-shape; missing
keys should pad with zeros, not be omitted.

Note also: the three "new" narrative metrics introduced in
`NarrativeGraphFeatures` (`narrative_graph_entropy`,
`narrative_graph_centralization`, `narrative_graph_flow_strength`) are
**never written to the vector** — they only ever appear in the dict
representation. The vectoriser is out of sync with the dataclass.

### G-S9 — L2 normalisation across heterogeneous feature blocks
**File:** `src/graph/graph_features.py:294-296`.

`_normalize_vector` runs a single L2 norm across {entity counts (raw 0–N),
entity metrics (0–1 normalised), embedding (already normalised),
narrative counts (raw 0–N)}. The relative scale between blocks is destroyed
— any block with large absolute values dominates the unit vector. The
"normalisation" makes the model less, not more, robust to graph size.
Recommended: per-block normalisation, or no normalisation and rely on the
batch-norm in the fusion head.

### G-S5 — Narrative succession edges lose count via `max` symmetrisation
**File:** `src/graph/narrative_graph_builder.py:331-335` + `graph_analysis.py:92-104`.

`canonicalize_weighted` symmetrises with `max(w_uv, w_vu)`. The entity
builder writes only one direction so this is a no-op for it. The narrative
builder writes both directions (intra-sentence one-direction *and* succession
both-direction-possible across sentences) — when `obama→putin` accumulates 3
co-occurrences and `putin→obama` accumulates 2, the canonical edge weight is
3, not 5. The information loss is silent and design-driven; consider
documenting the choice or switching to `max ↔ sum` per-builder via a
canonicalize policy flag.

### G-S11 — `narrative_graph_flow_strength` is mean weight, not flow
**File:** `src/graph/narrative_graph_builder.py:418`.

`flow_strength = float(np.mean(weights))`. Mean edge weight is not a flow
metric — flow strength in temporal narrative graphs is conventionally the
mean edge weight along the **temporal succession path** (sum of `i→i+1`
weights divided by sentence count). The metric as implemented is
indistinguishable from average co-occurrence intensity. Cosmetic but feeds
into the model under a misleading name.

### G-K3 — Two parallel naming conventions for the same entity metrics
**File:** `src/graph/entity_graph.py:65-77` + `graph_analysis.py:271-284`.

`EntityGraphFeatures` exposes `nodes / edges / avg_degree / density /
dominant_degree / degree_variance / clustering_coeff / centrality_mean`.
`compute_graph_metrics` exposes `graph_nodes / graph_edges /
graph_avg_degree / graph_density / graph_max_degree / graph_degree_variance
/ graph_clustering / graph_centrality_mean`. These are the same eight
metrics under two naming schemes, both included in the feature dict (when
G-K2 isn't crashing). The model receives effectively duplicate features
under different keys — wasted input bandwidth and potential collinearity for
linear heads. Pick one.

---

## 4. Token / entity alignment

### G-T4 — `TemporalGraphAnalyzer` uses regex tokens that don't align with entity ids
**File:** `src/graph/temporal_graph.py:59-61`.

```python
def _extract_entities(self, sentence: str) -> Set[str]:
    tokens = re.findall(r"\b[a-zA-Z]+\b", sentence.lower())
    return {t for t in tokens if len(t) >= self.min_token_length}
```

This is the same hand-rolled regex tokenizer that prior fix G-S1 / G-T2
*explicitly removed* from `NarrativeGraphBuilder` because it disagrees with
the spaCy tokenizer used by every other layer. As a result:

* `entity_recurrence`, `topic_shift_score`, `narrative_drift`, … in the
  temporal feature block measure **raw token recurrence**, not entity
  recurrence.
* The temporal metrics' "entity sets" do not intersect with the
  `entity_graph` node set — `obama` shows up in temporal metrics, `Barack
  Obama` shows up in the entity graph. The two layers cannot be cross-
  referenced.

This is the same class of bug that motivated the narrative-graph rewrite;
it remains present in the temporal layer. Re-use the shared spaCy pipeline
the same way — preferably accept the already-parsed `Doc` from
`_run_with_doc` so the parse is shared.

### Verified-OK alignment (entity / narrative)
* Entity builder uses `nlp.pipe` and emits per-mention spans (G-T1).
* Narrative builder uses noun-chunks + ents from the shared spaCy and emits
  per-keyword spans + a `tokenizer` discriminator (G-S1 / G-T2).
* `entity_spans` / `narrative_spans` / `narrative_tokenizer` are surfaced
  in the result dict — so the API layer can map node IDs back to
  highlightable text regions.

---

## 5. GPU / memory usage

The graph layer is intentionally CPU-only and there is no GPU utilisation
inside it. The `to_pinned_tensor` helper (G-G1) correctly uses
`torch.cuda.is_available()` to gate `pin_memory()` and swallows the
`RuntimeError` on pure-CPU builds — verified no leakage on the current
torch 2.5.1 CPU install.

* No tensors are kept on device past their use scope.
* No `Doc` objects are retained between calls (Python ref count drops at
  the end of `_run_with_doc`).

Memory issue worth flagging: see G-P5 for the dense N×N node2vec
allocation; for a misinformation feed of 1k 500-node articles that's 2 GB
peak per worker.

---

## 6. Recomputation

### G-R3 — `_node2vec` random walks are not seeded
**File:** `src/graph/graph_embeddings.py:281` (`np.random.choice(neighbors)`).

Uses the global numpy RNG. Two consecutive calls on the same graph return
different embeddings, breaking the cache key in
`batch_feature_pipeline._build_cache_key` (which assumes deterministic
features per `(text, config_fingerprint)` pair). Either seed via
`np.random.default_rng(seed=hash(graph_text))` or remove the option as
suggested in G-P5.

### Verified-OK
* Canonical graph is computed once at the top of `_run_with_doc` (G-P1) and
  passed to every downstream consumer.
* `entity_metrics` / `narrative_metrics` are computed once and passed to
  `extract_from_graphs` as kwargs (G-R2), avoiding a second
  `GraphAnalyzer.analyze` pass per graph.
* `GraphExplainer.explain` accepts a precomputed `temporal_features`
  parameter so the temporal analyzer is not re-invoked when called from
  the pipeline (G-C3).
* Singleton `_DEFAULT_PIPELINE` (G-R1) is honoured by every callsite I
  found (`api/app.py`, `feature_preparer.py`, `feature_pipeline.py`,
  `inference_pipeline.py`, `truthlens_pipeline.py`, `analyze_article.py`,
  `batch_inference.py`).

---

## 7. Unused / dead modules

### G-D2 — `src/graph/graph_utils.py` is not imported by any other file
Verified via `rg "from src.graph.graph_utils|graph_utils\."`. The
canonicalisation surface in `graph_analysis.py` (`canonicalize_weighted`)
already does everything the helpers in `graph_utils.py` do, **and does it
without the double-counting bug that `to_undirected_weighted` re-introduces**
(`undirected[u][v] += w; undirected[v][u] += w` against the same input
inflates each edge by 2× — exactly the bug fixed in canonicalize_weighted).
Either remove the file or wire it through and deprecate
`canonicalize_weighted`.

### G-D3 — `src/graph/graph_visualization.py` is not consumed in production
No production caller imports `GraphVisualizer`. Worse, see G-V1 — it cannot
even accept the canonical graph format the rest of the layer produces.
Recommended action: move under `tools/` or remove.

### G-D1 — `_entity_graph_from_doc` duplicates `EntityGraphBuilder.build_graph_with_spans`
**File:** `src/graph/graph_pipeline.py:243-287` vs `entity_graph.py:122-213`.

The per-doc factoring necessary for the `nlp.pipe` batch path was
re-implemented inline in the pipeline. The two implementations are
identical except for the `if not text.strip()` validation. Future
maintainers have to keep both in sync; the obvious fix is to add
`build_graph_with_doc(doc)` to the entity builder (parallel to G-P8's
proposed `NarrativeGraphBuilder.build_graph_with_doc`) and have the
pipeline call it.

### G-D4 — `from src.analysis.integration_runner import AnalysisIntegrationRunner` at module top
**File:** `src/graph/graph_pipeline.py:22`.

Top-level import means any consumer that reaches `src.graph.graph_pipeline`
also drags in `src.analysis.integration_runner` and its 15 analysis
modules at import time, even if `run_analysis_modules: false`. Slow first-
import (the comment in `get_default_pipeline` even acknowledges this). Move
the import inside `__init__` under the `if self.config.run_analysis_modules`
branch.

---

## 8. Config integration issues

### G-CFG4 — `config_fingerprint` does not include builder versions
**File:** `src/graph/graph_pipeline.py:212-225`.

The fingerprint hashes only `GraphPipelineConfig` dataclass fields. It
does not capture:

* spaCy model name in use (`en_core_web_sm` vs blank fallback — these
  produce structurally different graphs).
* The shared `nlp.pipe_names` (a missing `parser` materially changes
  `noun_chunks` output).
* Code revision of the builders.

`batch_feature_pipeline.py:38` uses this fingerprint as a cache-invalidation
key. A spaCy model swap silently re-uses cached graphs from the previous
model. Add `nlp.meta["name"]` and `nlp.meta["version"]` (and ideally a
sha-256 of the importable module sources) to the hashed payload.

### Verified-OK
* `GraphConfig` (G-CFG2) carries every tunable the pipeline reads; defaults
  align between `parse_graph_config`, the `GraphConfig` dataclass and
  `GraphPipelineConfig.from_graph_config`.
* `GraphConfigLoader._validate` enforces:
  * `min_keyword_length / max_keywords_per_sentence /
    temporal_min_token_length >= 1`
  * `0 <= min_edge_weight < max_edge_weight`
  * `0 < feature_scale <= 10`
  * `batch_size / spectral_dim / embedding_dim / walk_length / num_walks
    >= 1`
  * Convex-combination check on `explainer_*_weight` (G-CFG2).
  * `embedding_type ∈ {degree, centrality, spectral, hybrid, node2vec}`.
* `load_default_graph_config` falls back to dataclass defaults when
  `config/config.yaml` is missing (G-CFG1).
* `GraphExplainer.__init__` defensively re-validates the convex
  combination so a direct caller bypassing the YAML still trips the
  check.

---

## 9. Edge cases & error handling

### G-V1 — `GraphVisualizer._validate_graph` rejects the canonical weighted graph
**File:** `src/graph/graph_visualization.py:42-47`.

```python
def _validate_graph(self, graph: Dict[str, List[str]]):
    ...
    if not isinstance(k, str) or not isinstance(v, list):
        raise ValueError("Invalid graph format")
```

But every graph in the pipeline today is `Dict[str, Dict[str, float]]`.
Calling `GraphVisualizer.visualize(entity_graph_from_pipeline)` raises
`ValueError`. The visualizer is unreachable from the live pipeline. See
also G-D3.

### G-E5 — `run_batch` lacks per-doc error isolation
**File:** `src/graph/graph_pipeline.py:308-334`.

```python
return [self._run_with_doc(t, d) for t, d in zip(texts, docs)]
```

If any single doc raises in `_run_with_doc` (e.g., explainer crashes on a
malformed graph that slipped past validation), the whole batch returns
`None` and the caller loses the 31 successful results. Wrap each
`_run_with_doc` in `try/except` and return a per-doc `{"error": str}`
sentinel matching what `inference_pipeline._fail_safe` already expects.

### G-E3 — `TemporalGraphFeatures(*([0.0] * 7))` is positional and brittle
**File:** `src/graph/temporal_graph.py:75`.

The early-return for `< 2` sentences uses a positional 7-element zero
expansion. `TemporalGraphFeatures` has 7 slots today; if a future field is
added the constructor still succeeds but values shift silently. Use keyword
construction or a `cls.zeros()` classmethod.

### G-T3 — Singleton initialisation race
**File:** `src/graph/graph_pipeline.py:530-541`.

The comment claims the GIL makes `if _DEFAULT_PIPELINE is None` atomic with
the assignment. Two threads can both pass the check before either
assigns; the loser's `GraphPipeline()` (15 analysis modules + 6 builders)
is GC'd. Not a correctness bug — both pipelines are functionally
equivalent — but a real perf cliff at FastAPI worker startup under
concurrent first requests. A double-checked lock with `threading.Lock`
fixes it cheaply.

### G-S7 — `spectral_eigen_embedding` ordering choice
**File:** `src/graph/graph_embeddings.py:135-150`.

Dense path uses `np.linalg.eigvalsh` (returns ascending real values) and
sorts descending. Sparse path uses `eigsh(..., which="LA")` (largest
algebraic). For adjacency matrices with negative eigenvalues these two
methods return different sets of "top-k". For unweighted symmetric graphs
the largest eigenvalues are positive (Perron-Frobenius) so this is
typically a non-issue, but for narrative graphs with mixed signs it
matters. Document the choice or unify on `which="LM"` (largest magnitude)
which matches typical spectral-embedding semantics.

### G-S10 — Documentation/code mismatch in narrative fallback
**File:** `src/graph/narrative_graph_builder.py:205-220`.

The `_sentence_keywords_spacy` fallback to lemmatised content tokens is
documented as "blank pipeline path" but the trigger condition is `if not
items:` — i.e., it also fires when a real spaCy pipeline produces zero
noun-chunks and zero entities for a sentence. That's not "the blank
pipeline path"; that's a graceful degradation in production too. Update
the docstring.

### G-C6 — Spans / tokenizer not in the `GraphOutput` schema
**File:** `src/graph/graph_schema.py:171-201` + `graph_pipeline.py:493-495`.

`entity_spans`, `narrative_spans`, `narrative_tokenizer` are surfaced via
the result dict only — `GraphOutput` (with `extra="forbid"`) does not list
them. Any consumer that types its input as `GraphOutput` cannot read them;
they only flow as untyped dict keys. Either add them as optional fields on
`GraphOutput` or document explicitly that the typed envelope is a subset
of the result dict.

---

## 10. Verified-OK components (prior fixes still in place)

* **G-CFG1/2/3** — YAML `graph:` block flows end-to-end via
  `GraphConfig` → `GraphPipelineConfig.from_graph_config` →
  `GraphPipeline.__init__`. All knobs honored.
* **G-C1** — `EntityGraphBuilder.extract_graph_features` alias exists; no
  `AttributeError` on first request.
* **G-C2** — Empty node lists allowed by `GraphStructure`; pipeline wraps
  via `_to_graph_structure` and falls back to `Optional` when empty.
* **G-C3** — `GraphExplainer.explain(temporal_features=…)` signature is
  honored by the pipeline; no double `TemporalGraphAnalyzer.analyze`.
* **G-C4** — `entity_metrics` / `narrative_metrics` are first-class fields
  on `GraphOutput`; the consumer key names
  (`entity_graph_metrics` / `narrative_graph_metrics`) are emitted on the
  raw result dict.
* **G-C5** — Canonical graph type is weighted dict-of-dicts; legacy
  `Dict[str, List[str]]` accepted; weights flow through `to_nx`.
* **G-S1 / G-T1 / G-T2** — Narrative graph drives off shared spaCy
  pipeline; per-keyword spans surfaced; `tokenizer` discriminator in
  output.
* **G-S2** — `canonicalize_weighted` is idempotent and uses `max`
  symmetrisation; the historical 4× amplification bug is gone for the
  entity graph.
* **G-P1** — Canonicalisation done once at the top of `_run_with_doc`.
* **G-P2** — Average clustering uses sparse `A^2 ⊙ A`; no `O(N · k²)`
  Python loops.
* **G-P3** — `run_batch` uses `nlp.pipe` for the entity stage (but see
  G-P8 for the missing narrative half).
* **G-P4** — Spectral embedding uses `scipy.sparse.linalg.eigsh` for
  `n > 32`; dense fallback for tiny graphs.
* **G-R1** — Process-wide singleton honoured by every callsite.
* **G-R2** — Pre-computed metrics passed through to
  `extract_from_graphs`; no second analyzer pass per graph.
* **G-E1** — Embeddings always return the configured target dimension
  (4 / 4 / `spectral_dim` / `embedding_dim` / `4+4+spectral_dim`).
* **G-E2** — `temporal_consistency` returns 0 for documents with fewer
  than 2 transitions to match the 1-sentence early-return convention.
* **G-G1** — `to_pinned_tensor` correctly handles the CPU-only build by
  swallowing `pin_memory()` `RuntimeError`.

---

## 11. Final score

**Score: 5.5 / 10**

Justification:

* The **infrastructure work** in this layer is genuinely strong — the
  config layer (G-CFG1/2/3), the canonical-graph contract (G-C5 / G-S2),
  the sparse clustering (G-P2), the sparse spectral path (G-P4), the
  shared-spaCy keyword extraction (G-S1 / G-T1 / G-T2), the singleton
  (G-R1), the precomputed-metrics pass-through (G-R2), the pinned-tensor
  helper (G-G1), and the embedding-dim contract (G-E1) are all correct,
  defensive, and well-documented. If the layer were judged purely on
  these the score would be ~8.5.
* But the **runtime contract with the model is broken end-to-end**:
  * **G-K2** crashes the strict merge for every default request that
    builds both graphs — i.e., every request — and is silently swallowed
    one frame up. The model receives an all-zero graph signal in
    production today.
  * **G-K1** double-prefixes the graph metric keys and references two
    metric names (`narrative_flow`, `narrative_coherence`) that no
    producer in the codebase emits. Even if G-K2 were fixed the four
    schema-declared graph features would all default to 0.
  * **G-S4** double-counts narrative edges and inflates entropy by
    `log(2)`.
  * **G-S6** silently nullifies the entire temporal-aware embedding
    contribution by L2-normalising right after multiplying by the
    temporal scale.
  * **G-S8** drops the entire narrative slot of the feature vector when
    any one of six keys is missing — a recoverable failure becomes a
    schema-shape failure downstream.
  * **G-T4** keeps the regex-token / spaCy-token misalignment that prior
    audit work explicitly removed from the narrative builder still alive
    in the temporal analyzer.
  * **G-P8** invalidates the headline `run_batch` performance claim —
    narrative parsing is per-doc, not batched.

The cumulative effect is that the rich, well-instrumented graph stage
contributes effectively zero correct signal to the model in the default
configuration and a misleading partial signal once G-K2 is patched. Until
the contract bugs (G-K1, G-K2, G-S4, G-S6, G-S8, G-T4) and the run_batch
narrative gap (G-P8) are fixed, the engineering quality of the layer is
substantially ahead of the production behaviour it actually delivers.
