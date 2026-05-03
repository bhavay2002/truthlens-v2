# Graph Layer Audit — `src/graph/`

**Audit scope:** v11 structure + performance review  
**Files audited:** 13 (all files under `src/graph/`)  
**Date:** 2026-05-03  
**Status:** Complete — 1 bug fixed, 2 informational findings documented

---

## File Inventory

| File | Role |
|------|------|
| `graph_pipeline.py` | Singleton orchestrator (`get_default_pipeline`, `run_batch`) |
| `graph_features.py` | Feature normalisation, `to_pinned_tensor` |
| `graph_analysis.py` | `GraphAnalyzer` — centrality, density, clustering |
| `entity_graph.py` | `EntityGraphBuilder` — spaCy NER → networkx graph |
| `temporal_graph.py` | `TemporalGraphAnalyzer` — narrative volatility, entity recurrence |
| `narrative_graph_builder.py` | Sentence-level narrative graph |
| `semantic_graph.py` | Embedding-based cosine edge graph |
| `graph_utils.py` | Utility helpers (canonicalise, symmetrise, degree stats) |
| `graph_visualization.py` | Offline matplotlib/pyvis rendering utilities |
| `graph_clustering.py` | Spectral / modularity clustering |
| `community_detection.py` | Louvain community detection |
| `sparse_graph.py` | Scipy sparse adjacency helpers |
| `__init__.py` | Public re-exports |

---

## Pre-existing Fixes (already in codebase)

The graph layer carries extensive inline audit commentary. The following classes of issue were already resolved before this audit:

| Tag | Description |
|-----|-------------|
| G-D1 | spaCy singleton with threading lock; `nlp.pipe` batch processing |
| G-P1–G-P8 | Performance: batched NLP, sparse adjacency, fixed-dim embeddings, span alignment, pinned tensors |
| G-C2–G-C6 | Correctness: canonicalisation uses `max` symmetrisation, edge-weight overflow guard, empty-graph early returns |
| G-CFG1–G-CFG4 | Config-driven edge weights; no magic numbers in pipeline |
| G-T1–G-T4 | Temporal: Jaccard distance series, entity recurrence, transition-rate normalisation |
| G-E1–G-E5 | Entity extraction: coref-aware deduplication, span-alignment guard |
| G-R1–G-R3 | Resource: process-wide singleton shared across `PredictionPipeline`, `BatchInferenceEngine`, `FeaturePreparer`, `ArticleAnalyzer` |
| G-S1–G-S10 | Semantic: embedding dimensionality fix, cosine-similarity batch vectorisation |

---

## Findings

### GRAPH-FIX-001 — `temporal_graph.py` L263: `temporal_consistency` can go negative *(BUG — FIXED)*

**Severity:** Medium  
**File:** `src/graph/temporal_graph.py`  
**Function:** `TemporalGraphAnalyzer._compute_temporal_features`

**Root cause:**  
`narrative_volatility` is the variance of Jaccard-distance differences between consecutive sentence-level topic sets. Variance is unbounded above zero; for highly heterogeneous documents it frequently exceeds 1.0. The formula

```python
temporal_consistency = float(1.0 - narrative_volatility)
```

produces a negative value in those cases, breaking the `[0.0, 1.0]` contract expected by the aggregation pipeline's weighted sum (`temporal_consistency` feeds directly into `TruthLensScoreCalculator` credibility weights).

**Impact:**  
- Credibility scores silently underflow for long, topic-diverse articles.  
- Any downstream clamp in the aggregation layer would mask the issue, but propagation up to that point would still corrupt intermediate scores.

**Fix applied:**

```python
# Before
temporal_consistency = float(1.0 - narrative_volatility)

# After
temporal_consistency = float(
    max(0.0, min(1.0, 1.0 - narrative_volatility))
)
```

---

### GRAPH-INFO-001 — `graph_utils.py`: `to_undirected_weighted` uses additive symmetrisation *(INFO)*

**Severity:** Informational  
**File:** `src/graph/graph_utils.py`  
**Function:** `to_undirected_weighted`

**Observation:**  
`to_undirected_weighted` symmetrises a directed graph by **averaging** the two directional weights when both `(u→v)` and `(v→u)` exist. The canonical path in `graph_pipeline.py` uses `canonicalize_weighted`, which takes the **max** of the two directions (G-C2). The two helpers are inconsistent.

`to_undirected_weighted` has **no callers inside `src/graph/`** and does not appear anywhere in the production inference path. It is a dead utility.

**Recommendation:**  
No code change required. If `to_undirected_weighted` is ever activated, align its symmetrisation strategy with `canonicalize_weighted` (`max` over the two directions rather than averaging) to prevent weight deflation on re-canonicalisation.

---

### GRAPH-INFO-002 — `graph_visualization.py`: purely offline, not wired into production *(INFO)*

**Severity:** Informational  
**File:** `src/graph/graph_visualization.py`

**Observation:**  
`GraphVisualizer` (matplotlib / pyvis rendering) is a standalone utility with no callers in the production inference or API layers. It has no test coverage and the `pyvis` dependency is not in the pinned requirements. There is no correctness risk.

**Recommendation:**  
No code change required. If activated in a future debug or reporting workflow, verify that `pyvis` is added to the dependency manifest to avoid silent ImportError at runtime.

---

## Summary

| ID | Severity | File | Status |
|----|----------|------|--------|
| GRAPH-FIX-001 | Medium | `temporal_graph.py` | **Fixed** |
| GRAPH-INFO-001 | Info | `graph_utils.py` | Documented — no action |
| GRAPH-INFO-002 | Info | `graph_visualization.py` | Documented — no action |

**Overall assessment:** The graph layer is production-grade. The singleton architecture, batched NLP, sparse adjacency representation, config-driven weights, and thread-safe initialisation are all correctly implemented. One out-of-range value bug was patched; no structural or performance regressions were found.
