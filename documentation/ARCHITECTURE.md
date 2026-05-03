# Architecture

This document describes the **system architecture of TruthLens AI**.

TruthLens AI is a **multi-layer machine learning platform** designed for misinformation detection, credibility analysis, linguistic signal extraction, explainable AI, and scalable inference via REST API.

---

## High-Level Architecture

TruthLens processes news articles through eight analytical layers that each contribute signals to the final credibility evaluation:

```
News Article Input
       ↓
┌─────────────────────────────────────────────────────┐
│  1. Preprocessing & Text Cleaning                   │
└─────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────┐
│  2. Feature Engineering                             │
│     Lexical · Bias · Emotion · Narrative ·          │
│     Propaganda · Discourse · Graph                  │
└─────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────┐
│  3. MultiTask Transformer Model (RoBERTa)           │
│     Fake News · Bias · Propaganda ·                 │
│     Emotion · Ideology · Narrative Roles            │
└─────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────┐
│  4. Linguistic Analysis Modules                     │
│     Bias Profiling · Narrative Extraction ·         │
│     Rhetoric · Discourse Coherence                  │
└─────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────┐
│  5. Graph Reasoning                                 │
│     Entity Graphs · Narrative Graphs ·              │
│     Propagation Detection                           │
└─────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────┐
│  6. Explainability Layer                            │
│     SHAP · LIME · Attention Rollout ·               │
│     Token Attribution                               │
└─────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────┐
│  7. Score Aggregation Engine                        │
│     Signal Normalization · Weighted Scoring ·       │
│     Risk Assessment                                 │
└─────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────┐
│  8. API Response                                    │
│     FastAPI · JSON output · Swagger docs            │
└─────────────────────────────────────────────────────┘
```

---

## System Components

| Layer              | Location             | Purpose                                |
|--------------------|----------------------|----------------------------------------|
| Data Layer         | `src/data/`          | Dataset ingestion and preprocessing    |
| Feature Layer      | `src/features/`      | Structured feature extraction          |
| Model Layer        | `src/models/`        | Transformer and multitask models       |
| Analysis Layer     | `src/analysis/`      | Deep linguistic and narrative analysis |
| Graph Layer        | `src/graph/`         | Entity and narrative graph reasoning   |
| Explainability     | `src/explainability/`| Interpretable prediction explanations  |
| Aggregation Layer  | `src/aggregation/`   | Credibility scoring and risk levels    |
| Inference Layer    | `src/inference/`     | Production inference pipeline          |
| API Layer          | `api/app.py`         | FastAPI REST service                   |

---

## Data Layer

Responsible for **dataset ingestion, validation, and preprocessing**.

```
Raw Datasets (bias, emotion, ideology, narrative, propaganda)
       ↓
Data Cleaning (unicode normalization, URL removal, HTML stripping)
       ↓
Dataset Validation (schema checks, null ratio, duplicate ratio)
       ↓
Dataset Merging (unified label schema)
       ↓
Unified Dataset
       ↓
Train / Validation / Test Splits (70% / 15% / 15%)
```

Key modules: `load_data.py`, `merge_datasets.py`, `clean_data.py`, `validate_data.py`, `data_split.py`, `data_augmentation.py`

---

## Feature Engineering Layer

Transforms raw article text into **structured, interpretable features**.

```
Article Text
       ↓
Tokenization
       ↓
┌──────────────────────────────────────────────┐
│  Parallel Feature Extractors                 │
│  ├── Lexical (token counts, diversity)       │
│  ├── Semantic (contextual meaning)           │
│  ├── Syntactic (grammar, structure)          │
│  ├── Bias (lexicon density, framing)         │
│  ├── Emotion (20-label intensity scoring)    │
│  ├── Narrative (hero/villain/victim roles)   │
│  ├── Propaganda (manipulation patterns)      │
│  └── Graph (entity interaction signals)      │
└──────────────────────────────────────────────┘
       ↓
Feature Fusion & Scaling
       ↓
Unified Feature Representation
```

All feature extractors inherit from `BaseFeature` via `FeatureContext`.

---

## Model Layer — MultiTask Architecture

TruthLens uses a **shared RoBERTa encoder with six task-specific heads**.

```
Article Text
       ↓
Tokenizer (roberta-base, max_length=256)
       ↓
Shared RoBERTa Encoder
       ↓
┌─────────────────────────────────────────────────────────┐
│  Task-Specific Heads                                    │
│  ├── Bias Detection Head      (3-class: left/center/right) │
│  ├── Ideology Detection Head  (3-class)                 │
│  ├── Propaganda Head          (binary)                  │
│  ├── Emotion Head             (20-label multi-label)    │
│  ├── Narrative Roles Head     (hero/villain/victim)     │
│  └── Frame Detection Head     (RE/HI/CO/MO/EC)         │
└─────────────────────────────────────────────────────────┘
       ↓
Softmax / Sigmoid per head
       ↓
Per-task Probabilities
```

**Multi-task learning advantages:**
- Single forward pass produces outputs for all six tasks
- Shared encoder learns unified semantic representations
- Reduced training cost compared to six separate models
- Improved generalization from cross-task regularization

**Inference path** (`models/inference/predictor.py`):
- Model is loaded once and cached in memory
- Device-aware tensor routing (CPU / CUDA / MPS)
- `model.eval()` ensures dropout is disabled during inference

---

## Linguistic Analysis Layer

Performs deeper structural analysis beyond model probabilities.

| Module                       | Detects                               |
|------------------------------|---------------------------------------|
| `bias_profiler.py`           | Partisan framing, ideological lexicon |
| `narrative_extractor.py`     | Hero/villain/victim role assignment   |
| `propaganda_detector.py`     | Loaded language, fear appeals         |
| `rhetorical_device_detector.py` | Hyperbole, appeal to emotion       |
| `discourse_coherence_analyzer.py` | Argument structure consistency  |
| `context_omission_detector.py` | Missing context, selective facts    |

These modules produce **linguistic credibility signals** consumed by the aggregation engine.

---

## Graph Reasoning Layer

Constructs graphs representing relationships between entities and narrative elements.

```
Entity Extraction (via spaCy NER)
       ↓
Graph Construction (NetworkX)
       ↓
Graph Embeddings
       ↓
Topological Feature Extraction
       ↓
Graph-Based Credibility Signals
```

Graph types:
- **Entity graphs** — person/organization/location relationships
- **Narrative graphs** — story arc propagation and conflict detection
- **Temporal graphs** — event ordering and timeline consistency

---

## Explainability Layer

Provides interpretable explanations for every prediction.

| Method               | What it explains                              |
|----------------------|-----------------------------------------------|
| SHAP                 | Global feature importance across predictions  |
| LIME                 | Local token-level explanation per article     |
| Attention Rollout    | Transformer attention attribution             |
| Emotion Explainer    | Lexicon-based emotional signal breakdown      |
| Bias Explainer       | Per-sentence bias heatmap                     |

All explanations are returned as structured JSON in the `/analyze` endpoint response.

---

## Aggregation and Scoring

The aggregation engine combines all signals into a **single normalized credibility score**.

```
Inputs:
  fake_probability     (from transformer)
  bias_score           (from bias modules)
  propaganda_signals   (from propaganda head)
  emotion_intensity    (from emotion features)
  narrative_signals    (from narrative modules)
  discourse_signals    (from analysis layer)

       ↓
Signal Normalization
       ↓
Weighted Combination
  Bias weight:      0.40
  Emotion weight:   0.35
  Narrative weight: 0.25
       ↓
Risk Assessment
  Low (0.0–0.33) · Medium (0.34–0.66) · High (0.67–1.0)
       ↓
TruthLens Credibility Score
```

---

## API Layer

The system exposes all inference capabilities through a **FastAPI service** running on port 5000.

**Quick predict:**
```
POST /predict
{ "text": "Breaking news: ..." }

→ { "text": "...", "prediction": "FAKE", "fake_probability": 0.87, "confidence": 0.87 }
```

**Full analysis:**
```
POST /analyze
{ "text": "Breaking news: ..." }

→ {
    "prediction": "FAKE",
    "fake_probability": 0.87,
    "confidence": 0.87,
    "bias": { "bias_score": 0.12, "media_bias": "lean", "biased_tokens": [...], "sentence_heatmap": [...] },
    "emotion": { "dominant_emotion": "fear", "emotion_scores": {...}, "emotion_distribution": {...} },
    "explainability": { "emotion_explanation": {...}, "lime": {...} }
  }
```

See [API_REFERENCE.md](API_REFERENCE.md) for the complete endpoint specification.

---

## Training Architecture

```
Training Dataset (train.csv)
       ↓
Feature Pipeline (TF-IDF + lexical features)
       ↓
MultiTaskTruthLensModel (roberta-base encoder + 6 heads)
       ↓
Combined Multi-Task Loss (cross-entropy per head)
       ↓
AdamW Optimizer + Linear Warmup Scheduler
       ↓
Gradient Clipping (max_norm=1.0)
       ↓
Early Stopping (patience=2, metric=eval_loss)
       ↓
Checkpoint Saved to models/truthlens_model/
```

Configuration: `config/config.yaml` · Entry point: `python main.py`

---

## Inference Architecture

```
Incoming Request → FastAPI (api/app.py)
       ↓
Input Validation (min 10 chars, max 10,000 chars)
       ↓
ModelRegistry.load_model() [cached after first call]
       ↓
Tokenizer → RoBERTa Encoder → Softmax → Probabilities
       ↓
Parallel: BiasLexiconFeatures · EmotionLexiconAnalyzer · LIME
       ↓
JSON Response Assembly
       ↓
HTTP Response
```

---

## Deployment Architecture

TruthLens is deployed on **Replit Autoscale**.

```
Client Request
       ↓
Replit Autoscale (HTTPS, mTLS proxy)
       ↓
Gunicorn (UvicornWorker, port 5000)
       ↓
FastAPI Application
       ↓
Inference Pipeline
       ↓
JSON Response
```

Server command: `gunicorn --bind=0.0.0.0:5000 -k uvicorn.workers.UvicornWorker api.app:app`

Development server: `python -m uvicorn api.app:app --host 0.0.0.0 --port 5000 --reload --reload-dir api --reload-dir src --reload-dir config --reload-dir models`

---

## Design Principles

**Modularity** — Every subsystem is independently replaceable. Feature extractors, model heads, and analysis modules all plug into shared base classes.

**Scalability** — Batch inference, feature caching, and configurable pipeline steps support large-scale processing.

**Interpretability** — SHAP, LIME, and attention-based explanations are built into the core inference path, not added as afterthoughts.

**Reproducibility** — Fixed seeds, YAML configuration, and a comprehensive test suite ensure consistent results across runs.

**Fail-Safe Inference** — The API degrades gracefully: LIME errors are caught and returned as structured error objects; the `/health` endpoint reports model availability without crashing.
