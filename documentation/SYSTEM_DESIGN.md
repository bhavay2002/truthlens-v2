# System Design

This document describes the **end-to-end system design of TruthLens AI**, including data pipelines, feature engineering, model training, inference, and deployment.

TruthLens AI is a **modular machine learning system** that analyzes news articles and produces a **credibility score supported by interpretable linguistic signals**.

---

## System Overview

```
News Article
       ↓
Preprocessing & Text Cleaning
       ↓
Feature Engineering (7 feature domains)
       ↓
MultiTask Transformer (RoBERTa + 6 task heads)
       ↓
Linguistic Analysis (bias, narrative, propaganda, discourse)
       ↓
Graph Reasoning (entity & narrative graphs)
       ↓
Explainability (SHAP, LIME, attention rollout)
       ↓
Aggregation Engine (weighted credibility scoring)
       ↓
TruthLens Credibility Score + Risk Level
       ↓
FastAPI Response (JSON)
```

---

## Data Flow Architecture

### Offline Training Pipeline

```
Raw Datasets (data/raw/)
  bias/, emotion/, ideology/, narrative/, propaganda/
       ↓
Data Loading (src/data/load_data.py)
       ↓
Text Cleaning (src/data/clean_data.py)
  - Unicode normalization
  - URL removal
  - HTML stripping
  - Contraction expansion
  - Lowercasing + whitespace normalization
  - Min word count filtering (≥30)
       ↓
Dataset Validation (src/data/validate_data.py)
  - Schema integrity
  - Null ratio check (≤10%)
  - Duplicate ratio check (≤15%)
  - Class balance check (min_class_ratio ≥10%)
       ↓
Dataset Merging (src/data/merge_datasets.py)
  - Unified label schema across all tasks
       ↓
Class Balancing (method: oversample)
       ↓
Data Augmentation (synonym replacement, random swap, random deletion)
       ↓
Train / Validation / Test Split (70% / 15% / 15%)
  → data/splits/train.csv
  → data/splits/validation.csv
  → data/splits/test.csv
```

### Online Inference Pipeline

```
POST /predict or POST /analyze
       ↓
Input Validation (min 10, max 10,000 chars)
       ↓
ModelRegistry.load_model() (cached after first call)
       ↓
Tokenizer + RoBERTa Encoder → Softmax
       ↓
Prediction Output (fake_probability, label, confidence)
       ↓
(for /analyze only)
  ├── BiasLexiconFeatures.extract()
  ├── EmotionLexiconAnalyzer.analyze()
  └── LIME explanation (256 samples, top-8 features)
       ↓
JSON Response Assembly
       ↓
HTTP 200 Response
```

---

## Feature Engineering System

TruthLens extracts features from seven dimensions of article content. All extractors use the `FeatureContext → BaseFeature → dict[str, float]` pattern.

| Feature Domain | Key Signals                                          |
|----------------|------------------------------------------------------|
| Text           | Token count, lexical diversity, sentence length      |
| Bias           | Lexicon density, evaluative words, partisan framing  |
| Emotion        | 20-label intensity scores, dominant emotion          |
| Narrative      | Frame detection (RE/HI/CO/MO/EC), role assignment    |
| Propaganda     | Loaded language, fear appeals, exaggeration          |
| Discourse      | Argument structure, claim-evidence relationships     |
| Graph          | Entity co-occurrence, centrality, interaction density|

Feature pipeline configuration (`config/config.yaml`):
```yaml
features:
  engineered_text_column: engineered_text
  tfidf:
    enabled: true
    max_features: 5000
    top_terms_per_doc: 4
```

TF-IDF vectorizer artifact: `models/tfidf_vectorizer.joblib`

---

## Model Training Architecture

### Multi-Task Learning Design

TruthLens uses a **single shared encoder feeding six task-specific heads**:

```
Shared RoBERTa Encoder (roberta-base)
       ↓
┌──────────────────────────────────────────────────┐
│  Task Heads                    Loss              │
│  ├── Bias Detection            cross_entropy     │
│  ├── Ideology Detection        cross_entropy     │
│  ├── Propaganda Detection      cross_entropy     │
│  ├── Emotion Detection (x20)   binary_cross_entropy │
│  ├── Narrative Roles (x3)      binary_cross_entropy │
│  └── Frame Detection (x5)      binary_cross_entropy │
└──────────────────────────────────────────────────┘
       ↓
Combined multi-task loss
       ↓
AdamW backward pass
```

### Training Configuration

| Hyperparameter               | Value    |
|------------------------------|----------|
| Base model                   | roberta-base |
| Max token length             | 256      |
| Dropout                      | 0.1      |
| Epochs                       | 4        |
| Batch size                   | 8        |
| Gradient accumulation steps  | 2        |
| Effective batch size         | 16       |
| Learning rate                | 2.0e-5   |
| Weight decay                 | 0.01     |
| Warmup ratio                 | 0.1      |
| Scheduler                    | linear   |
| Optimizer                    | adamw    |
| Gradient clipping            | 1.0      |
| FP16 mixed precision         | enabled  |
| Early stopping patience      | 2 epochs |
| Early stopping metric        | eval_loss|
| Random seed                  | 42       |

### Training Support Features

- **Cross-validation** — `src/training/cross_validation.py` (configurable splits, disabled by default)
- **Hyperparameter tuning** — Optuna integration (`src/training/hyperparameter_tuning.py`, disabled by default)
- **Checkpoint management** — saves up to 3 recent checkpoints
- **Model registry** — `src/models/registry/model_registry.py` — loads, caches, and provides versioned access to model assets

---

## Linguistic Analysis Layer

After transformer inference, the analysis layer extracts higher-level structural signals:

| Module                           | Signal Type               | Used By              |
|----------------------------------|---------------------------|----------------------|
| `bias_profiler.py`               | Partisan framing score    | Aggregation engine   |
| `narrative_extractor.py`         | Hero/villain/victim roles | Narrative scoring    |
| `propaganda_detector.py`         | Manipulation patterns     | Propaganda scoring   |
| `rhetorical_device_detector.py`  | Hyperbole, loaded phrases | Bias and propaganda  |
| `discourse_coherence_analyzer.py`| Argument consistency      | Credibility scoring  |
| `context_omission_detector.py`   | Missing context detection | Credibility scoring  |

---

## Graph Reasoning Architecture

Entity and narrative graph reasoning provides **relational credibility signals**:

```
Named Entity Recognition (spaCy NER)
       ↓
Entity Extraction (persons, organizations, locations)
       ↓
Co-occurrence Graph (NetworkX)
  - Nodes = entities
  - Edges = co-occurrence in same sentence
       ↓
Graph Metrics (centrality, clustering coefficient, path length)
       ↓
Narrative Propagation Detection
Conflict Detection
Interaction Pattern Features
       ↓
Graph Feature Vector → Aggregation Engine
```

Graph types: entity graphs · narrative graphs · temporal event graphs

---

## Explainability Architecture

Every prediction can be accompanied by human-readable explanations:

| Method               | Granularity     | Output Format                          |
|----------------------|-----------------|----------------------------------------|
| LIME                 | Token-level     | `important_features: [{feature, weight}]` |
| SHAP                 | Feature-level   | Shapley value attribution per feature  |
| Attention Rollout    | Token-level     | Attention weight distribution          |
| Emotion Explainer    | Lexicon-based   | Emotion trigger words and intensities  |
| Bias Explainer       | Sentence-level  | Per-sentence bias heatmap              |

**LIME configuration:**
- Samples: 256
- Top features returned: 8
- Class names: `["Real", "Fake"]`
- LIME errors are caught and returned as `{ "error": "lime_unavailable" }` — the `/analyze` endpoint does not fail when LIME is unavailable.

---

## Aggregation and Scoring System

The aggregation engine produces the **final TruthLens Credibility Score**:

```
Signal Inputs:
  ├── fake_probability    (transformer output)
  ├── bias_score          (bias module)
  ├── propaganda_signals  (propaganda module)
  ├── emotion_intensity   (emotion module)
  ├── narrative_signals   (narrative module)
  └── discourse_coherence (analysis layer)
       ↓
TruthLensScoreCalculator
  ├── Signal Normalization
  ├── Weighted Combination
  │     Bias weight:      0.40
  │     Emotion weight:   0.35
  │     Narrative weight: 0.25
  └── Risk Assessment
        Low    0.00–0.33
        Medium 0.34–0.66
        High   0.67–1.00
       ↓
TruthLens Credibility Score (0.0–1.0)
Risk Level (Low / Medium / High)
Score Explanation (human-readable)
```

---

## API Design

The FastAPI service exposes all inference capabilities:

| Endpoint        | Method | Purpose                                              |
|-----------------|--------|------------------------------------------------------|
| `/`             | GET    | Health check, lists endpoints                        |
| `/health`       | GET    | Model availability and file status check             |
| `/predict`      | POST   | Binary classification (FAKE/REAL) + confidence       |
| `/analyze`      | POST   | Full analysis with bias, emotion, explainability     |
| `/project-view` | GET    | API metadata and directory structure                 |
| `/docs`         | GET    | Swagger interactive API documentation                |

**Request validation:**
- Text min length: 10 characters
- Text max length: 10,000 characters
- Content-Type: `application/json`

**Error handling:**
- `400` — Invalid input (text too short/long, malformed JSON)
- `503` — Model not trained yet (model files missing)
- `500` — Internal server error during inference

---

## Deployment Architecture

**Production deployment (Replit Autoscale):**
```
Client (HTTPS)
       ↓
Replit Autoscale proxy (mTLS)
       ↓
Gunicorn (UvicornWorker, port 5000)
       ↓
FastAPI (api/app.py)
       ↓
ModelRegistry → cached model and tokenizer
       ↓
Inference pipeline
       ↓
JSON response
```

Production run command:
```
gunicorn --bind=0.0.0.0:5000 --reuse-port -k uvicorn.workers.UvicornWorker api.app:app
```

Development run command:
```
python -m uvicorn api.app:app --host 0.0.0.0 --port 5000 --reload \
  --reload-dir api --reload-dir src --reload-dir config --reload-dir models
```

---

## Testing and Validation

TruthLens includes 236+ tests covering all system layers.

| Test Category        | What is tested                                   |
|----------------------|--------------------------------------------------|
| Data pipelines       | Cleaning, validation, merging, splitting         |
| Feature pipelines    | All feature extractors, fusion, caching          |
| Model training       | Forward passes, loss computation, checkpointing  |
| Inference            | predict(), predict_batch(), device handling      |
| Explainability       | LIME, SHAP, attention rollout outputs            |
| API endpoints        | All HTTP endpoints, error handling, edge cases   |
| Configuration        | YAML loading, schema validation, path resolution |
| Reproducibility      | Seed consistency, deterministic outputs          |

Run: `pytest` · Or with coverage: `pytest --cov=src --cov=api --cov=models`

---

## Design Principles

**Modularity** — Every subsystem is independently replaceable. Feature extractors, model heads, analysis modules, and explanation methods all plug in via shared base interfaces.

**Scalability** — Batch processing, configurable pipeline steps, and feature caching support dataset-scale operations without bottlenecks.

**Interpretability** — Explainability is built into the core inference path. Every prediction can be traced back to the specific tokens and features that drove it.

**Reproducibility** — Fixed random seeds, YAML-based configuration, and a comprehensive test suite ensure experiments are reproducible across runs and environments.

**Fail-Safe Inference** — The API catches errors at each layer (LIME, model loading, feature extraction) and degrades gracefully rather than crashing, returning structured error information to the caller.

**Production Readiness** — The system is designed for Replit Autoscale deployment with Gunicorn + UvicornWorker, model caching on first load, and health check endpoints for monitoring.
