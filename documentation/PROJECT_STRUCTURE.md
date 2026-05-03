# Project Structure

This document explains the **directory structure and architectural organization** of the TruthLens AI repository.

TruthLens AI is organized as a **modular machine learning platform** designed for:

- Misinformation detection and fake news classification
- Credibility analysis and bias profiling
- Linguistic signal extraction (emotion, propaganda, narrative)
- Explainable AI (SHAP, LIME, attention rollout)
- Scalable inference via REST API

The repository follows a **layered architecture** separating data processing, feature extraction, modeling, inference, and evaluation.

---

## Root Directory

```
TruthLens-AI/
├── api/                     # FastAPI REST service
├── config/                  # YAML configuration files
├── data/                    # Raw, processed, and split datasets
├── documentation/           # Architecture and system documentation
├── logs/                    # Training and inference logs
├── models/                  # Trained model artifacts and inference helpers
├── reports/                 # Evaluation reports and EDA outputs
├── src/                     # Core application source code
├── tests/                   # Unit and integration tests (236+ tests)
├── main.py                  # Training entry point
├── run_eda.py               # EDA report generator
├── requirements.txt         # Python dependencies
└── replit.md                # Replit-specific project notes
```

---

## API Layer — `api/`

```
api/
├── __init__.py
└── app.py                   # FastAPI application entry point
```

Exposes a **FastAPI-based REST service** for article analysis and model inference.

**Endpoints:**

| Method | Path             | Description                                   |
|--------|------------------|-----------------------------------------------|
| GET    | `/`              | Health check, lists all available endpoints   |
| GET    | `/health`        | Detailed health check (model file status)     |
| POST   | `/predict`       | Binary fake/real classification               |
| POST   | `/analyze`       | Full analysis: bias, emotion, explainability  |
| GET    | `/project-view`  | API metadata and directory structure          |
| GET    | `/docs`          | Interactive Swagger API documentation         |

---

## Configuration — `config/`

```
config/
├── config.yaml              # Model, training, API, and inference settings
└── data_config.yaml         # Dataset pipeline and preprocessing settings
```

Stores all system configuration parameters. See [CONFIGURATION.md](CONFIGURATION.md) for details.

---

## Data Layer — `data/`

```
data/
├── raw/                     # Original source datasets
│   ├── bias/
│   ├── emotion/
│   ├── ideology/
│   ├── narrative/
│   └── propaganda/
├── interim/                 # Intermediate processing outputs
├── processed/               # Cleaned and merged datasets
│   └── unified_dataset.csv
└── splits/                  # Train / validation / test CSVs
    ├── train.csv
    ├── validation.csv
    └── test.csv
```

Datasets cover: fake news, bias, emotion, narrative framing, propaganda, and ideology. All are unified using a **shared label schema**.

---

## Documentation — `documentation/`

```
documentation/
├── API_REFERENCE.md         # Complete REST API reference
├── ARCHITECTURE.md          # System architecture overview
├── CONFIGURATION.md         # Configuration file reference
├── CONTRIBUTING.md          # Contributor guidelines
├── DEPLOYMENT.md            # Deployment instructions
├── FEATURE_ENGINEERING.md   # Feature engineering system
├── MODEL_CARD.md            # Model details, datasets, limitations
├── PROJECT_STRUCTURE.md     # This file
├── SYSTEM_DESIGN.md         # End-to-end system design
├── TRAINING_GUIDE.md        # Model training walkthrough
└── TROUBLESHOOTING.md       # Common issues and fixes
```

---

## Logs — `logs/`

```
logs/
├── training.log             # Training run logs
└── inference.log            # Inference and API logs
```

---

## Models — `models/`

```
models/
├── inference/
│   ├── __init__.py
│   └── predictor.py         # predict() and predict_batch() functions
├── registry/
│   └── model_registry.py    # ModelRegistry — loads and caches model assets
├── cache/                   # HuggingFace model download cache
├── truthlens_model/         # Trained model artifacts (created after training)
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   ├── merges.txt
│   └── model.safetensors    # (or pytorch_model.bin)
└── tfidf_vectorizer.joblib  # TF-IDF vectorizer artifact
```

The `models/inference/predictor.py` module provides:
- `predict(text)` — single article inference, returns label + fake probability + confidence
- `predict_batch(texts)` — batch inference for LIME explanations

---

## Reports — `reports/`

```
reports/
├── evaluation_results.json
├── confusion_matrix.png
├── data_cleaning_report.json
└── figures/                 # EDA plots and charts
```

Generated by training runs and `python run_eda.py`.

---

## Source Code — `src/`

The `src/` directory contains the **core implementation of TruthLens AI**, organized into subsystems:

### Aggregation — `src/aggregation/`

Computes the **final TruthLens Credibility Score** by weighting signals from all analytical modules.

```
src/aggregation/
├── truthlens_score_calculator.py    # Main scoring engine
├── score_normalizer.py              # Signal normalization
├── risk_assessment.py               # Risk level classification
├── weight_manager.py                # Configurable signal weights
└── score_explainer.py               # Human-readable score explanations
```

Weights (configurable): Bias 0.40 · Emotion 0.35 · Narrative 0.25

### Analysis — `src/analysis/`

Performs deep linguistic analysis of article content.

```
src/analysis/
├── bias_profiler.py
├── narrative_extractor.py
├── propaganda_detector.py
├── rhetorical_device_detector.py
├── discourse_coherence_analyzer.py
└── context_omission_detector.py
```

### Data Processing — `src/data/`

Handles dataset ingestion, cleaning, and preprocessing.

```
src/data/
├── load_data.py
├── merge_datasets.py
├── clean_data.py
├── validate_data.py
├── data_split.py
└── data_augmentation.py
```

### Evaluation — `src/evaluation/`

Measures model performance with comprehensive metrics.

```
src/evaluation/
├── metrics.py
├── calibration.py
├── uncertainty_estimator.py
└── evaluation_dashboard.py
```

Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Calibration

### Explainability — `src/explainability/`

Provides interpretable explanations for model predictions.

```
src/explainability/
├── shap_explainer.py
├── lime_explainer.py
├── attention_rollout.py
├── attention_visualizer.py
├── emotion_explainer.py
├── bias_explainer.py
├── propaganda_explainer.py
├── explanation_aggregator.py
├── explanation_cache.py
├── explanation_metrics.py
└── explanation_report_generator.py
```

### Feature Engineering — `src/features/`

Generates structured features for the models. See [FEATURE_ENGINEERING.md](FEATURE_ENGINEERING.md) for full details.

```
src/features/
├── base/           # Base feature class and FeatureContext
├── bias/           # Bias and ideology lexicon features
├── discourse/      # Argument structure and coherence
├── emotion/        # Emotion lexicon and trajectory features
├── fusion/         # Feature combination and scaling
├── graph/          # Entity/narrative graph features
├── importance/     # Feature importance analysis tools
├── narrative/      # Frame detection and role features
├── pipelines/      # End-to-end feature pipeline orchestration
├── propaganda/     # Manipulative rhetoric patterns
├── text/           # Lexical, semantic, syntactic features
└── cache/          # Feature caching system
```

### Graph Analysis — `src/graph/`

Builds entity and narrative graphs for relational reasoning.

```
src/graph/
├── entity_graph.py
├── narrative_graph.py
├── graph_embeddings.py
└── graph_pipeline.py
```

### Models — `src/models/`

Contains model implementations and task-specific heads.

```
src/models/
├── encoder/         # Shared RoBERTa transformer encoder
├── multitask/       # MultiTaskTruthLensModel (main model class)
├── narrative/       # Narrative role classification head
├── propaganda/      # Propaganda detection head
├── ideology/        # Ideology classification head
├── emotion/         # Multi-label emotion classification head
├── ensemble/        # Ensemble methods
├── calibration/     # Model confidence calibration
├── training/        # Training utilities (optimizer, scheduler)
└── registry/        # ModelRegistry — model loading and caching
```

### Inference — `src/inference/`

Production inference pipeline.

```
src/inference/
├── inference_engine.py
├── prediction_pipeline.py
├── batch_inference.py
├── model_loader.py
└── report_generator.py
```

### Pipelines — `src/pipelines/`

End-to-end ML workflow orchestration.

```
src/pipelines/
├── preprocessing_pipeline.py
├── feature_pipeline.py
├── prediction_pipeline.py
└── truthlens_analysis_pipeline.py
```

### Training — `src/training/`

Model training and optimization utilities.

```
src/training/
├── cross_validation.py
├── hyperparameter_tuning.py
├── optimizer_factory.py
└── scheduler_factory.py
```

### Utilities — `src/utils/`

Shared utilities used across the project.

```
src/utils/
├── config_loader.py        # YAML configuration loading and dataclass conversion
├── settings.py             # Centralized settings system (primary config interface)
├── logging_utils.py        # Structured logging setup
├── device_utils.py         # CUDA / MPS / CPU detection and tensor routing
├── input_validation.py     # Text and DataFrame validation
├── json_utils.py           # JSON artifact save/load helpers
├── seed_utils.py           # Reproducibility (random, numpy, torch seeds)
├── time_utils.py           # Benchmarking timer and decorator
└── helper_functions.py     # General-purpose utilities
```

---

## Tests — `tests/`

```
tests/
├── test_data/
├── test_features/
├── test_models/
├── test_inference/
├── test_explainability/
├── test_api/
└── test_utils/
```

236+ tests covering: data processing, feature pipelines, model training, inference, explainability, API endpoints, and configuration validation. Run with:

```bash
pytest
```

---

## End-to-End System Pipeline

```
News Article Input
       ↓
Preprocessing & Text Cleaning
       ↓
Feature Engineering (Lexical · Bias · Emotion · Narrative · Propaganda)
       ↓
MultiTask Transformer (RoBERTa + 6 Task Heads)
       ↓
Linguistic Analysis Modules (Bias · Narrative · Propaganda · Discourse)
       ↓
Graph Analysis (Entity & Narrative Graphs)
       ↓
Explainability (SHAP · LIME · Attention Rollout)
       ↓
Score Aggregation Engine
       ↓
TruthLens Credibility Score + Risk Level + API Response
```
