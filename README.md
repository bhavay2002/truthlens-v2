
# TruthLens AI

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![Tests](https://img.shields.io/badge/Tests-344%20passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**TruthLens AI** is a **multi-layer AI system for misinformation detection and credibility analysis**.

The platform combines:

* Multi-task RoBERTa transformer models with shared encoder and task-specific heads
* Linguistic feature engineering and narrative discourse analysis
* Propaganda and ideological framing detection
* Bias and emotion classification (20-label multi-label)
* Graph-based entity and claim reasoning
* Explainable AI via SHAP, LIME, and attention rollout
* Weighted credibility score aggregation with risk assessment
* Model calibration, ensemble inference, and ONNX/TorchScript export

TruthLens evaluates news articles using **dozens of analytical signals** and produces a **structured credibility score with explanations**.

---

# Key Capabilities

### Fake News Detection

Binary classification of news articles:

```
REAL vs FAKE
```

Using multi-task transformer models and engineered features.

---

### Linguistic & Narrative Analysis

TruthLens performs deep linguistic analysis including:

* Bias profiling
* Emotion targeting (20-class multi-label)
* Narrative structure extraction (hero / villain / victim)
* Narrative frame detection (RE / HI / CO / MO / EC)
* Rhetorical device detection
* Context omission detection
* Information density analysis

---

### Propaganda & Ideology Detection

The system identifies propaganda techniques and ideological framing:

* Manipulation patterns
* Ideologically loaded language (left / center / right)
* Framing strategies
* Persuasion techniques

---

### Graph-Based Analysis

TruthLens constructs graphs representing relationships between:

* Entities
* Claims
* Narratives
* Sources

Graph reasoning enables **propagation and narrative conflict detection**.

---

### Explainable AI

Predictions are accompanied by explanations using:

* SHAP token importance
* LIME local interpretability
* Attention rollout
* Integrated gradients
* Explanation consistency checks

---

### Credibility Score Aggregation

Outputs a final **TruthLens credibility score** using weighted signals from multiple modules.

Example output:

```
Fake News Probability: 0.82
Bias Score: 0.61
Propaganda Score: 0.72
Narrative Manipulation Score: 0.58

TruthLens Credibility Score: 0.24
Risk Level: HIGH
```

---

# System Architecture

TruthLens follows a **multi-stage ML pipeline**.

```
Article Input
      ↓
Preprocessing
      ↓
Feature Engineering
      ↓
Multi-Task RoBERTa Encoder
      ↓
Task Heads (Bias / Ideology / Propaganda / Narrative / Emotion / Frame)
      ↓
Graph Analysis
      ↓
Explainability Layer (SHAP / LIME / Attention)
      ↓
Aggregation Engine
      ↓
TruthLens Credibility Score
```

---

# Repository Structure

```
TruthLens-AI/

api/                     FastAPI inference service (16 endpoints)
config/                  YAML configuration files
data/                    Root-level data pipeline orchestration
  data_pipeline.py       Pipeline entry point and config validator
  truthlens_sample_dataset.csv   10-row × 38-column sample dataset
experiments/             Experimental results
logs/                    Training & inference logs
models/                  Saved model artifacts
  truthlens_model/       Trained MultiTaskTruthLensModel weights + tokenizer
notebooks/               Research notebooks
reports/                 Evaluation reports and EDA
training/                Root-level training compatibility shim

src/

  aggregation/           Credibility scoring and risk assessment
  analysis/              Linguistic & narrative analysis modules
  data/                  Data ingestion & preprocessing
  evaluation/            Evaluation metrics and dashboards
  explainability/        SHAP, LIME, bias explainer, and explanation tools
  features/              Feature engineering pipelines
  graph/                 Graph construction & analysis
  inference/             Production inference engine
  models/
    encoder/             Shared transformer encoder
    heads/               Classification and multi-label task heads
    multitask/           MultiTaskTruthLensModel and config
    inference/           Predictor (predict / predict_batch)
    calibration/         Model confidence calibration
    ensemble/            Ensemble methods (average / weighted / majority vote)
    export/              ONNX and TorchScript export
  pipelines/             End-to-end ML pipelines
  training/              Training utilities, cross-validation, hyperparameter tuning
  utils/                 Config, logging, seed control, helper utilities
  visualization/         Plotting and evaluation visualization

tests/                   38 test modules, 344 tests (all passing)

main.py                  Training pipeline entry point
evaluate.py              Evaluation script
run_eda.py               Exploratory data analysis
```

---

# Installation

**Python 3.12+ required.**

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

```bash
# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

Install dependencies (CPU-only PyTorch recommended to reduce disk usage):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Download the spaCy language model:

```bash
python -m spacy download en_core_web_sm
```

---

# Training

Run the training pipeline:

```bash
python main.py
```

The pipeline performs:

1. Dataset merging
2. Data cleaning
3. Feature engineering
4. Multi-task model training
5. Evaluation report generation

---

# Evaluation

Evaluate trained models:

```bash
python evaluate.py
```

Evaluation outputs include:

* Precision, Recall, F1 Score
* Calibration analysis
* Confusion matrices
* Task correlation analysis
* Uncertainty quantification

Reports are stored in:

```
reports/
```

---

# Inference API

Start the FastAPI server:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 5000 --reload \
  --reload-dir api --reload-dir src --reload-dir config --reload-dir models
```

The API is available at:

```
http://localhost:5000
```

Interactive docs (Swagger UI):

```
http://localhost:5000/docs
```

---

# API Endpoints

| Method | Path                      | Description                                        |
|--------|---------------------------|----------------------------------------------------|
| GET    | `/`                       | Home — lists all endpoints, confirms status        |
| GET    | `/health`                 | Detailed health check (model file status)          |
| GET    | `/project-view`           | API metadata and directory structure               |
| GET    | `/docs`                   | Interactive Swagger documentation                  |
| POST   | `/predict`                | Binary FAKE/REAL classification + confidence       |
| POST   | `/batch-predict`          | Batch prediction for up to 50 articles             |
| POST   | `/analyze`                | Full analysis: bias, emotion, narrative, LIME      |
| POST   | `/report`                 | Structured credibility report                      |
| GET    | `/inference/model-info`   | Model registry status and path                     |
| POST   | `/cache/clear`            | Clear the inference result cache                   |
| GET    | `/calibration/info`       | Calibration method descriptions                    |
| POST   | `/calibration/metrics`    | Compute ECE, MCE, Brier score, NLL                 |
| GET    | `/ensemble/info`          | Ensemble strategy descriptions                     |
| POST   | `/ensemble/predict`       | Ensemble prediction (average / weighted / vote)    |
| GET    | `/export/info`            | Export format descriptions                         |
| POST   | `/export/onnx`            | Export model to ONNX format                        |
| POST   | `/export/torchscript`     | Export model to TorchScript format                 |

---

# API Example

```
POST /predict
```

Example request:

```bash
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{"text":"Breaking news: Scientists discover new species in the Amazon rainforest."}'
```

Example response:

```json
{
  "text": "Breaking news: Scientists discover new species...",
  "prediction": "REAL",
  "fake_probability": 0.0924,
  "confidence": 0.9076
}
```

Batch example:

```bash
curl -X POST http://localhost:5000/batch-predict \
-H "Content-Type: application/json" \
-d '{"texts": ["Article one text here...", "Article two text here..."]}'
```

---

# Sample Dataset

A 10-row sample dataset is included at `data/truthlens_sample_dataset.csv`.

It covers all 38 unified schema columns:

| Column Group              | Columns                                               |
|---------------------------|-------------------------------------------------------|
| Text                      | `title`, `text`                                       |
| Task labels               | `bias_label`, `ideology_label`, `propaganda_label`    |
| Narrative frame           | `frame`, `CO`, `EC`, `HI`, `MO`, `RE`                |
| Narrative roles           | `hero`, `villain`, `victim`                           |
| Entity lists (JSON)       | `hero_entities`, `villain_entities`, `victim_entities`|
| Emotion (20-label binary) | `emotion_0` … `emotion_19`                            |
| Dataset source            | `dataset`                                             |

The sample includes 5 non-biased and 5 biased articles from three source datasets (isot, liar, fakenewsnet) with all 5 narrative frame types represented.

---

# Datasets

TruthLens integrates multiple datasets including:

| Task       | Dataset                 |
| ---------- | ----------------------- |
| Fake News  | ISOT, LIAR, FakeNewsNet |
| Bias       | BABE, BASIL, MBIC       |
| Emotion    | GoEmotions, SemEval     |
| Narrative  | FrameNet                |
| Propaganda | PTC Propaganda          |
| Ideology   | AllSides                |

Datasets are unified using a **shared label schema** (`src/data/unified_label_schema.py`).

---

# Testing

Run the full test suite:

```bash
pytest
```

**344 tests across 38 modules — all passing.**

Coverage includes:

| Area | Test Modules |
|------|-------------|
| End-to-end dataset & API | `test_e2e_dataset.py` |
| API endpoints & error paths | `test_api.py`, `test_api_error_paths.py` |
| Aggregation & risk scoring | `test_aggregation.py` |
| Evaluation metrics & uncertainty | `test_evaluation.py`, `test_evaluation_metrics.py` |
| Explainability (SHAP / LIME / bias) | `test_explainability.py`, `test_shap_explainer.py` |
| Emotion lexicon analysis | `test_emotion.py` |
| Input validation | `test_input_validation.py` |
| Model architecture & registry | `test_model_subpackage_imports.py`, `test_model_registry.py`, `test_multitask_label_helpers.py` |
| Model training & tokenization | `test_model_training.py`, `test_tokenization.py` |
| Training pipeline & cross-validation | `test_training_pipeline.py` |
| Inference speed & prediction stability | `test_inference_speed.py`, `test_prediction_stability.py` |
| Data pipelines & schema | `test_data_pipeline_module.py`, `test_dataset_schema.py` |
| Configuration loading | `test_config_loading.py`, `test_config_integrity.py` |
| Reproducibility (seed control) | `test_reproducibility.py` |
| Utility functions | `test_utils.py` |
| Project structure | `test_project_structure.py` |

The end-to-end test file (`test_e2e_dataset.py`) contains **108 tests** across 13 classes covering the sample dataset schema, all 16 API endpoints, calibration metrics, ensemble strategies, export endpoints, error handling, and prediction self-consistency.

---

# Future Work

Planned extensions:

* Multilingual misinformation detection
* Real-time news monitoring
* Knowledge graph integration
* Cross-source narrative tracking
* Browser credibility extension

---

# License

MIT License
