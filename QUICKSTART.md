
# TruthLens AI — Quick Start

This guide explains how to **run TruthLens AI quickly using the multi-task RoBERTa architecture**.

TruthLens is an AI system designed to analyze news articles and detect:

* Fake news signals
* Political and media bias
* Propaganda techniques
* Narrative framing
* Ideological positioning
* Emotional manipulation

The system combines **multi-task transformer models and feature pipelines** to produce a **comprehensive credibility and narrative analysis** of an article.

---

# 1. Environment Setup

**Python 3.12+ required.**

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

```bash
# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

Install dependencies (CPU-only PyTorch is recommended to reduce disk usage):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Download the spaCy model:

```bash
python -m spacy download en_core_web_sm
```

---

# 2. Dataset Requirements

TruthLens uses **multi-task learning**, so datasets may contain labels for multiple signals.

A **10-row sample dataset** is included at `data/truthlens_sample_dataset.csv` and covers all 38 unified schema columns — useful for testing the API pipeline end-to-end without downloading external data.

Full training datasets should be placed in:

```
data/raw/bias/
data/raw/emotion/
data/raw/ideology/
data/raw/narrative/
data/raw/propaganda/
```

These are merged and preprocessed automatically during `python main.py`.

### Recommended Datasets

| Task                   | Datasets                         |
|------------------------|----------------------------------|
| Fake news detection    | ISOT, LIAR, FakeNewsNet          |
| Bias detection         | BABE, BASIL, MBIC                |
| Emotion classification | GoEmotions, SemEval 2018         |
| Ideology detection     | AllSides                         |
| Narrative analysis     | FrameNet                         |
| Propaganda detection   | PTC Propaganda Corpus            |

---

# 3. Configuration

Edit the configuration file:

```
config/config.yaml
```

Key settings:

### Training Configuration

| Parameter                        | Description                     |
|----------------------------------|---------------------------------|
| `training.epochs`                | Number of training epochs       |
| `training.batch_size`            | Per-device batch size           |
| `training.learning_rate`         | AdamW peak learning rate        |
| `training.device`                | `auto`, `cpu`, `cuda`, or `mps` |
| `training.fp16`                  | Mixed precision (CUDA only)     |
| `training.text_column`           | Column name used as input text  |

### Model Configuration

| Parameter                        | Description                             |
|----------------------------------|-----------------------------------------|
| `model.encoder.name`             | HuggingFace encoder ID (`roberta-base`) |
| `model.encoder.max_length`       | Max token length per input (256)        |
| `model.path`                     | Where to save trained model artifacts   |

### Optional Features

| Parameter                              | Description                   |
|----------------------------------------|-------------------------------|
| `cross_validation.enabled`            | Enable k-fold cross-validation|
| `hyperparameter_tuning.enabled`       | Enable Optuna search          |
| `hyperparameter_tuning.trials`        | Number of tuning trials       |

---

# 4. Train the Model

Run the full training pipeline:

```bash
python main.py
```

Training pipeline steps:

1. Dataset loading and merging (`data/raw/`)
2. Data cleaning and normalization
3. Dataset validation (null/duplicate/class balance checks)
4. Feature engineering (TF-IDF + lexical features)
5. Class balancing and augmentation
6. Stratified train / validation / test split (70 / 15 / 15)
7. Multi-task RoBERTa model training
8. Optional cross-validation and hyperparameter tuning
9. Model checkpoint saved to `models/truthlens_model/`
10. Evaluation report generation

---

# 5. Start the API Server

```bash
uvicorn api.app:app --host 0.0.0.0 --port 5000 --reload \
  --reload-dir api --reload-dir src --reload-dir config --reload-dir models
```

The API is available at:

```
http://localhost:5000
```

Interactive Swagger docs:

```
http://localhost:5000/docs
```

---

# 6. API Endpoints

| Method | Path                    | Description                                     |
|--------|-------------------------|-------------------------------------------------|
| GET    | `/`                     | Home — status and endpoint list                 |
| GET    | `/health`               | Model file status and readiness                 |
| GET    | `/project-view`         | API metadata and directory structure            |
| GET    | `/docs`                 | Interactive Swagger UI                          |
| POST   | `/predict`              | FAKE/REAL classification + confidence           |
| POST   | `/batch-predict`        | Batch prediction for up to 50 articles          |
| POST   | `/analyze`              | Full analysis: bias, emotion, narrative, LIME   |
| POST   | `/report`               | Structured credibility report                   |
| GET    | `/inference/model-info` | Model registry status                           |
| POST   | `/cache/clear`          | Clear the inference result cache                |
| GET    | `/calibration/info`     | Calibration method descriptions                 |
| POST   | `/calibration/metrics`  | Compute ECE, MCE, Brier score, NLL              |
| GET    | `/ensemble/info`        | Ensemble strategy descriptions                  |
| POST   | `/ensemble/predict`     | Ensemble prediction (average / weighted / vote) |
| GET    | `/export/info`          | Export format descriptions                      |
| POST   | `/export/onnx`          | Export model to ONNX                            |
| POST   | `/export/torchscript`   | Export model to TorchScript                     |

### Example request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Scientists confirm exercise reduces cardiovascular disease risk in landmark study."}'
```

### Example response

```json
{
  "text": "Scientists confirm exercise reduces cardiovas...",
  "prediction": "REAL",
  "fake_probability": 0.0924,
  "confidence": 0.9076
}
```

---

# 7. Evaluate Trained Models

```bash
python evaluate.py
```

Outputs include:

* Accuracy, Precision, Recall, F1 per task
* Calibration analysis (ECE, Brier score)
* Confusion matrices
* Multi-task evaluation report

Results are saved to `reports/evaluation_results.json`.

---

# 8. Run the Test Suite

Execute all 344 tests:

```bash
pytest
```

Or run quietly:

```bash
python -B -m pytest -q
```

The end-to-end tests require the server running on port 5000. Run them specifically with:

```bash
pytest tests/test_e2e_dataset.py -v
```

Other useful targeted test commands:

```bash
# Training utilities
pytest tests/test_training_pipeline.py -q

# Feature pipeline
pytest tests/test_feature_pipeline.py -q

# API endpoint tests
pytest tests/test_api.py tests/test_api_error_paths.py -q

# Model architecture
pytest tests/test_model_registry.py tests/test_multitask_label_helpers.py -q
```

---

# 9. Generated Artifacts

After training, the following artifacts are created:

| Artifact                                    | Description                         |
|---------------------------------------------|-------------------------------------|
| `models/truthlens_model/config.json`        | Model architecture configuration    |
| `models/truthlens_model/tokenizer.json`     | Tokenizer vocabulary                |
| `models/truthlens_model/pytorch_model.bin`  | Trained model weights               |
| `models/tfidf_vectorizer.joblib`            | Fitted TF-IDF vectorizer            |
| `logs/training.log`                         | Full training log                   |
| `reports/evaluation_results.json`           | Final evaluation metrics            |
| `reports/confusion_matrix.png`              | Confusion matrix visualization      |
| `reports/data_cleaning_report.json`         | Dataset processing summary          |
| `data/splits/train.csv`                     | Training split                      |
| `data/splits/validation.csv`                | Validation split                    |
| `data/splits/test.csv`                      | Test split                          |

---

# 10. Troubleshooting

### Model not found — `/health` returns `"status": "degraded"`

Run training first:

```bash
python main.py
```

Then verify model files:

```bash
ls models/truthlens_model/
# config.json  tokenizer.json  pytorch_model.bin  ...
```

### Training is slow

Options:
* Reduce `training.epochs` (try 2)
* Reduce `training.batch_size` (try 4)
* Use a lighter encoder: `model.encoder.name: distilroberta-base`
* Disable augmentation: `augmentation.enabled: false`

### Configuration file not found

Always run commands from the project root:

```bash
cd /home/runner/workspace
python main.py
```

### Hyperparameter tuning fails — Optuna not installed

```bash
pip install optuna
```

For full troubleshooting, see [documentation/TROUBLESHOOTING.md](documentation/TROUBLESHOOTING.md).
