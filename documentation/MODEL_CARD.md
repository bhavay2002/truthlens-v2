# Model Card — TruthLens AI

This document describes the **TruthLens AI model**, including its architecture, training data, input/output formats, evaluation metrics, intended uses, and known limitations.

---

## Model Overview

| Attribute          | Value                                                                 |
|--------------------|-----------------------------------------------------------------------|
| Model Name         | TruthLens AI                                                          |
| Version            | 2.0.0                                                                 |
| Model Type         | Transformer-based multi-task NLP system                               |
| Base Architecture  | RoBERTa (`roberta-base`)                                              |
| Tasks              | Fake news detection, bias detection, propaganda detection, ideology, emotion, narrative |
| Framework          | PyTorch + Hugging Face Transformers                                   |
| Tokenizer          | RoBERTa tokenizer, max length 256                                     |
| Interface          | FastAPI REST service (port 5000)                                      |
| Language Support   | English                                                               |
| Trained Model Path | `models/truthlens_model/`                                             |

---

## Model Architecture

TruthLens uses a **shared encoder with six task-specific classification heads**:

```
Article Text
       ↓
RoBERTa Tokenizer (max_length=256)
       ↓
Shared RoBERTa Encoder (roberta-base, 12 layers, 768 hidden dim)
       ↓
┌───────────────────────────────────────────────────────────┐
│  Task Heads                                               │
│                                                           │
│  ├── Bias Detection        3-class classification         │
│  │                         (cross_entropy loss)           │
│  │                                                        │
│  ├── Ideology Detection    3-class classification         │
│  │                         (cross_entropy loss)           │
│  │                                                        │
│  ├── Propaganda Detection  Binary classification          │
│  │                         (cross_entropy loss)           │
│  │                                                        │
│  ├── Emotion Detection     20-label multi-label           │
│  │                         (binary_cross_entropy loss)    │
│  │                                                        │
│  ├── Narrative Roles       Hero / Villain / Victim        │
│  │                         (binary_cross_entropy loss)    │
│  │                                                        │
│  └── Frame Detection       RE / HI / CO / MO / EC        │
│                            (binary_cross_entropy loss)    │
└───────────────────────────────────────────────────────────┘
       ↓
Per-task softmax / sigmoid outputs
```

**Architecture settings:**
- Shared encoder: `roberta-base`
- Dropout: 0.1
- Architecture type: `multitask_transformer`

---

## Training Data

TruthLens is trained on multiple public datasets unified under a shared label schema:

| Task                    | Datasets                               |
|-------------------------|----------------------------------------|
| Fake News Detection     | ISOT, LIAR, FakeNewsNet                |
| Bias Detection          | BABE, BASIL, MBIC                      |
| Emotion Classification  | GoEmotions, SemEval                    |
| Ideology Detection      | AllSides                               |
| Narrative Analysis      | FrameNet                               |
| Propaganda Detection    | PTC Propaganda Dataset                 |

Datasets are stored in `data/raw/` organized by task, then merged into `data/processed/unified_dataset.csv`.

**Data split ratios:**
- Train: 70%
- Validation: 15%
- Test: 15%

**Preprocessing steps applied:**
1. Unicode normalization
2. URL removal
3. HTML tag stripping
4. Contraction expansion
5. Lowercasing
6. Whitespace normalization
7. Minimum word count filtering (≥30 words)

---

## Training Procedure

**Entry point:** `python main.py`

**Training pipeline:**
```
Dataset Loading (data/splits/train.csv)
       ↓
Feature Engineering (TF-IDF + lexical features)
       ↓
MultiTaskTruthLensModel initialization
       ↓
Multi-task training loop
  - Optimizer: AdamW (weight_decay=0.01)
  - Learning rate: 2.0e-5
  - Scheduler: linear warmup (warmup_ratio=0.1)
  - Batch size: 8
  - Gradient accumulation steps: 2
  - Gradient clipping: 1.0
  - FP16: enabled
  - Epochs: 4
  - Early stopping: patience=2, metric=eval_loss
       ↓
Checkpoint saved to models/truthlens_model/
```

Hyperparameter tuning is supported via Optuna (`hyperparameter_tuning.enabled: true` in config).

---

## Input Format

The model accepts **news article text** as input.

Constraints:
- Minimum length: 10 characters
- Maximum length: 10,000 characters
- Language: English

**`POST /predict` request:**
```json
{
  "text": "Breaking news: Scientists discover a new species in the Amazon rainforest."
}
```

**`POST /analyze` request:**
```json
{
  "text": "Breaking news: Scientists discover a new species in the Amazon rainforest."
}
```

---

## Output Format

### `/predict` response

```json
{
  "text": "Breaking news: Scientists discover a new sp...",
  "prediction": "REAL",
  "fake_probability": 0.0812,
  "confidence": 0.9188
}
```

| Field             | Type   | Description                                      |
|-------------------|--------|--------------------------------------------------|
| `text`            | string | First 100 characters of the input text           |
| `prediction`      | string | `"REAL"` or `"FAKE"`                             |
| `fake_probability`| float  | Probability the article is fake (0.0–1.0)        |
| `confidence`      | float  | Model confidence in the predicted class (0.0–1.0)|

### `/analyze` response

```json
{
  "text": "Breaking news: Scientists discover a new sp...",
  "prediction": "REAL",
  "fake_probability": 0.0812,
  "confidence": 0.9188,
  "bias": {
    "bias_score": 0.0312,
    "media_bias": "center",
    "biased_tokens": [],
    "sentence_heatmap": [
      { "sentence": "Breaking news: Scientists discover...", "bias_score": 0.0 }
    ]
  },
  "emotion": {
    "dominant_emotion": "neutral",
    "emotion_scores": { "joy": 0.0, "fear": 0.0, "anger": 0.0, "...": "..." },
    "emotion_distribution": { "joy": 0.0, "fear": 0.0, "...": "..." }
  },
  "explainability": {
    "emotion_explanation": { "...": "..." },
    "lime": {
      "text": "...",
      "important_features": [
        { "feature": "scientists", "weight": 0.045 },
        { "feature": "discover", "weight": 0.038 }
      ]
    }
  }
}
```

**Media bias levels:**
- `"center"` — bias_score < 0.05
- `"lean"` — bias_score 0.05–0.15
- `"strong"` — bias_score ≥ 0.15

**Emotion labels (20-label set):** admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral

---

## Evaluation Metrics

| Metric       | Task Type                     |
|--------------|-------------------------------|
| Accuracy     | All classification tasks      |
| Precision    | All classification tasks      |
| Recall       | All classification tasks      |
| F1 Score     | All classification tasks      |
| Micro F1     | Multi-label (emotion)         |
| Macro F1     | Multi-label (emotion)         |
| ROC-AUC      | Multi-label (emotion)         |

Evaluation results are saved to `reports/evaluation_results.json`. Confusion matrices are saved to `reports/confusion_matrix.png`.

---

## Health Check

The `/health` endpoint reports model availability:

```json
{
  "status": "healthy",
  "model_path": "/workspace/models/truthlens_model",
  "model_exists": true,
  "model_files_complete": true,
  "training_text_column": "text",
  "vectorizer_required": false,
  "vectorizer_exists": true,
  "vectorizer_fallback_enabled": true,
  "vectorizer_effective_ready": true,
  "vectorizer_path": "/workspace/models/tfidf_vectorizer.joblib"
}
```

Status values: `"healthy"` · `"degraded"` (model not trained yet) · `"unhealthy"` (startup error)

---

## Intended Use

TruthLens AI is intended for:

- Misinformation detection research and academic NLP
- Media credibility analysis tools
- Journalism assistance and newsroom fact-checking support
- News monitoring and aggregation platforms
- Building explainable credibility scoring systems

**Example deployment scenarios:**
- Browser extension that evaluates news articles in real time
- Backend service for a fact-checking dashboard
- Research tool for studying narrative framing in media

---

## Out-of-Scope Use

TruthLens **must not be used as the sole authority** for determining factual truth.

- It cannot replace human fact-checkers or domain experts
- It should not be the basis for automated content removal or censorship without human review
- It may misclassify satire, parody, or opinion pieces as fake news
- It is not suitable for languages other than English without retraining

---

## Ethical Considerations

**Bias in training data:** The training datasets reflect political and cultural perspectives present in English-language media. The model's outputs may reflect these biases.

**Misuse risk:** Automated credibility scoring could be weaponized for political manipulation or targeted suppression of legitimate journalism. Human oversight is essential.

**Transparency:** Explainability modules (SHAP, LIME) are included to give users visibility into *why* a prediction was made, not just what the prediction is.

**Accountability:** No prediction should be acted upon without a human reviewing the explanation output alongside it.

---

## Limitations

- Trained primarily on **English-language datasets** — limited effectiveness on other languages
- May struggle with **satire, parody, and opinion journalism** that uses exaggerated language
- **Factual verification is not performed** — the system evaluates linguistic and structural signals, not truth of claims
- Performance depends on **dataset quality and coverage** of the specific domain
- Transformer encoder has a **maximum input of 256 tokens** — longer articles are truncated
- Model must be **trained before inference is available** — health endpoint reports `degraded` status on a fresh deployment

---

## Versioning

Model versions are tracked via:
- Checkpoint directory: `models/truthlens_model/`
- Required files: `config.json`, `tokenizer.json`, `model.safetensors` (or `pytorch_model.bin`)
- Model registry: `src/models/registry/model_registry.py`

Each model version records: configuration, training hyperparameters, dataset sources, and evaluation results.
