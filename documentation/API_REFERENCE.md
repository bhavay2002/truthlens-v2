# API Reference

This document is the **complete REST API reference** for TruthLens AI.

The API is built with FastAPI and served on port 5000. Interactive Swagger documentation is available at `/docs` when the server is running.

**Base URL (development):** `http://localhost:5000`  
**Base URL (production):** your Replit deployment URL

---

## Authentication

The API does not require authentication in its default configuration. Add authentication middleware to `api/app.py` if deploying publicly.

---

## Endpoints

### `GET /`

Home endpoint. Confirms the API is online and lists all available endpoints.

**Response `200 OK`:**
```json
{
  "message": "TruthLens AI API",
  "status": "online",
  "endpoints": {
    "predict": "/predict",
    "analyze": "/analyze",
    "health": "/health",
    "project_view": "/project-view",
    "docs": "/docs"
  }
}
```

---

### `GET /health`

Detailed health check. Reports model file availability and readiness for inference.

**Response `200 OK` — model ready:**
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

**Response `200 OK` — model not trained yet:**
```json
{
  "status": "degraded",
  "model_path": "/workspace/models/truthlens_model",
  "model_exists": false,
  "model_files_complete": false,
  "training_text_column": "text",
  "vectorizer_required": false,
  "vectorizer_exists": false,
  "vectorizer_fallback_enabled": true,
  "vectorizer_effective_ready": true,
  "vectorizer_path": "/workspace/models/tfidf_vectorizer.joblib"
}
```

**Status values:**
| Status      | Meaning                                               |
|-------------|-------------------------------------------------------|
| `healthy`   | Model and vectorizer are present and complete         |
| `degraded`  | Model files are missing — training required           |
| `unhealthy` | An exception occurred during the health check itself  |

**Required model files** (checked inside `model_path/`):
- `config.json`
- `tokenizer.json`
- `model.safetensors` or `pytorch_model.bin`

---

### `POST /predict`

Binary fake/real classification of a news article.

**Request body:**
```json
{
  "text": "News article text to analyze. Must be between 10 and 10,000 characters."
}
```

| Field  | Type   | Required | Min | Max    | Description              |
|--------|--------|----------|-----|--------|--------------------------|
| `text` | string | Yes      | 10  | 10,000 | News article text to analyze |

**Response `200 OK`:**
```json
{
  "text": "News article text to analyze. Must be between...",
  "prediction": "FAKE",
  "fake_probability": 0.8741,
  "confidence": 0.8741
}
```

| Field              | Type   | Range  | Description                                         |
|--------------------|--------|--------|-----------------------------------------------------|
| `text`             | string | —      | First 100 characters of the input (configurable)    |
| `prediction`       | string | —      | `"FAKE"` or `"REAL"`                               |
| `fake_probability` | float  | 0–1    | Probability the article is fake news                |
| `confidence`       | float  | 0–1    | Model confidence in the predicted class             |

**Example request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "BREAKING: Scientists have confirmed that eating chocolate every day cures cancer, according to a new miracle study that mainstream media is hiding from you."}'
```

**Error responses:**
| Code | Condition                            |
|------|--------------------------------------|
| 400  | Text too short (< 10 chars) or too long (> 10,000 chars) |
| 503  | Model files not found — train the model first |
| 500  | Internal inference error             |

---

### `POST /analyze`

Full article analysis including bias detection, emotion analysis, and LIME-based explainability. This endpoint runs the complete analysis pipeline and takes longer than `/predict`.

**Request body:**
```json
{
  "text": "News article text to analyze."
}
```

Same validation as `/predict` — text between 10 and 10,000 characters.

**Response `200 OK`:**
```json
{
  "text": "News article text to analyze...",
  "prediction": "FAKE",
  "fake_probability": 0.8741,
  "confidence": 0.8741,
  "bias": {
    "bias_score": 0.1823,
    "media_bias": "strong",
    "biased_tokens": ["allegedly", "shocking", "radical"],
    "sentence_heatmap": [
      {
        "sentence": "Scientists have allegedly confirmed shocking results",
        "bias_score": 0.3333
      },
      {
        "sentence": "This radical study changes everything",
        "bias_score": 0.1667
      }
    ]
  },
  "emotion": {
    "dominant_emotion": "fear",
    "emotion_scores": {
      "admiration": 0.0,
      "anger": 0.0,
      "fear": 0.0821,
      "joy": 0.0,
      "sadness": 0.0312,
      "surprise": 0.0,
      "neutral": 0.0
    },
    "emotion_distribution": {
      "admiration": 0.0,
      "anger": 0.0,
      "fear": 0.7241,
      "joy": 0.0,
      "sadness": 0.2759,
      "surprise": 0.0,
      "neutral": 0.0
    }
  },
  "explainability": {
    "emotion_explanation": {
      "dominant_emotion": "fear",
      "intensity": 0.0821,
      "trigger_words": ["cancer", "hiding", "mainstream media"],
      "sentence_breakdown": [...]
    },
    "lime": {
      "text": "BREAKING: Scientists have confirmed...",
      "important_features": [
        { "feature": "hiding", "weight": 0.1234 },
        { "feature": "miracle", "weight": 0.0987 },
        { "feature": "mainstream", "weight": 0.0876 },
        { "feature": "confirmed", "weight": -0.0654 }
      ]
    }
  }
}
```

**Response fields:**

**Top level:**
| Field              | Type   | Description                                    |
|--------------------|--------|------------------------------------------------|
| `text`             | string | Input text preview (first 100 chars)           |
| `prediction`       | string | `"FAKE"` or `"REAL"`                          |
| `fake_probability` | float  | Fake news probability (0.0–1.0)                |
| `confidence`       | float  | Model confidence (0.0–1.0)                     |
| `bias`             | object | Bias analysis result                           |
| `emotion`          | object | Emotion analysis result                        |
| `explainability`   | object | Explanations for emotion and prediction        |

**`bias` object:**
| Field             | Type         | Description                                        |
|-------------------|--------------|----------------------------------------------------|
| `bias_score`      | float        | Lexicon-based bias density (0.0–1.0)               |
| `media_bias`      | string       | `"center"` / `"lean"` / `"strong"`               |
| `biased_tokens`   | string[]     | List of bias-loaded words detected                 |
| `sentence_heatmap`| object[]     | Per-sentence `{sentence, bias_score}` breakdowns   |

**`emotion` object:**
| Field                 | Type        | Description                                    |
|-----------------------|-------------|------------------------------------------------|
| `dominant_emotion`    | string      | Highest-scoring emotion label (or `"neutral"`) |
| `emotion_scores`      | object      | Raw score per emotion label (0.0–1.0)          |
| `emotion_distribution`| object      | Normalized probability distribution (sums to 1)|

**`explainability.lime` object:**
| Field                | Type      | Description                                     |
|----------------------|-----------|-------------------------------------------------|
| `text`               | string    | Input text                                      |
| `important_features` | object[]  | `[{feature: string, weight: float}]` — positive weights push toward FAKE, negative toward REAL |
| `error`              | string    | Present only if LIME failed; value: `"lime_unavailable"` |

**Example request:**
```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "BREAKING: Scientists have confirmed that eating chocolate every day cures cancer, according to a new miracle study that mainstream media is hiding from you."}'
```

**Error responses:**
| Code | Condition                                |
|------|------------------------------------------|
| 400  | Input validation failure                 |
| 503  | Model not available (not yet trained)    |
| 500  | Internal analysis error                  |

---

### `GET /project-view`

Returns metadata about the API configuration, model settings, and directory structure. Useful for debugging and monitoring deployments.

**Response `200 OK`:**
```json
{
  "project_root": "/workspace",
  "api": {
    "title": "TruthLens AI API",
    "version": "2.0.0",
    "description": "Multi-task NLP system for bias, ideology, propaganda, emotion, and narrative analysis"
  },
  "config": {
    "model_name": "roberta-base",
    "model_path": "/workspace/models/truthlens_model",
    "training_text_column": "text",
    "vectorizer_path": "/workspace/models/tfidf_vectorizer.joblib"
  },
  "structure": {
    "src_exists": true,
    "api_exists": true,
    "config_exists": true,
    "tests_exists": true,
    "models_package_init_exists": true,
    "model_subpackages": {
      "emotion": { "directory_exists": true, "package_init_exists": true },
      "encoder": { "directory_exists": true, "package_init_exists": true },
      "ideology": { "directory_exists": true, "package_init_exists": true },
      "multitask": { "directory_exists": true, "package_init_exists": true },
      "narrative": { "directory_exists": true, "package_init_exists": true },
      "propaganda": { "directory_exists": true, "package_init_exists": true }
    }
  }
}
```

---

### `GET /docs`

Interactive Swagger UI documentation. Available when the server is running.

Navigate to `http://localhost:5000/docs` in your browser to:
- Browse all endpoints
- View request/response schemas
- Send test requests directly from the browser

---

## Data Models

### `NewsRequest`

Input model for `/predict` and `/analyze`.

```json
{
  "text": "string (10–10,000 chars)"
}
```

**Example:**
```json
{
  "text": "Breaking news: Scientists discover a new species in the Amazon rainforest."
}
```

### `NewsResponse`

Output model from `/predict`.

```json
{
  "text": "string",
  "fake_probability": "float (0.0–1.0)",
  "prediction": "string (FAKE | REAL)",
  "confidence": "float (0.0–1.0)"
}
```

### `AnalysisResponse`

Output model from `/analyze`.

```json
{
  "text": "string",
  "prediction": "string (FAKE | REAL)",
  "fake_probability": "float (0.0–1.0)",
  "confidence": "float (0.0–1.0)",
  "bias": {
    "bias_score": "float",
    "media_bias": "string (center | lean | strong)",
    "biased_tokens": "string[]",
    "sentence_heatmap": [{ "sentence": "string", "bias_score": "float" }]
  },
  "emotion": {
    "dominant_emotion": "string",
    "emotion_scores": "object (emotion → float)",
    "emotion_distribution": "object (emotion → float)"
  },
  "explainability": {
    "emotion_explanation": "object",
    "lime": {
      "text": "string",
      "important_features": [{ "feature": "string", "weight": "float" }]
    }
  }
}
```

---

## Error Responses

All errors return JSON with a `detail` field:

```json
{
  "detail": "Human-readable error message"
}
```

| HTTP Code | When returned                                            |
|-----------|----------------------------------------------------------|
| 400       | Text validation failure (too short, too long)            |
| 422       | Request body malformed (missing `text` field, wrong type)|
| 500       | Internal server error during inference or analysis       |
| 503       | Model unavailable — run `python main.py` to train        |

---

## Configuration

The API reads settings from `config/config.yaml` via `src/utils/settings.py`. Key API settings:

```yaml
api:
  title: TruthLens AI API
  description: Multi-task NLP system for bias, ideology, propaganda, emotion, and narrative analysis
  version: 2.0.0
  text_preview_chars: 100   # Number of input chars shown in response 'text' field

inference:
  allow_raw_text_fallback: true   # Use raw text if vectorizer unavailable
```

---

## Rate Limiting and Performance

- `/predict` — typically responds in under 1 second once the model is warmed up
- `/analyze` — slower due to LIME (256 samples) and parallel feature extraction; expect 2–10 seconds depending on article length and hardware
- First request after startup is slower (model loading); subsequent requests use the cached model

---

## Python Client Example

```python
import requests

BASE_URL = "http://localhost:5000"

article = """
Scientists at a major university have published findings suggesting that 
regular moderate exercise significantly reduces the risk of cardiovascular 
disease, confirming decades of prior research on the subject.
"""

# Quick prediction
response = requests.post(f"{BASE_URL}/predict", json={"text": article})
result = response.json()
print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.2%})")

# Full analysis
response = requests.post(f"{BASE_URL}/analyze", json={"text": article})
analysis = response.json()
print(f"Dominant emotion: {analysis['emotion']['dominant_emotion']}")
print(f"Media bias: {analysis['bias']['media_bias']}")
print(f"Top LIME feature: {analysis['explainability']['lime']['important_features'][0]}")
```
