# Feature Engineering

This document describes the **feature engineering system used in TruthLens AI**.

Feature engineering converts **raw article text into structured signals** that can be used by machine learning models and analysis modules. TruthLens uses a **multi-layer feature system** that is modular, interpretable, and extensible.

---

## Overview

```
Article Text
       ↓
Text Preprocessing (cleaning, normalization)
       ↓
Tokenization
       ↓
┌──────────────────────────────────────────────────┐
│  Parallel Feature Extractors                     │
│  ├── Text Features (lexical, semantic, syntactic)│
│  ├── Bias Features (lexicon density, framing)    │
│  ├── Emotion Features (20-label intensity)       │
│  ├── Narrative Features (frame, roles)           │
│  ├── Propaganda Features (manipulation patterns) │
│  ├── Discourse Features (argument structure)     │
│  └── Graph Features (entity interactions)        │
└──────────────────────────────────────────────────┘
       ↓
Feature Fusion (scaling, selection)
       ↓
Unified Feature Representation → Model Input
```

All feature extractors are located in `src/features/` and inherit from `BaseFeature` via a `FeatureContext` object.

---

## Base System

### `FeatureContext`

Every feature extractor receives a `FeatureContext` object as input:

```python
from src.features.base.base_feature import FeatureContext

context = FeatureContext(text="News article text here...")
```

`FeatureContext` standardizes the input surface so any feature extractor can be composed into a pipeline without knowing about other extractors.

### `BaseFeature`

All extractors implement a single `.extract(context: FeatureContext) -> dict` method that returns a flat dictionary of feature names to numeric values:

```python
extractor = BiasLexiconFeatures()
features = extractor.extract(context)
# → {"bias_lexicon_density": 0.04, "evaluative_word_count": 2, ...}
```

---

## Feature Categories

| Category    | Location                    | Description                           |
|-------------|-----------------------------|---------------------------------------|
| Text        | `src/features/text/`        | Basic linguistic properties           |
| Bias        | `src/features/bias/`        | Ideological bias and framing signals  |
| Emotion     | `src/features/emotion/`     | Emotional tone and intensity          |
| Narrative   | `src/features/narrative/`   | Story structure and frame detection   |
| Propaganda  | `src/features/propaganda/`  | Manipulative rhetoric patterns        |
| Discourse   | `src/features/discourse/`   | Argument structure and coherence      |
| Graph       | `src/features/graph/`       | Entity interaction signals            |

---

## Text Features — `src/features/text/`

Capture **basic linguistic properties** of articles.

| Module                    | Features Extracted                                        |
|---------------------------|-----------------------------------------------------------|
| `lexical_features.py`     | Token count, unique token ratio, lexical diversity        |
| `semantic_features.py`    | Contextual embedding signals, semantic density            |
| `syntactic_features.py`   | Sentence length distribution, parse tree complexity       |
| `token_features.py`       | TF-IDF top terms, character n-gram statistics             |

**Example features:**
- `avg_sentence_length` — average word count per sentence
- `lexical_diversity` — unique tokens / total tokens
- `tfidf_top_terms` — top-N TF-IDF weighted terms per document

---

## Bias Features — `src/features/bias/`

Detect **ideological framing and partisan language** patterns.

| Module                      | Features Extracted                                    |
|-----------------------------|-------------------------------------------------------|
| `bias_lexicon_features.py`  | Lexicon density, evaluative/assertive word counts     |
| `bias_features.py`          | Aggregate bias signal composition                     |
| `framing_features.py`       | Narrative framing strategy indicators                 |
| `ideological_features.py`   | Political ideology markers                            |

**Key feature — `bias_lexicon_density`:**
Ratio of bias-indicative tokens to total article tokens. Used directly in `compute_bias_features()` to determine media bias level:
- `< 0.05` → `"center"`
- `0.05–0.15` → `"lean"`
- `≥ 0.15` → `"strong"`

**API wrapper:** `src/features/bias/bias_lexicon.py` provides `compute_bias_features(text)` which returns a `BiasResult` containing:
- `bias_score` — normalized float (0.0–1.0)
- `media_bias` — "center" / "lean" / "strong"
- `biased_tokens` — list of detected bias-loaded words
- `sentence_heatmap` — per-sentence bias scores

---

## Emotion Features — `src/features/emotion/`

Analyze the **emotional tone of articles** across 20 emotion labels.

| Module                           | Features Extracted                              |
|----------------------------------|-------------------------------------------------|
| `emotion_lexicon_features.py`    | Per-emotion lexicon match counts                |
| `emotion_features.py`            | Aggregate emotion signal vector                 |
| `emotion_intensity_features.py`  | Emotional intensity and volatility              |
| `emotion_trajectory_features.py` | Emotional arc across article sections           |
| `emotion_target_features.py`     | Target entities of emotional language           |
| `emotion_schema.py`              | Canonical 20-label emotion schema (`EMOTION_LABELS`) |

**Emotion labels:** admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral

**Feature key format:** `lexicon_emotion_{emotion_name}` → float score per emotion

**API wrapper:** `src/features/emotion/emotion_lexicon.py` provides `EmotionLexiconAnalyzer().analyze(text)` which returns an `EmotionResult` containing:
- `dominant_emotion` — the highest-scoring emotion (or `"neutral"` if all scores are 0)
- `emotion_scores` — dict of all 20 emotion scores
- `emotion_distribution` — normalized probability distribution

---

## Narrative Features — `src/features/narrative/`

Capture **story structure and framing techniques** used in articles.

| Module                       | Features Extracted                                   |
|------------------------------|------------------------------------------------------|
| `narrative_features.py`      | General narrative signal composition                 |
| `narrative_frame_features.py`| Frame category detection (Resolution, Human Interest, Conflict, Moral, Economic) |
| `narrative_role_features.py` | Hero / Villain / Victim role assignment              |
| `conflict_features.py`       | Adversarial framing and conflict intensity           |

**Narrative frame categories (from config):**
| Code | Frame           |
|------|-----------------|
| RE   | Resolution      |
| HI   | Human Interest  |
| CO   | Conflict        |
| MO   | Moral           |
| EC   | Economic        |

---

## Propaganda Features — `src/features/propaganda/`

Detect **persuasive and manipulative rhetoric** patterns.

| Module                           | Features Extracted                          |
|----------------------------------|---------------------------------------------|
| `propaganda_features.py`         | Aggregate propaganda signal                 |
| `propaganda_lexicon_features.py` | Loaded language, flag words, emotionalization |
| `manipulation_patterns.py`       | Fear appeals, exaggeration, false dichotomy |

These features directly feed the propaganda detection task head and the aggregation scoring engine.

---

## Discourse Features — `src/features/discourse/`

Analyze the **argument structure and logical coherence** of articles.

| Module                          | Features Extracted                         |
|---------------------------------|--------------------------------------------|
| `discourse_features.py`         | Discourse connective analysis              |
| `argument_structure_features.py`| Claim–evidence relationship detection      |

Discourse coherence is used as a positive credibility signal in the aggregation layer — well-structured arguments are more likely to reflect factual reporting.

---

## Graph Features — `src/features/graph/`

Capture **relationships between named entities and interactions** in articles.

| Module                         | Features Extracted                            |
|--------------------------------|-----------------------------------------------|
| `entity_graph_features.py`     | Entity co-occurrence, centrality, connectivity|
| `interaction_graph_features.py`| Interaction pattern density, narrative hubs   |

Graph construction uses **spaCy NER** for entity extraction and **NetworkX** for graph computation.

---

## Feature Fusion — `src/features/fusion/`

After individual features are extracted, they are combined into a unified representation.

```
Multiple Feature Dictionaries
       ↓
Feature Normalization / Scaling (StandardScaler / MinMaxScaler)
       ↓
Feature Selection (variance threshold, top-K importance)
       ↓
Unified Feature Vector
```

| Module                 | Purpose                                     |
|------------------------|---------------------------------------------|
| `feature_fusion.py`    | Combines feature dicts from all extractors  |
| `feature_scaling.py`   | Normalizes numerical feature values         |
| `feature_selection.py` | Drops low-variance or redundant features    |

---

## Feature Pipelines — `src/features/pipelines/`

Feature pipelines orchestrate the complete extraction workflow.

| Pipeline                   | Purpose                                    |
|----------------------------|--------------------------------------------|
| `feature_pipeline.py`      | Standard single-article feature extraction |
| `batch_feature_pipeline.py`| Batch processing for dataset-scale runs    |

**Pipeline flow:**
```
Input Text / Dataset
       ↓
Preprocessing
       ↓
Feature Extractors (all categories)
       ↓
Feature Fusion
       ↓
Output Feature Matrix (ready for model or training)
```

---

## Feature Caching — `src/features/cache/`

Feature extraction is computationally expensive, especially for large datasets. TruthLens includes a caching layer.

| Module            | Purpose                            |
|-------------------|------------------------------------|
| `cache_manager.py`| Cache lifecycle management          |
| `feature_cache.py`| Stores and retrieves feature outputs|

Caching speeds up repeated training experiments without re-running the full extraction pipeline.

---

## Feature Importance — `src/features/importance/`

Tools to analyze **which features most influence predictions**.

| Module                      | Method                                     |
|-----------------------------|---------------------------------------------|
| `permutation_importance.py` | Shuffles features and measures accuracy drop|
| `shap_importance.py`        | SHAP values for global feature attribution  |
| `feature_ablation.py`       | Removes feature groups and measures impact  |

---

## Feature Validation — `src/features/`

| Module                       | Purpose                                      |
|------------------------------|----------------------------------------------|
| `feature_schema_validator.py`| Validates feature dict keys and value types  |
| `feature_statistics.py`      | Computes distribution stats for all features |
| `dataset_feature_generator.py`| Generates features for a full dataset CSV   |

---

## TF-IDF Features (config-driven)

TF-IDF features are configured in `config/config.yaml`:

```yaml
features:
  tfidf:
    enabled: true
    max_features: 5000
    top_terms_per_doc: 4
```

The TF-IDF vectorizer is trained on the training set and persisted to `models/tfidf_vectorizer.joblib`.

---

## Adding a New Feature Extractor

1. Create a new module in the appropriate `src/features/{category}/` directory.
2. Subclass `BaseFeature` and implement the `.extract(context)` method.
3. Return a flat `dict[str, float]` of feature names to values.
4. Register the extractor in the relevant feature pipeline.
5. Add tests in `tests/test_features/`.

Example:

```python
from src.features.base.base_feature import BaseFeature, FeatureContext

class MyNewFeature(BaseFeature):
    def extract(self, context: FeatureContext) -> dict:
        tokens = context.text.lower().split()
        return {
            "my_feature_word_count": len(tokens),
        }
```

---

## Design Principles

**Modularity** — Each feature extractor is independent. Adding or removing one does not affect others.

**Interpretability** — Feature names are human-readable strings that map directly to linguistic concepts.

**Extensibility** — The `BaseFeature` / `FeatureContext` pattern makes it straightforward to add new signal types.

**Efficiency** — Feature caching and batch pipelines support large-scale dataset processing without redundant computation.
