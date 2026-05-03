# Configuration Reference

This document describes all configuration options available in TruthLens AI.

Configuration is split across two YAML files:
- `config/config.yaml` — model architecture, training, API, and inference settings
- `config/data_config.yaml` — data pipeline, preprocessing, augmentation, and EDA settings

Settings are loaded via `src/utils/settings.py` (primary interface used by the API and training) and `src/utils/config_loader.py` (low-level loader with dataclass conversion).

---

## `config/config.yaml`

### `model.encoder`

Controls the base transformer encoder used by the multi-task model.

```yaml
model:
  encoder:
    name: roberta-base             # HuggingFace model ID or local path
    tokenizer_name: roberta-base   # Tokenizer to use (usually same as encoder)
    max_length: 256                # Maximum token length per input
    cache_dir: models/cache        # Where to cache downloaded model weights
```

| Key              | Default        | Description                                    |
|------------------|----------------|------------------------------------------------|
| `name`           | `roberta-base` | Encoder model identifier                       |
| `tokenizer_name` | `roberta-base` | Tokenizer identifier                           |
| `max_length`     | `256`          | Tokens per input (longer inputs are truncated) |
| `cache_dir`      | `models/cache` | HuggingFace download cache directory           |

---

### `model.architecture`

Controls the shared architecture style.

```yaml
model:
  architecture:
    type: multitask_transformer    # Architecture type
    shared_encoder: true           # Whether to share the encoder across heads
    dropout: 0.1                   # Dropout rate applied in task heads
```

---

### `model.heads`

Defines the task-specific classification heads attached to the shared encoder.

```yaml
model:
  heads:
    bias_detection:
      num_labels: 3          # Number of bias classes
      loss: cross_entropy
      label_column: bias_label

    ideology_detection:
      num_labels: 3
      loss: cross_entropy
      label_column: ideology_label

    propaganda_detection:
      num_labels: 2          # Binary: propaganda vs not
      loss: cross_entropy
      label_column: propaganda_label

    emotion_detection:
      num_labels: 20         # 20-label multi-label classification
      type: multi_label
      loss: binary_cross_entropy
      label_prefix: emotion_ # Column prefix in dataset (emotion_joy, emotion_fear, etc.)

    narrative_roles:
      hero: hero             # Column name for hero binary label
      villain: villain
      victim: victim
      loss: binary_cross_entropy

    frame_detection:
      labels:
        - RE                 # Resolution
        - HI                 # Human Interest
        - CO                 # Conflict
        - MO                 # Moral
        - EC                 # Economic
      loss: binary_cross_entropy
```

---

### `model.path`

Path where the trained model artifacts are saved and loaded from.

```yaml
model:
  path: models/truthlens_model
```

This is a relative path from the project root. After training, this directory will contain `config.json`, `tokenizer.json`, and `model.safetensors`.

---

### `training`

Controls all training hyperparameters.

```yaml
training:
  seed: 42                          # Random seed for reproducibility
  device: auto                      # "auto", "cpu", "cuda", or "mps"

  epochs: 4                         # Total training epochs
  batch_size: 8                     # Per-device batch size
  gradient_accumulation_steps: 2    # Effective batch = batch_size × accumulation

  learning_rate: 2.0e-5             # Peak learning rate for AdamW
  weight_decay: 0.01                # L2 regularization strength
  warmup_ratio: 0.1                 # Fraction of training steps used for warmup

  scheduler: linear                 # LR scheduler type
  optimizer: adamw                  # Optimizer type

  fp16: true                        # Enable mixed-precision training (requires CUDA)
  gradient_clipping: 1.0            # Max gradient norm

  resume_from_checkpoint: false     # Resume training from latest checkpoint

  text_column: text                 # Column name in dataset containing article text
  title_column: title               # Column name for article title (optional)

  early_stopping:
    enabled: true
    patience: 2                     # Stop after N epochs without improvement
    metric: eval_loss               # Metric to monitor
```

| Key                          | Default  | Description                                |
|------------------------------|----------|--------------------------------------------|
| `seed`                       | `42`     | Random seed for all RNG sources            |
| `device`                     | `auto`   | `auto` selects CUDA > MPS > CPU            |
| `epochs`                     | `4`      | Training epochs                            |
| `batch_size`                 | `8`      | Samples per gradient step                  |
| `gradient_accumulation_steps`| `2`      | Effective batch = 8 × 2 = 16              |
| `learning_rate`              | `2.0e-5` | AdamW peak learning rate                   |
| `weight_decay`               | `0.01`   | AdamW weight decay                         |
| `warmup_ratio`               | `0.1`    | Fraction of steps for LR warmup            |
| `scheduler`                  | `linear` | LR schedule type                           |
| `fp16`                       | `true`   | Mixed-precision (CUDA only)                |
| `gradient_clipping`          | `1.0`    | Max gradient norm                          |

---

### `data`

Paths to dataset split files used during training.

```yaml
data:
  train_path: data/splits/train.csv
  validation_path: data/splits/validation.csv
  test_path: data/splits/test.csv

  raw_dir: data/raw
  interim_dir: data/interim
  processed_dir: data/processed

  augmentation_multiplier: 2    # Multiply training data N times via augmentation
```

---

### `features`

Feature engineering settings.

```yaml
features:
  engineered_text_column: engineered_text    # Column name for TF-IDF engineered text

  tfidf:
    enabled: true
    max_features: 5000         # Maximum TF-IDF vocabulary size
    top_terms_per_doc: 4       # Top TF-IDF terms to prepend to each text
```

---

### `cross_validation`

Cross-validation settings (disabled by default).

```yaml
cross_validation:
  enabled: false
  splits: 5
  metric: eval_loss
```

---

### `hyperparameter_tuning`

Optuna-based hyperparameter search (disabled by default).

```yaml
hyperparameter_tuning:
  enabled: false
  trials: 10
  direction: minimize
  metric: eval_loss

  search_space:
    learning_rate:
      min: 1e-6
      max: 5e-5
    batch_size:
      - 8
      - 16
    epochs:
      - 3
      - 4
```

---

### `evaluation`

Metrics computed during and after training.

```yaml
evaluation:
  metrics:
    classification:
      - accuracy
      - precision
      - recall
      - f1
    multi_label:
      - micro_f1
      - macro_f1
      - roc_auc
```

---

### `logging`

Logging and checkpoint configuration.

```yaml
logging:
  log_level: INFO
  training_log_path: logs/training.log

  save_steps: 500           # Save checkpoint every N steps
  eval_steps: 500           # Run validation every N steps
  save_total_limit: 3       # Keep only last N checkpoints
```

---

### `paths`

Output directories and artifact file paths.

```yaml
paths:
  models_dir: models
  logs_dir: logs
  reports_dir: reports

  tfidf_vectorizer_path: models/tfidf_vectorizer.joblib

  evaluation_results_path: reports/evaluation_results.json
  confusion_matrix_path: reports/confusion_matrix.png
  cleaning_report_path: reports/data_cleaning_report.json
```

---

### `api`

FastAPI application metadata.

```yaml
api:
  title: TruthLens AI API
  description: Multi-task NLP system for bias, ideology, propaganda, emotion, and narrative analysis
  version: 2.0.0
  text_preview_chars: 100    # Characters of input text to show in API responses
```

---

### `inference`

Runtime inference settings.

```yaml
inference:
  batch_size: 16              # Batch size for batch inference calls
  device: auto                # Device for inference ("auto", "cpu", "cuda", "mps")
  allow_raw_text_fallback: true   # Use raw text if TF-IDF vectorizer is unavailable

  return_outputs:             # Toggle which outputs to include in full analysis
    bias: true
    ideology: true
    propaganda: true
    emotion: true
    narrative_roles: true
    frames: true
```

---

## `config/data_config.yaml`

### `project`

Project metadata (informational).

```yaml
project:
  name: TruthLens AI
  version: 2.0
  description: Multi-task NLP pipeline for bias, ideology, propaganda, emotion, narrative roles, and frame detection.
```

---

### `dataset.unified_schema`

Defines the column mapping used when merging datasets from different sources into a unified format.

```yaml
dataset:
  unified_schema:
    text_fields:
      - title
      - text
    label_fields:
      bias_label: bias_label
      ideology_label: ideology_label
      propaganda_label: propaganda_label
      frame: frame
      hero: hero
      villain: villain
      victim: victim
      emotion_prefix: emotion_    # Columns: emotion_joy, emotion_fear, etc.
      RE: resolution
      HI: human_interest
      CO: conflict
      MO: moral
      EC: economic
```

---

### `dataset.datasets`

Per-task dataset source directories.

```yaml
dataset:
  datasets:
    bias:
      path: data/raw/bias
      text_column: text
      label_column: bias_label

    ideology:
      path: data/raw/ideology
      text_column: text
      label_column: ideology_label

    propaganda:
      path: data/raw/propaganda
      text_column: text
      label_column: propaganda_label

    narrative:
      path: data/raw/narrative
      text_column: text
      hero_entities: hero_entities
      villain_entities: villain_entities
      victim_entities: victim_entities

    emotion:
      path: data/raw/emotion
      text_column: text
      emotion_columns_prefix: emotion_
```

---

### `validation`

Data quality validation thresholds applied before training.

```yaml
validation:
  required_columns:
    - text
  max_null_ratio: 0.10        # Reject dataset if >10% of values are null
  max_duplicate_ratio: 0.15   # Reject dataset if >15% of rows are duplicates
  min_text_length: 20         # Minimum character count per row
  min_word_count: 30          # Minimum word count per row
  label_checks:
    min_class_ratio: 0.10     # Each class must make up at least 10% of labels
    allow_missing_labels: true
```

---

### `cleaning`

Text cleaning operations applied to all datasets.

```yaml
cleaning:
  normalize_unicode: true    # Convert unicode characters to ASCII equivalents
  normalize_numbers: true    # Normalize numeric formats
  remove_emojis: true        # Strip emoji characters
  remove_urls: true          # Remove http/https URLs
  remove_html: true          # Strip HTML tags
  expand_contractions: true  # Expand "don't" → "do not", etc.
  lowercase: true
  strip_whitespace: true
  min_word_count: 30         # Drop rows with fewer than N words after cleaning
```

---

### `balancing`

Class imbalance handling.

```yaml
balancing:
  enabled: true
  method: oversample    # "oversample", "undersample", or "smote"
  random_state: 42
```

---

### `augmentation`

Text augmentation techniques to expand the training set.

```yaml
augmentation:
  enabled: true
  multiplier: 2    # Generate N augmented copies per original sample

  techniques:
    synonym_replacement: true
    random_swap: true
    random_deletion: true
    back_translation: false    # Disabled by default (requires translation API)
```

---

### `split`

Dataset split ratios.

```yaml
split:
  train_ratio: 0.70
  validation_ratio: 0.15
  test_ratio: 0.15
  stratified: true          # Maintain class proportions in each split
  random_state: 42
```

---

### `eda`

Exploratory Data Analysis settings.

```yaml
eda:
  enabled: true
  figures_dir: reports/figures
  top_words: 30
  max_tfidf_features: 10000
  ngrams:
    - 1
    - 2
    - 3
  plots:
    label_distribution: true
    text_length_distribution: true
    tfidf_top_terms: true
    emotion_distribution: true
```

Run EDA with: `python run_eda.py`

---

### `output`

Output file paths for processed datasets.

```yaml
output:
  unified_dataset: data/processed/unified_dataset.csv
  processed_data_dir: data/processed
  splits_dir: data/splits
  logs_dir: logs
  save_formats:
    - csv
    - parquet
```

---

## Accessing Configuration in Code

### Primary interface — `src/utils/settings.py`

Used by the API and main training pipeline:

```python
from src.utils.settings import load_settings

settings = load_settings()
print(settings.model.path)       # Path object to model directory
print(settings.training.epochs)  # Integer
print(settings.api.title)        # "TruthLens AI API"
```

### Low-level interface — `src/utils/config_loader.py`

Direct YAML access with nested key retrieval:

```python
from src.utils.config_loader import load_config, get_config_value

config = load_config()
max_len = get_config_value(config, "model", "encoder", "max_length", default=256)
```

### Structured dataclass — `load_app_config()`

Converts YAML to typed dataclasses:

```python
from src.utils.config_loader import load_app_config

app_config = load_app_config()
print(app_config.model.name)           # "roberta-base"
print(app_config.training.batch_size)  # 8
```

Both loaders use `@lru_cache` — the YAML file is read from disk once per process.
