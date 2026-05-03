# TruthLens AI — Resolved Configuration Values

This document fills the supplied YAML template with the values **actually used by the
project today**. For each key the source-of-truth file is shown in `[brackets]`.

Three categories of values appear:

- **Concrete** — value is set explicitly in a YAML file or Python dataclass default
  that the runtime reads.
- **Default-from-code** — value is not in any YAML file; it comes from a dataclass
  default in a loader (`src/utils/settings.py`, `src/models/config/model_config.py`,
  `src/inference/constants.py`, etc.). Marked `# default (code)`.
- **Not-implemented** — the template field has no consumer in the codebase and is
  effectively dead. Marked `# NOT IMPLEMENTED` with a one-line note explaining
  what the closest real knob is, if any.

The canonical config file is **`config/config.yaml`** (project root). The data-pipeline
file is **`config/data_config.yaml`**. The model-training schema in
`src/models/config/model_config.py` is a **separate** richer schema that
`ModelConfigLoader.load_multitask_config()` consumes — it is **not** wired to
`config/config.yaml` today, so its richer fields (per-head `loss_weight`, regression,
mixup, adversarial training, etc.) are listed with their dataclass defaults and a
note that no YAML currently feeds them.

---

```yaml
model:
  encoder:
    name: "roberta-base"                   # [config/config.yaml::model.encoder]
    tokenizer_name: "roberta-base"         # default (code) — main.py:141 uses AutoTokenizer.from_pretrained(config.model.encoder)
    max_length: 512                        # default (code) — src/utils/settings.py:161 (DEFAULT_MAX_LENGTH=512 in src/inference/constants.py)
    cache_dir: null                        # NOT IMPLEMENTED — HF default cache (~/.cache/huggingface) is used; no override key

    # memory / speed
    gradient_checkpointing: false          # default (code) — src/models/encoder/encoder_config.py:37
    flash_attention: true                  # default (code) — exposed as `enable_fused_attention=True` in src/models/config/model_config.py:20
    torch_compile: null                    # default (code) — tri-state auto: True on CUDA, False on CPU (src/models/encoder/encoder_config.py:52)
    compile_mode: "default"                # default (code) — src/models/encoder/encoder_config.py:53

  architecture:
    type: "multitask_transformer"          # implied by src/models/multitask/multitask_truthlens_model.py
    shared_encoder: true                   # default (code) — src/models/config/model_config.py:159
    dropout: 0.1                           # [config/config.yaml::model.dropout]

    # optimization
    reduce_intermediate_allocation: true   # default (code) — src/models/config/model_config.py:160
    layer_norm_eps: 1.0e-5                 # NOT IMPLEMENTED — falls through to HF RoBERTa default (1e-5); no override
    hidden_dropout_prob: 0.1               # NOT IMPLEMENTED — uses HF RoBERTa default; only top-level `dropout` is read

  # NOTE on heads:
  # config/config.yaml uses the LIGHTWEIGHT schema:
  #   tasks:
  #     bias: "multiclass"          → num_labels defaulted by src/utils/config_loader.py:176 (multiclass→3, multilabel→2)
  # No per-head loss_weight is set there; instead a top-level `task_weights` block
  # gives a uniform 1.0 to every task.
  # The richer schema below (per-head num_labels + loss_weight) is only honoured by
  # src/models/config/model_config.py::ModelConfigLoader, which is NOT currently
  # called by main.py or api/app.py — so today these values come from
  # `task_weights` in config.yaml (all 1.0) and from the runtime defaults in
  # src/utils/config_loader.py:176-180.

  heads:
    bias:
      num_labels: 3                        # default (code) — multiclass → 3 (src/utils/config_loader.py:177)
      loss_weight: 1.0                     # [config/config.yaml::task_weights.bias]
    ideology:
      num_labels: 3                        # default (code) — multiclass → 3
      loss_weight: 1.0                     # [config/config.yaml::task_weights.ideology]
    propaganda:
      num_labels: 3                        # default (code) — multiclass → 3
      loss_weight: 1.0                     # [config/config.yaml::task_weights.propaganda]
    narrative_frame:                       # SINGLE 5-class multilabel head, NOT three sub-heads
      # The template's CO / EC / HI breakdown does not match the project.
      # src/models/architectures/hybrid_truthlens_model.py:"narrative_frame": 5
      # frames are RE / HI / CO / MO / EC (Realistic, Human-Interest, Conflict,
      # Morality, Economic) — one head emitting 5 labels.
      num_labels: 5                        # [src/models/architectures/hybrid_truthlens_model.py]
      loss_weight: 1.0                     # [config/config.yaml::task_weights.narrative_frame]
      # The CO / EC / HI breakdown in the template is NOT IMPLEMENTED.
      CO: { num_labels: null }             # NOT IMPLEMENTED — folded into narrative_frame head
      EC: { num_labels: null }             # NOT IMPLEMENTED — folded into narrative_frame head
      HI: { num_labels: null }             # NOT IMPLEMENTED — folded into narrative_frame head

  # Two ADDITIONAL heads exist in config.yaml that the template omits:
  # narrative:
  #   task_type: multilabel  num_labels: 2 (default)  loss_weight: 1.0
  # emotion:
  #   task_type: multilabel  num_labels: 2 (default)  loss_weight: 1.0

  path: "saved_models"                     # default (code) — src/utils/settings.py:153 (resolved to <project_root>/saved_models)


# =========================================================
# TRAINING
# =========================================================

training:
  seed: 42                                 # [config/config.yaml::project.seed]
  device: "auto"                           # default (code) — src/inference/inference_config.py resolves auto→cuda/cpu

  epochs: 4                                # [config/config.yaml::training.epochs]
  batch_size: 16                           # [config/config.yaml::data.batch_size]
  gradient_accumulation_steps: 2           # [config/config.yaml::training.gradient_accumulation_steps]

  learning_rate: 3.0e-5                    # [config/config.yaml::optimizer.lr]
  weight_decay: 0.01                       # [config/config.yaml::optimizer.weight_decay]
  warmup_ratio: null                       # NOT IMPLEMENTED — replaced by absolute `scheduler.warmup_steps: 1000`
  scheduler: "linear"                      # [config/config.yaml::scheduler.name]
  optimizer: "adamw"                       # [config/config.yaml::optimizer.name]

  # stability
  label_smoothing: 0.0                     # default (code) — src/models/config/model_config.py:120 (RegularizationConfig.label_smoothing)
  max_grad_norm: 1.0                       # default (code) — src/models/config/model_config.py:89 (TrainingConfig.max_grad_norm)

  amp:
    enabled: true                          # [config/config.yaml::precision.use_amp]
    dtype: "bf16"                          # [config/config.yaml::precision.amp_dtype]

  gradient_clipping: 1.0                   # default (code) — same as max_grad_norm above

  # ------------------------------------------------------
  # DATA PIPELINE BEHAVIOR
  # ------------------------------------------------------

  data_pipeline:
    enable_cache: true                     # [main.py:202 — DataPipelineConfig(enable_cache=True)]
    force_rebuild: false                   # default (code) — src/data_processing/data_pipeline.py DataPipelineConfig default
    enable_cleaning: true                  # default (code) — DataPipelineConfig default
    enable_validation: true                # default (code) — DataPipelineConfig default
    enable_augmentation: true              # [config/data_config.yaml::augmentation.enabled]
    enable_profiling: false                # default (code) — DataPipelineConfig default (no profiling keys in YAML)
    enable_leakage_check: true             # default (code) — DataPipelineConfig default
    enable_multitask_validation: true      # default (code) — DataPipelineConfig default
    enable_label_analysis: true            # default (code) — DataPipelineConfig default

  # ------------------------------------------------------
  # TOKENIZATION / BATCHING
  # ------------------------------------------------------

  dynamic_padding: true                    # default (code) — DataCollatorWithPadding behavior in src/data_processing/collate.py
  group_by_length: false                   # NOT IMPLEMENTED — no length-bucketed sampler in src/data_processing/dataloader_factory.py

  dataloader:
    num_workers: 4                         # [config/config.yaml::data.num_workers]
    persistent_workers: false              # default (code) — DataLoaderConfig default in src/data_processing/dataloader_factory.py
    prefetch_factor: 2                     # default (code) — PyTorch default
    pin_memory: true                       # [config/config.yaml::data.pin_memory]

  # ------------------------------------------------------
  # EXECUTION OPTIMIZATION
  # ------------------------------------------------------

  torch_compile: null                      # default (code) — tri-state auto (see encoder above)
  compile_mode: "default"                  # default (code)
  fused_attention: true                    # default (code) — `enable_fused_attention=True`

  # ------------------------------------------------------
  # CHECKPOINTING
  # ------------------------------------------------------

  checkpoint:
    save_steps: 500                        # [config/config.yaml::training.checkpoint_every]
    save_total_limit: 3                    # [config/config.yaml::checkpoint.max_checkpoints]
    async_save: false                      # NOT IMPLEMENTED — sync save in src/models/checkpointing/checkpoint_manager.py
    save_optimizer_every: 500              # default (code) — same cadence as save_steps; not separately tunable

  resume_from_checkpoint: null             # NOT IMPLEMENTED — no `resume_from_checkpoint` key wired into trainer

  # ------------------------------------------------------
  # INPUT SCHEMA
  # ------------------------------------------------------

  text_column: "text"                      # default (code) — src/utils/settings.py:184
  title_column: null                       # NOT IMPLEMENTED — no title column is read by data_pipeline today
  label_columns:                           # [config/data_config.yaml::dataset.datasets.*.label_column]
    bias: "bias_label"
    ideology: "ideology_label"
    propaganda: "propaganda_label"
    narrative:
      hero_entities: "hero_entities"
      villain_entities: "villain_entities"
      victim_entities: "victim_entities"
    emotion: "emotion_*"                   # column-prefix scheme; see config/data_config.yaml::dataset.datasets.emotion.emotion_columns_prefix

  # ------------------------------------------------------
  # EARLY STOPPING
  # ------------------------------------------------------

  early_stopping:
    enabled: true                          # default (code) — src/models/config/model_config.py:95 (early_stopping=True)
    patience: 5                            # [config/config.yaml::training.early_stopping_patience]
    metric: "val_loss"                     # [config/config.yaml::checkpoint.monitor_metric]


# =========================================================
# DATA
# =========================================================
#
# config/config.yaml has NO top-level `data.train_path/validation_path/test_path`.
# Per-task split paths are built dynamically in src/config/settings_loader.py:114
# via a `data/<split>/<task>.csv` layout, and config/data_config.yaml gives a
# different per-task layout with `data/raw/<task>` directories.
# Both forms are resolved at runtime; nothing in YAML names a single split path.

data:
  train_path: "data/train/<task>.csv"      # template (code) — src/config/settings_loader.py:117 builds this per task
  validation_path: "data/val/<task>.csv"   # template (code)
  test_path: "data/test/<task>.csv"        # template (code)

  raw_dir: "data/raw"                      # [config/data_config.yaml::dataset.datasets.*.path prefix]
  interim_dir: null                        # NOT IMPLEMENTED — no interim dir is consumed
  processed_dir: null                      # NOT IMPLEMENTED — processed data lives under cache_dir, not a fixed processed_dir

  # versioning
  dataset_version: "2.0"                   # [config/data_config.yaml::project.version]
  shuffle_seed: 42                         # [config/data_config.yaml::reproducibility.random_seed] (also split.random_state)

  augmentation_multiplier: 1.5             # [config/data_config.yaml::augmentation.multiplier]


# =========================================================
# FEATURES
# =========================================================

features:
  engineered_text_column: null             # NOT IMPLEMENTED — no engineered text column is materialised; features run on raw text
  use_text_combination: false              # NOT IMPLEMENTED — no title+text merge is performed (no title column)

  tfidf:
    enabled: true                          # default (code) — src/features/utils/tfidf_engineering.py wired into pipeline
    max_features: null                     # default (code) — sklearn TfidfVectorizer default (unbounded); no override key
    top_terms_per_doc: null                # NOT IMPLEMENTED — top-k term selection per doc is not exposed

  embeddings_cache:
    enabled: false                         # NOT IMPLEMENTED — encoder embeddings are not cached to disk between batches
    path: null                             # NOT IMPLEMENTED


# =========================================================
# CROSS VALIDATION
# =========================================================

cross_validation:
  enabled: false                           # default (code) — src/training/cross_validation.py not invoked by main.py
  splits: 5                                # default (code) — KFold default in src/training/cross_validation.py
  metric: "f1"                             # default (code)
  stratified: true                         # [config/data_config.yaml::split.stratified]


# =========================================================
# HYPERPARAMETER TUNING
# =========================================================
#
# A tuner exists at src/training/hyperparameter_tuning.py but no YAML block
# currently configures it. The values below are the dataclass defaults from
# that module.

hyperparameter_tuning:
  enabled: false                           # NOT IMPLEMENTED in YAML — must be invoked programmatically
  trials: 20                               # default (code)
  direction: "minimize"                    # default (code) — minimises val_loss
  metric: "val_loss"                       # default (code)

  sampler: "tpe"                           # default (code) — Optuna TPESampler
  pruner: "median"                         # default (code) — Optuna MedianPruner

  search_space:
    learning_rate:
      min: 1.0e-5                          # default (code)
      max: 5.0e-5                          # default (code)
    batch_size: [8, 16, 32]                # default (code)
    epochs: [2, 3, 4]                      # default (code)
    dropout: [0.1, 0.2, 0.3]               # default (code)
    weight_decay: [0.0, 0.01, 0.1]         # default (code)


# =========================================================
# EVALUATION
# =========================================================

evaluation:
  metrics:
    classification: ["accuracy", "f1_macro", "precision", "recall"]   # default (code) — src/evaluation/metrics_engine.py
    multi_label:    ["f1_micro", "f1_macro", "hamming_loss", "subset_accuracy"]  # default (code)

  per_task_metrics:
    bias:            ["accuracy", "f1_macro"]                          # default (code) — applied per multiclass task
    ideology:        ["accuracy", "f1_macro"]                          # default (code)
    propaganda:      ["accuracy", "f1_macro"]                          # default (code)
    narrative_frame: ["f1_micro", "f1_macro", "hamming_loss"]          # default (code) — multilabel
    # Also evaluated (omitted from template):
    # narrative:     ["f1_micro", "f1_macro"]
    # emotion:       ["f1_micro", "f1_macro"]

  save_predictions: true                   # default (code) — src/evaluation/prediction_collector.py always persists predictions
  error_analysis: true                     # default (code) — src/evaluation/error_analysis.py runs in evaluation_pipeline


# =========================================================
# LOGGING
# =========================================================

logging:
  log_level: "INFO"                        # default (code) — src/utils/logging_utils.py configure_logging() default
  training_log_path: "artifacts/logs/training.log"   # [src/config/settings_loader.py:187 — logs_dir/training.log]
  eval_steps: 1                            # [config/config.yaml::training.eval_every]  (eval every epoch)

  use_wandb: true                          # [config/config.yaml::tracking.backend == "wandb"]
  project_name: "truthlens"                # [config/config.yaml::tracking.project_name]
  run_name: null                           # [config/config.yaml::tracking.run_name]


# =========================================================
# PATHS
# =========================================================
#
# The PathSettings dataclass in src/config/settings_loader.py builds these
# under <project_root>/artifacts/ by default. They can be overridden via the
# environment variables in brackets.

paths:
  models_dir: "artifacts/models"           # default (code) — src/config/settings_loader.py:170 [env: TRUTHLENS_MODELS_DIR]
  logs_dir: "artifacts/logs"               # default (code) — :171 [env: TRUTHLENS_LOGS_DIR]
  reports_dir: "reports"                   # default (code) — src/utils/settings.py:116

  cache_dir: "artifacts/cache"             # default (code) — src/config/settings_loader.py:173 [env: TRUTHLENS_CACHE_DIR]

  tfidf_vectorizer_path: "saved_models/tfidf_vectorizer.joblib"   # default (code) — src/utils/settings.py:110

  evaluation_results_path: "reports/evaluation_results.json"      # default (code) — src/utils/settings.py:111
  confusion_matrix_path:  "reports/confusion_matrix.png"          # default (code) — src/utils/settings.py:112
  cleaning_report_path:   "reports/cleaning_report.json"          # default (code) — src/utils/settings.py:113


# =========================================================
# API
# =========================================================

api:
  title: "TruthLens AI API"                # default (code) — src/utils/settings.py:96
  description: "Multi-task NLP system"     # default (code) — src/utils/settings.py:97
  version: "2.0.0"                         # default (code) — src/utils/settings.py:98

  text_preview_chars: 100                  # default (code) — src/utils/settings.py:99 (clamped to >=1 in api/app.py:85)

  max_batch_size: 50                       # [api/app.py:269 — BatchNewsRequest.texts max_length=50]
  timeout_seconds: null                    # NOT IMPLEMENTED — no request timeout middleware; uvicorn defaults apply


# =========================================================
# INFERENCE
# =========================================================

inference:
  batch_size: 32                           # default (code) — src/inference/constants.py::DEFAULT_INFERENCE_BATCH_SIZE
                                           #   (also src/utils/settings.py:103)
  device: "auto"                           # default (code) — src/utils/settings.py:104  (resolved cuda/cpu in inference_config.py:124)

  use_half_precision: false                # default (code) — src/utils/settings.py:105
  allow_raw_text_fallback: true            # default (code) — src/utils/settings.py:106 (read by api/app.py:86)

  # inference optimization
  use_onnx: false                          # NOT IMPLEMENTED in YAML — ONNX export exists (src/models/export/onnx_export.py)
                                           #   but inference path does not auto-load an ONNX runtime; export is on-demand
  use_tensorrt: false                      # NOT IMPLEMENTED — no TensorRT integration in repo

  return_probabilities: true               # default (code) — predict_api always returns fake_probability + confidence
  threshold: 0.5                           # default (code) — src/config/task_config.py:100 (per-task default; auto-thresholding
                                           #   is enabled for multilabel tasks via auto_threshold=True)
```

---

## Cross-reference summary

| Section | Source file(s) actually read at runtime |
|---|---|
| `model.encoder.*`, `model.architecture.*`, `model.heads.*` | `config/config.yaml` (light schema) → `src/utils/config_loader.py` |
| Richer `model.encoder.*`, regularisation, mixup, MC-dropout | `src/models/config/model_config.py` (no YAML wiring today) |
| `training.*` core loop | `config/config.yaml::{training, optimizer, scheduler, precision, data}` |
| `data.*` per-task paths | `src/config/settings_loader.py::_build_data_paths` (`data/<split>/<task>.csv`) |
| `data.*` raw layout, cleaning, augmentation | `config/data_config.yaml` |
| `paths.*` | `src/utils/settings.py` defaults + `src/config/settings_loader.py` |
| `api.*`, `inference.*` | `src/utils/settings.py` defaults |
| `aggregation.*`, `graph.*`, `features.*`, `explainability.*` | `config/config.yaml` (additional blocks not in the template) |

## Honest gaps to be aware of

1. **`load_config` duplication is still alive.** The repo has two functions named
   `load_config` returning different types: a strict dataclass loader at
   `src/config/config_loader.py:169` (used by `main.py`) and a permissive dict
   loader at `src/utils/config_loader.py:127` (used by `task_config.py` and
   others). They read the same YAML but enforce different invariants. Any future
   field added to `config.yaml` must satisfy **both** schemas or one path will
   silently ignore it.

2. **The richer multitask schema in `src/models/config/model_config.py` is not
   wired to any YAML.** Per-head `loss_weight`, `RegularizationConfig.label_smoothing`,
   mixup, adversarial training, MC-dropout, etc., all have dataclass defaults but
   no entry point reads them from YAML today. If you want to drive them from
   `config.yaml`, you would need to call `ModelConfigLoader.load_multitask_config(...)`
   on a YAML that mirrors that schema (separate from the current `config.yaml`).

3. **`narrative_frame` is a single 5-class multilabel head**, not a CO/EC/HI
   triplet of sub-heads as the template suggests. The 5 frames are
   RE / HI / CO / MO / EC (see `src/models/architectures/hybrid_truthlens_model.py`).
   If you intended three independent heads, that is a model-architecture change,
   not a config change.

4. **`num_labels` is defaulted, not declared.** The current `config.yaml::tasks`
   block only declares the task *type* (`"multiclass"` / `"multilabel"`).
   `src/utils/config_loader.py:176-180` then maps that to a default num_labels
   (multiclass→3, multilabel→2). For `narrative_frame` this default of `2` is
   **wrong** — the model expects 5. If you switch to using the richer
   `ModelConfigLoader`, declare `num_labels: 5` explicitly there.

5. **`reports_dir`, `tfidf_vectorizer_path`, `cleaning_report_path`, etc., are
   only defaults in `src/utils/settings.py`.** They are not present in any YAML
   today; if you ever want to relocate `reports/` you must either add a `paths:`
   block to `config/config.yaml` or set the corresponding `TRUTHLENS_*` env
   variables.

6. **`use_onnx`, `use_tensorrt`, `embeddings_cache`, `group_by_length`,
   `async_save`, `resume_from_checkpoint`, `title_column`, `top_terms_per_doc`,
   `interim_dir`, `processed_dir`** are all in the template but have **no
   consumer** in the codebase. Adding them to a YAML file today will not change
   behaviour.
