# Test Suite Guide

This document describes each file under `tests/`, what it validates, and why it exists.

**Total: 344 tests across 38 modules — all passing.**

Run the full suite:

```bash
pytest
```

---

## Shared Setup

| File | What it does | What it tests |
|---|---|---|
| `conftest.py` | Shared pytest fixtures and project-path bootstrap. | Ensures `src` and `api` imports work consistently and provides reusable sample DataFrames. |

---

## End-to-End Tests

| File | What it does | What it tests |
|---|---|---|
| `test_e2e_dataset.py` | **108 tests across 13 classes.** Validates the 10-row sample dataset and exercises every API endpoint against the live server via HTTP. | Dataset integrity (38 columns, labels, binary fields, entity JSON), unified label schema, all 16 API endpoints (GET and POST), prediction probability ranges, cache hits, calibration metrics (ECE/MCE/Brier/NLL), three ensemble strategies, ONNX/TorchScript export, error handling (422/404), and prediction self-consistency (probability/label agreement, confidence formula, single vs batch agreement). |

---

## API Tests

| File | What it does | What it tests |
|---|---|---|
| `test_api.py` | Tests FastAPI endpoints with a stubbed prediction function. | Health endpoint availability, `/predict` response schema, and request validation behavior. |
| `test_api_error_paths.py` | Tests all error paths in the API. | 422 validation errors for short/long text, missing fields, over-limit batch sizes, and malformed inputs. |

---

## Configuration Tests

| File | What it does | What it tests |
|---|---|---|
| `test_config_integrity.py` | Loads typed app settings. | Required training/model settings are present and sane. |
| `test_config_loading.py` | Loads YAML configuration. | Core config sections (`model`, `training`, `data`) exist. |

---

## Data Tests

| File | What it does | What it tests |
|---|---|---|
| `test_data_augmentation.py` | Validates augmentation interface behavior. | `multiplier=1` no-op and invalid multipliers raise `ValueError`. |
| `test_data_leakage.py` | Checks train/test text overlap after splitting. | No duplicate texts leak across split boundaries. |
| `test_data_pipeline_module.py` | Tests the root-level data pipeline module. | Entry point and config validator are importable and functional. |
| `test_data_processing.py` | Tests cleaning utilities. | URL/lowercasing cleanup and DataFrame text-column sanitation flow. |
| `test_data_validation.py` | Tests schema validator. | Required columns are accepted by `DataValidator`. |
| `test_dataset_schema.py` | Unified schema contract check. | Dataset contains the canonical 7-task unified columns (`title`, `text`, task labels, narrative frame flags, `emotion_0`–`emotion_19`, `dataset`). |
| `test_dataset_split_integrity.py` | Split integrity test on synthetic data. | Train/test split has no overlapping text rows. |

---

## Evaluation & Explainability Tests

| File | What it does | What it tests |
|---|---|---|
| `test_evaluation.py` | Runs evaluation on small label arrays. | Accuracy metric is returned and bounded in `[0, 1]`. |
| `test_evaluation_metrics.py` | Tests calibration, uncertainty, and per-task metrics. | ECE, Brier score, uncertainty estimator, and multi-task evaluation dashboards. |
| `test_explainability.py` | Tests explainability pipeline with monkeypatched internals. | `explain_bias` returns the full expected result schema including `bias_heatmap`, `token_importance`, `attention_scores`, `integrated_gradients`, `bias_intensity`, `biased_tokens`, `sentence_bias_scores`. |
| `test_shap_explainer.py` | Tests the SHAP explainer module. | SHAP explainer is importable and produces correctly structured output. |

---

## Feature Engineering Tests

| File | What it does | What it tests |
|---|---|---|
| `test_feature_pipeline.py` | Runs feature engineering on fixture data. | Engineered text column creation, row-count preservation, and vectorizer output. |

---

## Inference & Input Tests

| File | What it does | What it tests |
|---|---|---|
| `test_inference_speed.py` | Lightweight performance sanity check. | Basic inference-like loop completes under a simple threshold. |
| `test_input_validation.py` | Tests integer validator utility. | Valid positive ints pass, invalid values raise `ValueError`. |
| `test_prediction_pipeline.py` | Tests prediction function with mocked tokenizer/model. | Prediction output schema and empty-input validation. |
| `test_prediction_stability.py` | Determinism check with dummy model. | Same input returns identical prediction twice. |

---

## Logging & Project Structure Tests

| File | What it does | What it tests |
|---|---|---|
| `test_logging.py` | Logger initialization smoke test. | Logger instance is available and callable. |
| `test_project_structure.py` | Filesystem contract tests. | Required project directories and core files exist. |

---

## Model Tests

| File | What it does | What it tests |
|---|---|---|
| `test_model_registry.py` | Tests `ModelRegistry` loading and caching. | Model and tokenizer are loaded and cached; fallback paths work correctly. |
| `test_model_subpackage_imports.py` | Import smoke tests for all model subpackages. | All model subpackage `__init__.py` files import cleanly. |
| `test_model_utils.py` | Serialization utility tests. | Saving and loading model payloads through filesystem. |
| `test_multitask_label_helpers.py` | Tests `MultiTaskLabelHelper` class attributes. | `BIAS_LABELS`, `IDEOLOGY_LABELS`, `PROPAGANDA_LABELS`, `EMOTION_LABELS`, `NARRATIVE_ROLES`, `FRAME_LABELS` are defined with correct types and lengths. |

---

## Training Tests

| File | What it does | What it tests |
|---|---|---|
| `test_model_training.py` | Tests training split/validation helpers (no heavy model training). | Disjoint train/val/test partitions and split DataFrame validation errors. |
| `test_training_pipeline.py` | Tests CV and tuning wrappers with dummy trainer. | Cross-validation summary contract and hyperparameter tuning result contract. |
| `test_tokenization.py` | Tests tokenization helper with stub tokenizer. | `tokenize_function` output keys and shape consistency. |

---

## Utility Tests

| File | What it does | What it tests |
|---|---|---|
| `test_reproducibility.py` | Randomness control test. | Same random seed reproduces identical NumPy outputs. |
| `test_utils.py` | Utility integration tests. | Typed settings, folder creation behavior, and model utils with `Path` objects. |

---

## How To Run

Full suite:

```bash
pytest
```

Specific module:

```bash
pytest tests/test_e2e_dataset.py -v
```

With output on failure:

```bash
pytest -s --tb=short
```

End-to-end tests require the server to be running on `http://localhost:5000`. Start it with:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 5000 --reload \
  --reload-dir api --reload-dir src --reload-dir config --reload-dir models
```
