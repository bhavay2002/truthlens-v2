# TruthLens AI Repository Structure

Last updated: 2026-04-02

This document summarizes the active project layout and the runtime-relevant artifacts.

## Top-Level Layout

```text
Truthlens Ai/
  api/                   FastAPI app
  config/                YAML configs
  data/                  raw/interim/processed/splits + unified dataset outputs
  experiments/           experiment files
  logs/                  runtime logs
  models/                saved model artifacts
  notebooks/             notebooks
  reports/               evaluation + EDA artifacts
  src/                   core code packages
  tests/                 automated tests

  README.md
  architecture.md
  KNOWLEDGE.md
  PROJECT_REVIEW.md
  structure.md
```

## Source Package Map

```text
src/
  aggregation/           final score aggregation
  analysis/              argument/narrative/context/emotion-target analysis
  data/                  load/merge/clean/validate/split/schema helpers
  evaluation/            metrics and evaluation visualizations
  explainability/        SHAP/LIME and task explainers
  features/              feature extraction modules
    bias/
    discourse/
    emotion/
    narrative/
  graph/                 entity and narrative graph analysis
  inference/             high-level analysis inference orchestrators
  models/                training, prediction, multitask and registries
    emotion/
    encoder/
    ideology/
    multitask/
    narrative/
    propaganda/
  pipelines/             data/feature/emotion/truthlens orchestration pipelines
  training/              CV and hyperparameter tuning utilities
  utils/                 settings, config, validation, logging, helpers
  visualization/         generic visualization helpers
```

## Unified Dataset Outputs (Current)

Generated unified split files:

- `data/unified_dataset_train.csv`
- `data/unified_dataset_validation.csv`
- `data/unified_dataset_test.csv`

Canonical unified columns:

- `title`, `text`
- `bias_label`, `ideology_label`, `propaganda_label`, `frame`
- `CO`, `EC`, `HI`, `MO`, `RE`
- `hero`, `villain`, `victim`
- `hero_entities`, `villain_entities`, `victim_entities`
- `emotion_0` ... `emotion_19`
- `dataset`

## Runtime-Critical Files

- `main.py`: training orchestration entry point.
- `api/app.py`: serving interface.
- `src/models/train_roberta.py`: model training.
- `src/models/multitask/multitask_truthlens_model.py`: multi-head task model.
- `src/data/unified_label_schema.py`: schema normalization/validation.
- `src/pipelines/data_pipeline.py`: data flow orchestration.
- `src/evaluation/evaluate_model.py`: metrics computation.

## Common Artifacts

- Model directory: `models/roberta_model/`
- Vectorizer: `models/tfidf_vectorizer.joblib`
- Evaluation report: `reports/evaluation_results.json`
- Confusion matrix: `reports/confusion_matrix.png`
- Cleaning report: `reports/data_cleaning_report.json`

## Notes

- The repository includes exploratory/transitional root scripts (for example `ztest*.py`).
- Generated artifacts under `reports/`, `logs/`, and parts of `models/` are expected to change between runs.
