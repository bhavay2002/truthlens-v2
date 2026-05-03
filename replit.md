# TruthLens AI

## Overview
TruthLens AI is a multi-layer AI platform designed for misinformation detection and news credibility analysis. It integrates deep linguistic analysis, narrative extraction, propaganda detection, and graph-based reasoning to generate an interpretable "Credibility Score." The platform aims to enhance media literacy and combat the spread of false information by providing users with tools to assess the trustworthiness of news content. Key capabilities include multi-task deep learning models, explainability features, and robust data processing pipelines.

## User Preferences
I want iterative development and detailed explanations. I want to be asked before you make major changes. Do not make changes to files outside of the `src/` directory unless explicitly instructed.

## System Architecture
The system is built on a FastAPI REST API using Python 3.12, served by Uvicorn. Machine learning and natural language processing tasks leverage PyTorch, Hugging Face Transformers, spaCy, NLTK, LIME, and SHAP.

**Core Features and Design Patterns:**

*   **Multi-Task Learning:** The system uses a `MultiTaskTruthLensModel` with a shared encoder (RoBERTa-base) and per-task heads for different misinformation detection aspects (bias, ideology, propaganda, narrative, emotion, framing). This design optimizes resource usage and promotes knowledge transfer between related tasks.
*   **Explainability:** Integrated explainability techniques (SHAP, Integrated Gradients, LIME, attention rollout) provide insights into model predictions, generating token-level importance scores and aggregated explanations. An `ExplainabilityOrchestrator` manages the lifecycle of explanations, ensuring faithfulness and performance.
*   **Graph-based Analysis:** Entity and narrative graphs are constructed and analyzed to detect propagation patterns and conflicts. A `GraphPipeline` handles graph building, feature extraction, and graph-based explanations, utilizing Node2Vec, spectral, or hybrid embedding types.
*   **Robust Data Processing:** A comprehensive `DataPipeline` manages data loading, cleaning, augmentation, and caching. It supports multiple tasks, handles class imbalance with weighted sampling (including the new `TaskPresenceMaskSampler` for cross-task balanced batches), and includes rigorous leakage checks.
*   **Feature Engineering:** A modular `FeaturePipeline` extracts a wide array of linguistic, semantic, and graph-based features using `FeatureScalingPipeline` and `FeatureReductionPipeline` for preprocessing. Lexicon-based features are vectorized for performance.
*   **Training & Checkpointing:** The `Trainer` class manages the training loop, supporting multi-task training, mixed precision (AMP), gradient accumulation, and learning rate scheduling. Checkpoints are saved atomically and include the full training state (model, optimizer, scheduler, scaler). Instrumentation for anomaly detection (loss spikes, gradient anomalies) is integrated to enhance training stability.
*   **Interacting Multi-Task Architecture (Phase 2):** A new `InteractingMultiTaskModel` in `src/models/multitask/interacting_model.py` transforms the encoder → independent heads pipeline into a joint reasoning network. (1) `MultiViewPooling` replaces CLS-only pooling with a three-way representation (CLS + masked mean + learned attention pooling → Linear → H_shared). (2) Per-task `TaskProjectionBlock` + `TaskEmbeddings` give each task its own projection subspace and an explicit learnable task identity embedding. (3) `CrossTaskInteractionLayer` implements gated cross-attention — each task's representation attends to all other tasks' representations, gated by a per-task sigmoid, with residual + LayerNorm for stability; this encodes the spec's bias↔emotion↔ideology and propaganda↔narrative interaction patterns as learned relationships. (4) `LatentFusionHead` concatenates all refined task representations into a unified latent vector Z (B, D) and produces a scalar credibility score via sigmoid. (5) `blend_credibility()` supports hybrid aggregation: `α * neural_score + (1−α) * rule_score`. The model is fully backward-compatible — same `task_logits` output format, same `self.encoder`/`self.task_heads`/`self.heads` API, same explainability contracts.
*   **Semantic Alignment (Phase 1):** The dataset pipeline now supports semantically aligned multi-task learning: (1) `task_mask` (per-row binary mask over tasks) and `derived_features` (cross-task signals: emotional_bias_score, propaganda_intensity, ideological_emotion) are computed in every dataset and propagated through collate into every batch; (2) `MultiTaskLoss.forward` accepts a `task_mask` tensor and uses it to gate per-task loss contributions for sparse/partial supervision; (3) `TaskLossConfig` and `LossEngineConfig` expose `temperature` (per-task logit scaling, e.g. T=1.5 for emotion, T=0.8 for propaganda) and `label_smoothing` (ε per task) which are applied inside `TaskLossRouter` before loss computation; (4) `_build_per_row_task_mask` correctly infers which tasks each row actually has labels for, enabling `MultiTaskAlignedDataset` to build accurate per-row masks for mixed-corpus training; (5) `TaskPresenceMaskSampler` weights rows by their cross-task label richness, up-sampling multi-task rows to improve shared-encoder gradient signal.
*   **API Endpoints:** The API exposes endpoints for core prediction (`/predict`, `/batch-predict`), detailed analysis (`/analyze`), full explainability + credibility aggregation (`/explain`), structured reports (`/report`), calibration management (`/calibration`), ensemble predictions (`/ensemble`), model export (`/export`), and various utility functions (`/health`, `/model-info`).
*   **Heuristic Fallback:** When the ML model has not been trained yet, `/analyze` and `/explain` automatically fall back to a lexicon-based heuristic `predict_fn` (combining bias score and emotion intensity) so all linguistic analysis and explainability pipelines still run and return results.
*   **Singleton Management:** Key components like the `InferenceCache`, `InferenceEngine`, `GraphPipeline`, and `AnalysisRegistry` are implemented as singletons to optimize resource utilization and avoid redundant initialization.
*   **Configuration:** Project-wide settings are managed through `config/config.yaml`, supporting flexible configuration of model paths, training parameters, and API behavior.
*   **Error Handling and Resilience:** Extensive error handling, including atomic file operations, input validation, NaN/Inf guards, and structured logging, ensures system stability and provides clear diagnostics.

## External Dependencies
*   **Deep Learning Frameworks:** PyTorch, Hugging Face Transformers
*   **NLP Libraries:** spaCy, NLTK
*   **Explainability Libraries:** LIME, SHAP
*   **Web Framework:** FastAPI
*   **ASGI Server:** Uvicorn
*   **Data Manipulation:** Pandas, NumPy
*   **Graph Processing:** NetworkX, SciPy (for sparse matrix operations)
*   **Machine Learning Utilities:** scikit-learn
*   **Serialization:** `portalocker` (for file locking)
*   **Tokenizers:** `sentencepiece` or `protobuf` (for Hugging Face tokenizers)