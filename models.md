# `src/models/` — TruthLens Model System
### Production-Grade Technical Documentation

---

## Table of Contents

1. [Overview](#1-overview)
2. [Folder Architecture](#2-folder-architecture)
3. [Model Architecture](#3-model-architecture)
4. [End-to-End Model Workflow](#4-end-to-end-model-workflow)
5. [File-by-File Deep Dive](#5-file-by-file-deep-dive)
6. [Model Inputs & Outputs](#6-model-inputs--outputs)
7. [Loss Functions & Optimization](#7-loss-functions--optimization)
8. [Training Integration](#8-training-integration)
9. [Inference Logic](#9-inference-logic)
10. [Model Checkpoints & Persistence](#10-model-checkpoints--persistence)
11. [Config Integration](#11-config-integration)
12. [Optimization & Efficiency](#12-optimization--efficiency)
13. [Extensibility Guide](#13-extensibility-guide)
14. [Common Pitfalls & Risks](#14-common-pitfalls--risks)
15. [Example Usage](#15-example-usage)
16. [Simple Explanation (For Non-Technical Reviewers)](#16-simple-explanation-for-non-technical-reviewers)

---

## 1. Overview

`src/models/` is the central neural-network layer of TruthLens AI. It defines, trains, saves, loads, and serves every machine-learning model that the system uses to detect misinformation and assess content credibility.

The module implements a **shared-encoder, multi-headed transformer architecture**: one large pre-trained language model (defaulting to `roberta-base`) encodes every article or claim into a dense vector, then six independent task heads simultaneously classify that vector into distinct credibility signals — media bias, political ideology, propaganda presence, narrative framing, narrative roles, and emotional tone. Every component of this pipeline — encoder, heads, loss functions, calibration, checkpointing, export, and uncertainty — lives in a self-contained subfolder and communicates through explicit typed contracts.

**What this module is responsible for:**

| Concern | What lives here |
|---|---|
| Core model classes | `base/`, `multitask/`, `architectures/` |
| Encoder abstraction | `encoder/` |
| Task classification heads | `heads/` |
| Loss orchestration | `loss/` |
| Post-hoc calibration | `calibration/` |
| Checkpoint management | `checkpointing/` |
| Model registry & factory | `registry/` |
| Inference predictor | `inference/` |
| Parameter-efficient adapters | `adapters/` |
| Ensemble strategies | `ensemble/` |
| Export (ONNX, TorchScript, quantization) | `export/` |
| Uncertainty estimation | `uncertainty/` |
| Optimizer & LR scheduling | `optimization/` |
| Monitoring, regularization, benchmarking | `monitoring/`, `regularization/`, `benchmarking/` |

**Key design invariants enforced throughout:**

- **G4 — Calibration parameter isolation:** Temperature scalars are never seen by the main training optimizer. `BaseModel.get_calibration_parameters()` and `get_optimization_parameters()` split these cleanly.
- **N1 — Log-space entropy:** Every entropy calculation goes through `log_softmax` or `logsigmoid` instead of `log(p + eps)` to avoid numerical bias at the tails.
- **P1 — Training-mode guard:** Probabilities, confidence, and entropy are only materialized at inference time (`not self.training`), eliminating dead compute in the autograd graph during training.
- **C1.3 — Secure checkpoint loading:** All `torch.load` calls use `weights_only=True`.
- **A3.4 — Dict-only head contract:** Every task head must return a `dict` with at least a `"logits"` key; tensor-returning heads raise immediately.

---

## 2. Folder Architecture

```
src/models/
├── _device.py                  # Device detection: CUDA → MPS → CPU
├── adapters/                   # Parameter-efficient fine-tuning
│   ├── adapter_config.py       # Adapter hyperparameter dataclass
│   ├── adapter_layer.py        # Bottleneck adapter layer
│   └── lora_adapter.py         # LoRA (Low-Rank Adaptation) linear layer
├── architectures/
│   ├── __init__.py
│   └── hybrid_truthlens_model.py  # Hybrid (encoder + feature) model variant
├── base/
│   ├── base_model.py           # Abstract BaseModel (nn.Module + ABC)
│   ├── base_classifier.py      # Single-task classifier base
│   └── multitask_base_model.py # Shared-encoder multi-task abstraction
├── benchmarking/               # Speed and throughput benchmarking tools
├── calibration/
│   ├── __init__.py
│   ├── calibration_metrics.py  # ECE, MCE, reliability diagram metrics
│   ├── isotonic_calibration.py # IsotonicCalibrator (sklearn-wrapped)
│   └── temperature_scaling.py  # TemperatureScaler (learnable scalar T)
├── checkpointing/
│   ├── artifact_manager.py     # Multi-artifact checkpoint bundling
│   ├── checkpoint_manager.py   # CheckpointManager (V2, atomic save)
│   ├── integrity.py            # SHA-256 integrity hashing
│   ├── io_utils.py             # atomic_save / safe_load / fsync_dir
│   ├── loader_utils.py         # Optimizer device migration
│   ├── metadata.py             # metadata.json save/load
│   ├── model_loader.py         # High-level model loading helpers
│   ├── resolver.py             # Checkpoint path resolution
│   ├── schema.py               # Schema versioning & migration
│   ├── selection.py            # Best-checkpoint tracking
│   └── validator.py            # State-dict key validation
├── config/
│   └── model_config.py         # MultiTaskModelConfig (YAML-backed dataclass)
├── distillation/               # Knowledge distillation (teacher→student)
├── emotion/                    # Emotion-specific model utilities
├── encoder/
│   ├── encoder_config.py       # EncoderConfig dataclass
│   ├── encoder_factory.py      # EncoderFactory.create_transformer_encoder()
│   ├── __init__.py
│   └── transformer_encoder.py  # TransformerEncoder (wraps HF AutoModel)
├── ensemble/
│   ├── ensemble_model.py       # EnsembleModel (average/weighted/vote)
│   ├── stacking_ensemble.py    # Learned stacking meta-model
│   ├── weighted_ensemble.py    # Confidence-weighted ensemble
│   └── _utils.py               # logit extraction utility
├── export/
│   ├── __init__.py
│   ├── onnx_export.py          # ONNXExporter with verification
│   ├── quantization.py         # Dynamic INT8 / static quantization
│   └── torchscript_export.py   # TorchScript tracing/scripting
├── heads/
│   ├── classification_head.py  # ClassificationHead (multi-class)
│   └── multilabel_head.py      # MultiLabelHead (binary-per-label)
├── ideology/                   # Ideology-specific model utilities
├── inference/
│   ├── predictor.py            # Predictor (batch forward + calibration)
│   └── prediction_output.py    # PredictionOutput structured container
├── interpretability/           # Attribution / explainability hooks
├── loss/
│   ├── base_balancer.py        # BaseBalancer ABC (GradNorm / Uncertainty)
│   ├── coverage_tracker.py     # EMACoverageTracker (sparse supervision)
│   ├── loss_normalizer.py      # EMALossNormalizer (scale normalization)
│   ├── multitask_loss.py       # MultiTaskLoss (main orchestrator)
│   ├── task_loss_router.py     # TaskLossRouter (per-task loss dispatch)
│   └── uncertainty.py          # UncertaintyBalancer (Kendall & Gal)
├── metadata/
│   ├── model_metadata.py       # ModelMetadata (training provenance)
│   └── model_versioning.py     # ModelVersionRegistry (semver)
├── monitoring/                 # Training metric monitoring hooks
├── multitask/
│   ├── multitask_output.py     # MultiTaskOutput container
│   └── multitask_truthlens_model.py  # MultiTaskTruthLensModel (flagship)
├── narrative/                  # Narrative-specific model utilities
├── optimization/
│   ├── lr_scheduler.py         # LR scheduler factory (cosine, linear, etc.)
│   └── optimizer_factory.py    # OptimizerFactory + param group builder
├── propaganda/                 # Propaganda-specific model utilities
├── registry/
│   ├── __init__.py
│   ├── model_factory.py        # ModelFactory.create(type, config)
│   ├── model_registry.py       # ModelRegistry.load_model() (entry point)
│   └── __pycache__/
├── regularization/             # Dropout variants, R-Drop, consistency loss
├── representation/             # Embedding space analysis tools
├── tasks/                      # Task-level evaluation harness
├── uncertainty/
│   ├── ensemble_uncertainty.py # Disagreement across ensemble members
│   ├── mc_dropout.py           # MCDropoutPredictor (stochastic sampling)
│   └── uncertainty_head.py     # Learned uncertainty prediction head
└── utils/                      # Shared tensor / device utilities
```

**30 subfolders, ~90 files.** The core inference-time stack touches only: `_device.py`, `encoder/`, `base/`, `heads/`, `multitask/`, `calibration/`, `inference/`, `registry/`.

---

## 3. Model Architecture

### 3.1 Class Hierarchy

```
nn.Module (PyTorch)
└── BaseModel  [base/base_model.py]
    ├── TransformerEncoder  [encoder/transformer_encoder.py]
    └── MultiTaskTruthLensModel  [multitask/multitask_truthlens_model.py]
        ├── .encoder  → TransformerEncoder
        └── .task_heads (nn.ModuleDict)
            ├── "bias"           → ClassificationHead (2 classes)
            ├── "ideology"       → ClassificationHead (3 classes)
            ├── "propaganda"     → ClassificationHead (2 classes)
            ├── "narrative"      → MultiLabelHead (3 labels)
            ├── "narrative_frame"→ MultiLabelHead (5 labels)
            └── "emotion"        → MultiLabelHead (11 labels)

nn.Module
└── BaseModel
    └── BaseClassifier  [base/base_classifier.py]
        └── Single-task classification (bias, ideology, etc.)

nn.Module
└── MultiTaskBaseModel  [base/multitask_base_model.py]
    └── Legacy multi-task abstraction (superseded by MultiTaskTruthLensModel)
```

### 3.2 Encoder: `TransformerEncoder`

- Wraps HuggingFace `AutoModel` (default: `roberta-base`, 125M parameters, hidden dim 768).
- Three **pooling modes**: `cls` (default — `hidden[:, 0]`), `mean` (masked average), `attention` (softmax-weighted sum).
- Returns `_EncoderOutput`, a `dict` subclass that also supports attribute access (HF-style `out.last_hidden_state` AND dict-style `out["pooled_output"]`) to serve both the training pipeline and the explainability layer.
- Supports `inputs_embeds` as an alternate entry point for Integrated-Gradients attribution.
- `gradient_checkpointing`, `freeze`/`unfreeze`, AMP (`autocast` on CUDA only).
- `torch.compile` was evaluated and **deliberately disabled** (`COMPILE-OFF`) due to spurious bf16 overflow warnings and environment-specific instability.

### 3.3 Task Heads

**`ClassificationHead`** (multi-class):
- Optional hidden layer (`input_dim → hidden_dim → num_classes`) with GELU/ReLU/Tanh activation.
- Optional `LayerNorm` at the input and/or hidden layer.
- `_init_weights`: Xavier-uniform on all `Linear` layers.
- Training output: `{"logits": Tensor}`.
- Inference output: adds `probabilities`, `confidence`, `entropy` (all via a single `log_softmax` pass — N1).

**`MultiLabelHead`** (independent binary per label):
- Same optional hidden-layer topology.
- Uses `BCEWithLogitsLoss` (fused sigmoid for numerical stability).
- Inference adds per-label `probabilities` (sigmoid), `predictions` (threshold ≥ 0.5), `confidence` (mean probability), `entropy` (via `logsigmoid`/`logsigmoid(-x)` — N1).
- Supports `return_features` to expose intermediate hidden activations for attribution.

### 3.4 Default Task Label Vocabulary

| Task | Head type | Labels | Count |
|---|---|---|---|
| `bias` | ClassificationHead | `["non_bias", "bias"]` | 2 |
| `ideology` | ClassificationHead | `["left", "center", "right"]` | 3 |
| `propaganda` | ClassificationHead | `["non_propaganda", "propaganda"]` | 2 |
| `narrative` | MultiLabelHead | `["hero", "villain", "victim"]` | 3 |
| `narrative_frame` | MultiLabelHead | `["RE", "HI", "CO", "MO", "EC"]` | 5 |
| `emotion` | MultiLabelHead | `emotion_0` … `emotion_10` | 11 |

---

## 4. End-to-End Model Workflow

```
Raw text (str)
     │
     ▼
[Tokenizer] (HuggingFace AutoTokenizer)
     │  input_ids: (B, L)
     │  attention_mask: (B, L)
     ▼
[TransformerEncoder.forward(input_ids, attention_mask)]
     │  → HF AutoModel (RoBERTa etc.)
     │  → last_hidden_state: (B, L, H)    H=768 for roberta-base
     │  → _pool(hidden, mask)
     │  pooled_output: (B, H)
     ▼
[MultiTaskTruthLensModel._extract_pooled(encoder_outputs)]
     │  strict: requires "pooled_output" or "pooler_output"
     │  fails loudly if encoder didn't produce one
     ▼
[For each task head in nn.ModuleDict]:
     ├─ ClassificationHead.forward(pooled)  → {logits, (probs, conf, entropy)}
     └─ MultiLabelHead.forward(pooled)      → {logits, (probs, preds, conf, entropy)}
     │
     ▼
[outputs dict]:
     {
       "bias":          {"logits": (B,2), "probabilities": (B,2), ...},
       "ideology":      {"logits": (B,3), ...},
       "propaganda":    {"logits": (B,2), ...},
       "narrative":     {"logits": (B,3), ...},
       "narrative_frame": {"logits": (B,5), ...},
       "emotion":       {"logits": (B,11), ...},
       "task_logits":   {"bias": (B,2), "ideology": (B,3), ...}  ← thin view
     }
     │
     ▼ (at inference, via Predictor._format_outputs)
[_flatten_task_logits per task]:
     bias_logits, bias_probabilities, bias_predictions, bias_confidence, bias_entropy
     ideology_logits, ...
     ...
     │
     ▼ (optional)
[Calibration]:
     TemperatureScaler.predict_proba(logits)  → adjusted probabilities
  OR IsotonicCalibrator.predict_proba(probs)  → adjusted probabilities
     │
     ▼
[PredictionOutput.from_flat(formatted)] or flat dict
     │
     ▼
[src/aggregation/ or src/inference/] → credibility score
```

**Training-time differences:**

- Heads return only `{"logits"}` (P1 guard skips probability computation).
- `MultiTaskLoss.forward(logits_dict, labels_dict)` receives `outputs["task_logits"]` and the collated labels; returns `(total_loss, raw_per_task_losses)`.
- `CheckpointManager.save(step=..., model=..., optimizer=..., metrics=...)` is called by the trainer at configurable intervals.

---

## 5. File-by-File Deep Dive

### 5.1 `_device.py`

```python
def detect_device(device: str | None = None) -> torch.device:
    # Priority: explicit > CUDA > MPS (Apple Silicon) > CPU
```

- Centralised device selection used by all model constructors.
- Prevents environment drift: EncoderFactory, Predictor, and training benchmarks all call the same function.

---

### 5.2 `base/base_model.py` — `BaseModel`

**Role:** Abstract base for every TruthLens neural network.

**Key methods:**

| Method | Purpose |
|---|---|
| `set_device(device)` | Moves entire module; caches `self._device` |
| `device` (property) | Returns `self._device` fast path, falls back to parameter walk (A3.3) |
| `attach_module(name, module)` | Registers a new sub-module AND immediately moves it to current device (A5.3) |
| `get_calibration_parameters()` | Returns all params named `"temperature"` (G4) |
| `get_optimization_parameters()` | Complement of above |
| `save_checkpoint(path, ...)` | Saves `state_dict` + optimizer state to a `.pt` file |
| `load_checkpoint(path, ...)` | Loads with `weights_only=True`; strict by default (C1.3) |
| `freeze()` / `unfreeze()` | All parameters |
| `freeze_encoder()` / `unfreeze_encoder()` | Encoder trunk only |
| `freeze_head(task)` | Specific task head |
| `num_parameters(trainable_only)` | Parameter count |
| `parameter_breakdown()` | Dict of `{"encoder": N, "head_bias": M, ...}` |
| `summary()` | JSON-serializable model description |

**Design note:** `CALIBRATION_PARAMETER_NAMES = ("temperature",)` is a class-level tuple that subclasses can extend. The matching is done on the trailing leaf of the dotted parameter name, so deeply nested `task_heads.bias.temperature` is still caught.

---

### 5.3 `base/base_classifier.py` — `BaseClassifier`

**Role:** Adds a single classification head on top of `BaseModel`.

**Key behaviors:**
- Accepts `labels` in `forward()` to compute loss in-place.
- Supports `CrossEntropyLoss` with label smoothing (multi-class) or `BCEWithLogitsLoss` (multilabel/binary).
- `compute_loss` is soft-label-aware: if `labels.dtype` is float (i.e., already a probability distribution), it switches to `KLDivLoss` instead of `CrossEntropyLoss` (A4.3).
- Stable entropy: uses `F.log_softmax` (N1).

---

### 5.4 `base/multitask_base_model.py` — `MultiTaskBaseModel`

**Role:** Legacy multi-task abstraction. Superseded by `MultiTaskTruthLensModel` for new code, but still used as a mixin for some task-specific variants.

**Key behaviors:**
- Shared encoder with `nn.ModuleDict` task heads.
- `forward()` loops over heads, collects per-task losses into a list, and sums with `torch.stack().sum()` (A4.1 — avoids Python-level loop accumulation on scalar tensors).
- Inference-only probability derivation gated by `not self.training` (P1).

---

### 5.5 `encoder/transformer_encoder.py` — `TransformerEncoder`

**Role:** Production encoder wrapper around HuggingFace `AutoModel`.

**Critical design decisions:**

| Decision | Rationale |
|---|---|
| `_EncoderOutput` dict-subclass with `__getattr__` | Supports both `out["pooled_output"]` (internal training code) and `out.last_hidden_state` (explainability / HF-style callers) without forcing migration |
| `embeddings` property → `self.encoder.embeddings` | Lets Integrated-Gradients attribution access the token embedding matrix directly |
| `inputs_embeds` alternate entry | IG re-enters the encoder with a custom embedding tensor (required for gradient attribution) |
| `_cached_device` plain attribute | Avoids per-batch `next(self.parameters()).device` generator construction (P2.7) |
| `use_compile` is a no-op (COMPILE-OFF) | `torch.compile` was removed project-wide; parameter retained for back-compat |
| `add_pooling_layer = False` in HF config | Prevents a second (unused) pooling head from being constructed inside the HF model |

**Pooling strategies:**

```
cls:       hidden[:, 0]
mean:      sum(hidden * mask_expanded) / clamp(mask_sum, min=1e-9)
attention: softmax(hidden.mean(-1)) weighted sum over sequence
```

---

### 5.6 `encoder/encoder_factory.py` — `EncoderFactory`

```python
EncoderFactory.create_transformer_encoder(EncoderConfig(...)) -> TransformerEncoder
```

- Single factory method that reads `EncoderConfig` and constructs a `TransformerEncoder`.
- Used by the convenience construction path (`MultiTaskTruthLensModel(config=...)`).

---

### 5.7 `multitask/multitask_truthlens_model.py` — `MultiTaskTruthLensModel`

**Role:** Flagship production model. Inherits `BaseModel` (not bare `nn.Module`).

**Three construction paths:**

| Path | Usage |
|---|---|
| `MultiTaskTruthLensModel(encoder, task_heads)` | Raw module injection (testing, research) |
| `MultiTaskTruthLensModel(config=MultiTaskTruthLensConfig(...))` | Convenience path (fast unit tests, default heads) |
| `MultiTaskTruthLensModel.from_model_config(MultiTaskModelConfig)` | Full-fidelity YAML-driven path (model registry, inference engine) |

**`_HeadsLogitsView`:** A read-only `Mapping` that wraps `task_heads` and adapts each head's `dict`-returning `forward()` into a callable that returns just the logits tensor. Exposed as `model.heads` so the explainability layer (which does `model.heads[task](cls_embedding)`) gets a tensor back without crashing.

**`forward(**inputs)`:**
1. Filters `inputs` to only `{"input_ids", "attention_mask"}` before passing to encoder (MT-MODEL-ENC-KWARG-FIX — prevents `labels` leaking into the strict encoder signature).
2. Calls `_extract_pooled(encoder_outputs)` — strict, no implicit CLS fallback (A6.1).
3. Loops over `task_heads`, enforces `dict` output with `"logits"` key (A3.4).
4. Builds `task_logits` as a thin view (not a copy) over per-task entries (A3.6).

**`_DEFAULT_TASK_SPEC`:** Single source of truth for all default head sizes and label names. Referenced by `BIAS_LABELS`, `IDEOLOGY_LABELS`, etc. class attributes for downstream label helpers and test suites.

---

### 5.8 `heads/classification_head.py` — `ClassificationHead`

| Config field | Type | Default | Effect |
|---|---|---|---|
| `input_dim` | int | required | Encoder hidden size (768 for roberta-base) |
| `num_classes` | int | required | Output classes |
| `hidden_dim` | int or None | None | Optional bottleneck layer |
| `dropout` | float | 0.1 | Applied before final linear |
| `activation` | str | `"gelu"` | `relu`, `gelu`, `tanh` |
| `use_layernorm` | bool | False | Normalizes input (and hidden if present) |
| `return_features` | bool | False | Adds pre-logit activations to output dict |

**Forward output (inference mode):**
```python
{
  "logits": Tensor(B, C),
  "probabilities": Tensor(B, C),    # log_softmax then exp (single pass — N1)
  "confidence": Tensor(B,),         # max prob per row via log_probs.max().exp()
  "entropy": Tensor(B,),            # -(probs * log_probs).sum(-1)
  "features": Tensor(B, hidden_dim) # only if return_features=True
}
```

---

### 5.9 `heads/multilabel_head.py` — `MultiLabelHead`

| Config field | Type | Default | Notes |
|---|---|---|---|
| `num_labels` | int | required | Independent binary labels |
| `threshold` | float | 0.5 | Binarization threshold |
| `activation` | str | `"gelu"` | Also supports `"elu"` |

**Forward output (inference mode):**
```python
{
  "logits": Tensor(B, L),
  "probabilities": Tensor(B, L),    # sigmoid
  "predictions": BoolTensor(B, L),  # probs >= threshold
  "confidence": Tensor(B,),         # mean(probs, dim=-1)
  "entropy": Tensor(B,),            # via logsigmoid / logsigmoid(-x)  (N1)
  "loss": Tensor()                  # BCEWithLogitsLoss — only if labels passed
}
```

---

### 5.10 `loss/multitask_loss.py` — `MultiTaskLoss`

**Role:** Multi-task loss orchestrator. The single `nn.Module` that every training step calls.

**Pipeline for each active task:**

```
raw_loss  = TaskLossRouter.compute(task, logits, labels)
           ↓ (if use_coverage)
coverage_update(task, labels)
           ↓ (if use_normalizer)
loss = EMALossNormalizer.normalize(task, raw_loss)
           ↓ (if coverage)
loss = EMACoverageTracker.weight(task, loss)
           ↓
weighted_loss = loss * task_config.weight
```

After all tasks:

```
if balancer: total_loss = balancer(raw_losses)
else:        total_loss = sum(weighted_losses)

if normalization == "active":  total_loss /= active_heads
if normalization == "fixed":   total_loss /= len(all_tasks)
if normalization == "sum":     unchanged
```

**Return value:** `(total_loss, raw_per_task_losses)` — the raw dict (MT-4) is what the task scheduler, auto-debug engine, and instrumentation consume; they need raw magnitudes, not post-normalized values.

**Device safety (GPU-3):** `loss_functions` is registered as `self.loss_functions` (an `nn.ModuleDict`) **before** being passed to `TaskLossRouter`, ensuring that `pos_weight`/`class_weights` buffers are moved by `MultiTaskLoss.to(device)`.

**Empty-batch guard:** If no task has both logits and labels in the batch, returns a zero loss that is still connected to the computation graph (if any input requires grad) so AMP/scaler stays consistent.

---

### 5.11 `loss/uncertainty.py` — `UncertaintyBalancer`

Implements the Kendall & Gal (2018) homoscedastic uncertainty approach:

```
weighted_i = exp(-log_var_i) * loss_i + log_var_i
total = Σ weighted_i
```

- `log_vars` is an `nn.ParameterDict`, so the learnable uncertainty weights are included in the model's parameter list and updated by the optimizer.
- Clamped to `[−10, 10]` to prevent vanishing/exploding precision terms.
- `get_weights()` returns `exp(-log_var)` per task — the effective precision (inverse uncertainty) for each task.

---

### 5.12 `loss/loss_normalizer.py` — `EMALossNormalizer`

Maintains a running Exponential Moving Average of each task's loss magnitude and divides the current loss by that average. This prevents tasks with intrinsically large loss scales from dominating tasks with small scales, independent of class distributions or dataset sizes. Alpha defaults to `0.1`; overridable from YAML via `normalizer_alpha`.

---

### 5.13 `loss/coverage_tracker.py` — `EMACoverageTracker`

Tracks what fraction of each batch actually has supervision for each task (sparse multi-task datasets often label only a subset of tasks per example). Tasks with low recent coverage receive a multiplier < 1 to dampen gradient noise from sparse batches; tasks with full coverage receive weight 1.0.

---

### 5.14 `calibration/temperature_scaling.py` — `TemperatureScaler`

```python
class TemperatureScaler(nn.Module):
    temperature = nn.Parameter(torch.ones(1))  # starts at T=1 (no scaling)

    def forward(self, logits):
        return logits / clamp(temperature, min=1e-12)
```

- Post-hoc calibration: fitted on validation logits via LBFGS after training, never updated by the main training optimizer.
- Belongs to `BaseModel.get_calibration_parameters()` (G4).
- CFG6 fix: previously lived in `src/evaluation/calibration`; moved here so the dependency arrow runs `evaluation → models`, not the reverse.

---

### 5.15 `calibration/isotonic_calibration.py` — `IsotonicCalibrator`

- Wraps sklearn's `IsotonicRegression` with multi-class support (one isotonic function per class).
- `fit(logits, labels)` and `predict_proba(probs)` interface.
- Non-parametric alternative to temperature scaling; better for datasets with non-monotonic miscalibration.

---

### 5.16 `inference/predictor.py` — `Predictor`

**Role:** Production inference orchestrator. Wraps a model with AMP, device management, calibration, and output formatting.

```python
predictor = Predictor(
    model=model,
    device="cuda",
    temperature_scaler=ts,
    isotonic_calibrator=ic,
)
result = predictor.predict(input_ids, attention_mask)
```

**Key behaviors:**
- All public methods decorated with `@torch.inference_mode()` (disables autograd entirely).
- AMP on CUDA: uses `torch.autocast` with dtype resolved from `TRUTHLENS_AMP_DTYPE` env var (`float16`, `bfloat16`, or `float32`). Falls back from bf16 to fp16 if the card lacks bf16 support.
- Output flattening (INFERENCE-CONTRACT-FIX V7): handles three output shapes: (1) legacy `_logits`-suffix tensors, (2) per-task dicts with `"logits"` key, (3) `"task_logits"` parallel dict. All are normalized to `{task}_logits`, `{task}_probabilities`, `{task}_predictions`, `{task}_confidence`, `{task}_entropy`.
- Calibration applied per-task via `_calibrate(logits, probs)`.
- `build_fake_real_output()` extracts fake-news probability from whichever head key matches `_FAKE_HEAD_KEYS`.

---

### 5.17 `checkpointing/checkpoint_manager.py` — `CheckpointManager`

**Role:** Production-grade training checkpoint manager.

**Save pipeline:**
```
build state dict
  → validate (key sanity check)
  → attach_schema (version tag)
  → atomic_save (write to temp file, rename)
  → fsync_dir (durability on Linux)
  → attach_integrity_metadata (SHA-256 of file)
  → save_metadata (metadata.json alongside checkpoint)
  → update_best_checkpoint (optional, if save_best=True)
```

**Load pipeline:**
```
resolve path (handles "latest", step number, or full path)
  → safe_load (weights_only=True)
  → prepare_checkpoint (schema migration if stale)
  → verify_from_metadata (SHA-256 check if metadata.json present)
  → backup_state = model.state_dict().clone()  (C1.8)
  → model.load_state_dict(strict=strict)
  → on failure: restore backup_state
  → load optimizer/scheduler/scaler states
```

**Distributed safety:** `_is_primary()` — only rank 0 writes checkpoints in DDP mode.

**Cleanup:** `cleanup(keep=3)` removes all but the N most recent checkpoint directories.

---

### 5.18 `registry/model_registry.py` — `ModelRegistry`

**Role:** Central model loading entry point for the inference engine and API.

```python
result = ModelRegistry.load_model(model_name="truthlens_model", device="cuda")
# Returns: {"model", "tokenizer", "vectorizer", "device", "metadata"}
```

**Load dispatch:**
- If `config.json` declares `model_type = "multitask_truthlens"` → `_load_multitask_model(path, device)`.
- If no model_type → HuggingFace `AutoModelForSequenceClassification.from_pretrained`.
- If explicit `model_type` argument → `ModelFactory.create(type, cfg_dict)` + load `model.pt`.

All paths use `weights_only=True` and log missing/unexpected keys.

---

### 5.19 `adapters/lora_adapter.py` — `LoRALinear`

Implements Low-Rank Adaptation:

```
W' = W + (alpha / r) * (B @ A)
```

- `A`: `(r, in_features)` — initialized Kaiming-uniform.
- `B`: `(out_features, r)` — initialized zeros (so delta starts at zero).
- `merge()`: folds the LoRA delta into `W` for zero-overhead inference.
- `unmerge()`: subtracts the delta to restore trainable separation.
- `apply_lora_to_linear(module, target_keywords=("query", "key", "value", "dense"))`: walks the module tree and replaces matching `nn.Linear` layers with `LoRALinear`.

---

### 5.20 `ensemble/ensemble_model.py` — `EnsembleModel`

**Strategies:** `average`, `weighted`, `majority_vote`.

**Majority vote fix:** Vote counts (integer per-class tallies) are normalized to probabilities *before* being passed downstream — the pre-fix code fed raw counts to `softmax`, which smeared probability mass incorrectly.

**Memory management (A5.5):** Per-member output dict is `del`-ed immediately after logit extraction; on CUDA, `torch.cuda.empty_cache()` is called to release the blocks before the next member forward — material for ensembles of 5+ models on 80 GB cards.

**Device safety (G3):** Runtime device is always read from `next(self.parameters()).device`, not the construction-time `config.device` string, so `EnsembleModel.to(new_device)` propagates correctly.

---

### 5.21 `uncertainty/mc_dropout.py` — `MCDropoutPredictor`

**Algorithm:**
1. Set model to `.eval()` (disables BN running stats).
2. Explicitly re-enable all `nn.Dropout` modules via `_enable_dropout()`.
3. Run `mc_samples` stochastic forward passes.
4. Stack on device, single `cpu().numpy()` transfer (G2 — avoids N per-sample device syncs).

**Uncertainty metrics:**
- `predictive_variance`: `var(samples, axis=0).mean(axis=1)` — spread across the sample dimension.
- `predictive_entropy`: `H[E_q[p(y|x)]]` — entropy of the mean prediction.
- `mutual_information`: `H[E_q[p]] - E_q[H[p]]` — epistemic uncertainty (BALD).

**`_xlogx` helper (N1):** Uses `np.where(p > 0, p * log(p), 0)` instead of `p * log(p + eps)` to avoid EPS bias at zero-probability entries.

---

### 5.22 `export/onnx_export.py` — `ONNXExporter`

- ONNX opset 17, dynamic batch axis by default.
- Post-export verification: runs the ONNX model via `onnxruntime` and checks `max(|pt_out - ort_out|) <= atol` (default `1e-4`).
- Requires `onnx` and `onnxruntime` (soft imports; raises `ImportError` if absent only at verification time).

---

### 5.23 `optimization/optimizer_factory.py`

**`build_parameter_groups(model)`:**
- Splits into `decay` / `no_decay` groups: biases and LayerNorm weights get `weight_decay=0`.
- Also excludes calibration parameters (A6.4): anything matching `_is_calibration_parameter_name()` is omitted entirely from the main optimizer's parameter list.

**`create_optimizer(model, optimizer_type, lr, ...)`:**
- Supports `adamw`, `adam`, `sgd`, `rmsprop`, `adagrad`.
- All branches honour the `weight_decay` argument (C1.4 bug fix — previous code silently fell back to `0` for non-AdamW optimizers).

---

## 6. Model Inputs & Outputs

### 6.1 Encoder Inputs

| Tensor | Shape | Dtype | Source |
|---|---|---|---|
| `input_ids` | `(B, L)` | `int64` | HF tokenizer |
| `attention_mask` | `(B, L)` | `int64` (0/1) | HF tokenizer |
| `inputs_embeds` | `(B, L, H)` | `float32` | IG attribution only |

`L` = sequence length, up to 512. `B` = batch size. `H` = hidden dim (768 for roberta-base).

### 6.2 Encoder Output

```python
_EncoderOutput({
    "sequence_output": Tensor(B, L, H),    # last_hidden_state alias
    "pooled_output":   Tensor(B, H),       # after pooling
    "last_hidden_state": Tensor(B, L, H),  # HF-style alias
})
```

### 6.3 Head Inputs / Outputs (Training)

```python
# Input to each head
features: Tensor(B, H)   # pooled encoder output, contiguous, 2D

# Output (training)
{"logits": Tensor(B, C)}

# Output (inference)
{
    "logits": Tensor(B, C),
    "probabilities": Tensor(B, C),
    "confidence": Tensor(B,),
    "entropy": Tensor(B,),
}
```

### 6.4 Loss Inputs / Outputs

```python
# MultiTaskLoss.forward()
logits: Dict[str, Tensor]   # {"bias": (B,2), "ideology": (B,3), ...}
labels: Dict[str, Tensor]   # {"bias": (B,), "emotion": (B,11), ...}

# Returns
(total_loss: Tensor[scalar], raw_losses: Dict[str, Tensor[scalar]])
```

### 6.5 Predictor (Flat) Output

After `_format_outputs` and calibration, every per-task entry is normalized to:

```python
{
    "{task}_logits":       Tensor(B, C),
    "{task}_probabilities":Tensor(B, C),
    "{task}_predictions":  Tensor(B,),    # argmax (multiclass) or bool mask (multilabel)
    "{task}_confidence":   Tensor(B,),
    "{task}_entropy":      Tensor(B,),
}
```

This flat format is what `src/aggregation/` and `src/inference/` consume.

---

## 7. Loss Functions & Optimization

### 7.1 Per-Task Loss Selection

| Task type | Loss function | When used |
|---|---|---|
| `multiclass` (balanced) | `CrossEntropyLoss(weight=None)` | Default |
| `multiclass` (imbalanced) | `CrossEntropyLoss(weight=class_weights)` | When inverse-frequency weights set by `training.loss_balancer` |
| `multiclass` (extreme imbalance) | `FocalLoss(gamma=2.0, weight=class_weights)` | When dominant class > focal threshold |
| `binary` / `multilabel` | `BCEWithLogitsLoss(reduction="none", pos_weight=...)` | Default for these types |
| `regression` | `MSELoss()` | When task type is regression |

### 7.2 Multi-Task Loss Pipeline

```
raw loss
  → EMA normalization (scale alignment across tasks)
  → coverage weighting (downweight sparse supervision)
  → static task weight (from config)
  ─────────────────────────────────────────────────
  Option A: no balancer → sum, divide by active heads
  Option B: UncertaintyBalancer → learned precision weighting
  Option C: custom BaseBalancer subclass (GradNorm, PCGrad, etc.)
```

### 7.3 Gradient Accumulation Notes

The trainer accumulates gradients across micro-batches before calling `optimizer.step()`. `MultiTaskLoss` is stateless per forward call; the `EMALossNormalizer` and `EMACoverageTracker` update their running statistics on every `forward()`, so they should be called at the micro-batch level, not the accumulation-step level.

### 7.4 Optimizer Parameter Groups

```python
# Correct pattern for production training
param_groups = build_parameter_groups(model)
# Groups: [{"params": decay_params, "weight_decay": 0.01},
#          {"params": no_decay_params, "weight_decay": 0.0}]
# Calibration params (temperature) are excluded from both groups.

optimizer = create_optimizer(model, "adamw", lr=2e-5)
```

---

## 8. Training Integration

### 8.1 Typical Training Step Pseudo-Code

```python
# Setup (once per training run)
model = MultiTaskTruthLensModel(config=config)
optimizer = create_optimizer(model, "adamw", lr=2e-5)
scheduler = get_lr_scheduler(optimizer, ...)
scaler = torch.cuda.amp.GradScaler()  # AMP gradient scaler
loss_fn = MultiTaskLoss(task_configs, normalization="active")
ckpt_mgr = CheckpointManager("checkpoints/")

# Training step
optimizer.zero_grad()
with torch.autocast("cuda", dtype=torch.float16):
    outputs = model(**batch)
    total_loss, per_task_losses = loss_fn(
        logits=outputs["task_logits"],
        labels=batch_labels,
        shared_parameters=gather_shared_parameters(model),
    )
scaler.scale(total_loss).backward()
loss_fn.on_after_backward()         # balancer hook
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
scheduler.step()
loss_fn.on_step_end()               # balancer hook

# Checkpointing
if step % save_every == 0:
    ckpt_mgr.save(
        step=step, model=model, optimizer=optimizer,
        scheduler=scheduler, scaler=scaler,
        metrics={"val_loss": val_loss},
        save_best=True,
    )
    ckpt_mgr.cleanup(keep=3)
```

### 8.2 Post-Training Calibration

```python
# After training is complete, on held-out validation set
scaler = TemperatureScaler()
scaler_optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01)
# fit loop: minimize NLLLoss on validation logits

# OR
ic = IsotonicCalibrator()
ic.fit(val_logits, val_labels)

# Attach to predictor
predictor = Predictor(model, temperature_scaler=scaler, isotonic_calibrator=ic)
```

---

## 9. Inference Logic

### 9.1 `Predictor` Entry Points

| Method | Input | Output |
|---|---|---|
| `predict(input_ids, attention_mask)` | Single sample tensors (auto-batched) | Flat dict, squeezed |
| `predict_batch(batch_dict)` | Dict of stacked tensors | Flat dict |
| `predict_batch_structured(batch_dict)` | Dict of stacked tensors | `PredictionOutput` object |
| `build_fake_real_output(formatted)` | Formatted flat dict | `{"label", "fake_probability", "confidence"}` |

### 9.2 AMP Dtype Resolution

The `TRUTHLENS_AMP_DTYPE` environment variable controls autocast precision:

| Value | Dtype |
|---|---|
| `float16` or `fp16` or `half` | `torch.float16` |
| `bfloat16` or `bf16` | `torch.bfloat16` (falls back to fp16 if unsupported) |
| `float32` | `torch.float32` (disables AMP effectively) |

This env var is respected by both `Predictor` and `PredictionPipeline` (GPU-3 fix).

### 9.3 Ensemble Inference

If `ensemble_model` is provided to `Predictor`, `_run_ensemble(batch)` is called instead of the standard forward. The ensemble model aggregates member logits according to its configured strategy and returns `{"ensemble_logits": tensor}`.

---

## 10. Model Checkpoints & Persistence

### 10.1 Checkpoint Directory Layout

```
checkpoints/
└── checkpoint-1000/
    ├── checkpoint.pt       # model + optimizer + scheduler + scaler state
    └── metadata.json       # step, epoch, metrics, SHA-256 hash
└── checkpoint-2000/
    ├── checkpoint.pt
    └── metadata.json
└── best_checkpoint.pt      # symlink or copy of best metric checkpoint
```

### 10.2 Atomic Save Protocol

1. Write to `checkpoint.pt.tmp` in the same directory.
2. `os.rename(tmp_path, final_path)` — atomic on POSIX filesystems.
3. `fsync` the directory fd — flushes the directory entry to disk before returning.
4. Compute SHA-256 of the final file and write to `metadata.json`.

This ensures no partial checkpoint is ever seen as valid by a concurrent reader.

### 10.3 Schema Versioning

`schema.py` tags every checkpoint with `{"schema_version": N}`. On load, `prepare_checkpoint()` runs migration functions for each version gap, ensuring old checkpoints remain loadable after schema changes.

### 10.4 Integrity Verification

On load, if `metadata.json` exists, `verify_from_metadata(path, meta)` recomputes the SHA-256 of the `.pt` file and compares against the stored hash. Mismatch raises `RuntimeError`, preventing silent corruption.

### 10.5 Rollback Safety (C1.8)

Before `model.load_state_dict()`, a full clone of the current state is taken. If `load_state_dict` raises for any reason, the model is restored to the pre-load state. The caller's model object is never left half-overwritten.

### 10.6 Model Export Formats

| Format | Class | File | Use case |
|---|---|---|---|
| PyTorch native | `BaseModel.save_checkpoint` | `.pt` | Training, fine-tuning |
| ONNX | `ONNXExporter` | `.onnx` | Serving, mobile |
| TorchScript | `TorchScriptExporter` | `.pt` (scripted) | C++ deployment |
| Dynamic INT8 | `quantize_dynamic()` | `.pt` | CPU inference speed |

---

## 11. Config Integration

### 11.1 `MultiTaskTruthLensConfig` (Convenience Path)

```python
@dataclass
class MultiTaskTruthLensConfig:
    model_name: str = "roberta-base"
    pooling: str = "cls"
    dropout: float = 0.1
    device: Optional[str] = None
    init_from_config_only: bool = False    # avoids HF download in tests
    bias_weight: float = 1.0
    ideology_weight: float = 1.0
    propaganda_weight: float = 1.0
    narrative_weight: float = 1.0
    emotion_weight: float = 1.0
    task_num_labels: Optional[Dict[str, int]] = None
    enabled_tasks: Optional[List[str]] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)
```

**Strict dataclass (CFG2):** Unknown fields raise `TypeError` at construction time. No silent ignore.

### 11.2 `MultiTaskModelConfig` (Full YAML-Backed Path)

```python
@dataclass
class MultiTaskModelConfig:
    encoder: EncoderConfig          # model_name, pooling, hidden_size, etc.
    tasks: Dict[str, TaskConfig]    # per-task loss_weight, num_classes, type
    training: TrainingConfig        # batch_size, epochs, lr, warmup, etc.
    uncertainty: UncertaintyConfig  # use_mc_dropout, mc_samples, etc.
    regularization: RegularizationConfig
    monitoring: MonitoringConfig
```

**Loading from YAML:**
```python
config = MultiTaskModelConfig.from_yaml("config/model_config.yaml")
model = MultiTaskTruthLensModel.from_model_config(config)
```

**A6 naming caveat:** `MultiTaskTruthLensConfig` and `MultiTaskModelConfig` are **not interchangeable**. Passing the wrong one raises `TypeError` immediately (not a silent misbehavior).

### 11.3 Config Fields That Affect Model Behavior at Inference

| Config field | Effect |
|---|---|
| `encoder.pooling` | Determines which encoder hidden states feed the heads |
| `encoder.model_name` | Which HF pretrained weights to load |
| `tasks.{name}.num_classes` | Head output size |
| `uncertainty.use_mc_dropout` | Whether `MCDropoutPredictor` is used at inference |
| `uncertainty.mc_samples` | Number of stochastic forward passes |

---

## 12. Optimization & Efficiency

### 12.1 Automatic Mixed Precision (AMP)

- Encoder uses `torch.autocast(device_type="cuda", dtype=float16)` on CUDA during both training and inference.
- Gradient scaler (`torch.cuda.amp.GradScaler`) prevents underflow during training.
- On MPS or CPU: AMP is disabled; full `float32` is used.
- `TRUTHLENS_AMP_DTYPE` env var allows operator override.

### 12.2 Gradient Checkpointing

```python
encoder = TransformerEncoder(model_name="roberta-base", gradient_checkpointing=True)
```

- Recomputes intermediate activations during backward pass instead of storing them.
- Reduces GPU memory by ~40-60% at the cost of ~30% extra compute time.
- Critical for training on 16 GB GPUs with long sequences or large batches.

### 12.3 LoRA (Parameter-Efficient Fine-Tuning)

```python
model = apply_lora_to_linear(model, r=8, alpha=16.0, dropout=0.1,
                              target_keywords=("query", "key", "value", "dense"))
# Freeze base weights, train only A and B matrices
for name, param in model.named_parameters():
    if "lora" not in name.lower():
        param.requires_grad = False
```

- Trains only ~0.1-1% of parameters for task adaptation.
- `merge()` folds the LoRA delta into base weights for inference with zero overhead.

### 12.4 MC Dropout Sample Efficiency (G2)

All MC dropout samples are accumulated as GPU tensors in a list; a single `torch.stack(...).cpu().numpy()` transfers the entire `(T, B, C)` block at once. The previous implementation did `T` separate CPU transfers, one per sample, which dominated inference wall time on GPU.

### 12.5 Ensemble Memory Management (A5.5)

Per-member output dicts are deleted immediately after logit extraction. On CUDA, `torch.cuda.empty_cache()` is called between members. This keeps peak GPU memory proportional to one member's activations rather than all members' simultaneously.

### 12.6 Encoder Device Caching (P2.7)

`TransformerEncoder._cached_device` is a plain `torch.device` attribute updated by `set_device()`. Used in `forward()` to avoid `next(self.parameters()).device` generator construction on every batch — measurable overhead at high batch rates on small inputs.

### 12.7 ONNX Export & Quantization

```python
# ONNX export
exporter = ONNXExporter(ONNXExportConfig(opset_version=17, verify_export=True))
exporter.export(model, example_input, "models/truthlens.onnx")

# Dynamic INT8 quantization (CPU inference only)
from src.models.export.quantization import quantize_dynamic
quantized = quantize_dynamic(model, dtype=torch.qint8)
```

---

## 13. Extensibility Guide

### 13.1 Adding a New Classification Task

**Step 1:** Add the task to `_DEFAULT_TASK_SPEC` in `multitask_truthlens_model.py`:
```python
_DEFAULT_TASK_SPEC["satire"] = {
    "task_type": "multi_class",
    "labels": ["not_satire", "satire"],
}
```

**Step 2:** Update the class-level label lists:
```python
SATIRE_LABELS: List[str] = list(_DEFAULT_TASK_SPEC["satire"]["labels"])
NUM_SATIRE: int = len(SATIRE_LABELS)
```

**Step 3:** Add a `TaskLossConfig` for the new task:
```python
task_configs["satire"] = TaskLossConfig(task_type="multiclass", weight=1.0)
```

**Step 4:** Add a `ClassificationHead` in `MultiTaskTruthLensModel._build_default_heads()`.

**Step 5:** Update `MultiTaskTruthLensConfig` with a `satire_weight` field.

**Step 6:** Add the task key to your YAML config's `tasks:` section.

**Step 7:** Update the aggregation pipeline in `src/aggregation/` to consume `satire_logits`, `satire_confidence`, etc.

### 13.2 Adding a Custom Loss Balancer

```python
class MyBalancer(BaseBalancer):
    def combine(self, task_losses: Dict[str, Tensor]) -> Tensor:
        # Implement gradient balancing logic
        ...

    def on_before_backward(self, raw_losses, *, shared_parameters=None):
        # Optional: GradNorm-style gradient shaping
        ...

loss_fn = MultiTaskLoss(task_configs)
loss_fn.attach_task_balancer(MyBalancer(task_names=list(task_configs)))
```

### 13.3 Adding a Custom Encoder

```python
class MyEncoder(BaseModel):
    def forward(self, input_ids, attention_mask):
        # Must return a dict with "pooled_output" key
        ...
        return {"pooled_output": pooled, "sequence_output": hidden}
```

Pass directly to `MultiTaskTruthLensModel(encoder=my_encoder, task_heads=...)`.

### 13.4 Adding a Custom Export Format

Subclass or implement alongside `ONNXExporter` in `export/`. The interface convention is:
- `export(model, example_input, output_path) -> Path`
- `verify(output_path, model, example_input) -> (bool, float)`

### 13.5 Swapping Calibration Methods

```python
predictor.set_temperature_scaler(new_scaler)     # replaces current
predictor.set_isotonic_calibrator(new_calibrator)
```

Both are applied per-task in `_calibrate(logits, probs)`: temperature scaler takes logits and returns adjusted probabilities; isotonic calibrator takes probabilities (numpy array) and returns calibrated probabilities.

---

## 14. Common Pitfalls & Risks

### P1 — Computing Probabilities During Training

**Risk:** Calling `predict_batch()` on the model while `model.training == True` skips probabilities/entropy (guarded by `not self.training`). Any downstream consumer that reads `{task}_probabilities` will get `KeyError`.

**Fix:** Always call `model.eval()` before inference. The `Predictor` class does this in `__init__`.

### P2 — Device Mismatch on `pos_weight` / `class_weights`

**Risk:** `BCEWithLogitsLoss(pos_weight=...)` and `CrossEntropyLoss(weight=...)` register their weight tensors as buffers on the loss module. If the loss is a plain Python object (not an `nn.Module`), `.to(device)` won't propagate to these buffers.

**Fix (GPU-3):** `MultiTaskLoss.loss_functions` is an `nn.ModuleDict` registered on `self` before being passed to `TaskLossRouter`. A single `MultiTaskLoss.to(device)` propagates to every per-task loss buffer automatically.

### P3 — Calibration Parameters in the Main Optimizer

**Risk:** If `build_parameter_groups(model)` is called without A6.4 exclusion, the temperature scalar participates in gradient updates every training step, destroying post-hoc calibration.

**Fix:** Always use `build_parameter_groups(model)` (which calls `_is_calibration_parameter_name`) or `model.get_optimization_parameters()` as the optimizer's parameter source.

### P4 — Mixing `MultiTaskTruthLensConfig` and `MultiTaskModelConfig`

**Risk:** Both have similar names but different fields. Passing the wrong one silently constructs a different model with wrong head sizes.

**Fix (CFG2):** `MultiTaskTruthLensModel.__init__` performs an `isinstance(config, MultiTaskTruthLensConfig)` check and raises `TypeError` immediately if the wrong type is passed.

### P5 — `strict=False` Load Without Backup

**Risk:** `load_state_dict(strict=False)` mutates the model in place. If validation fails post-load (e.g., unexpected head sizes), the model is left in a half-overwritten state.

**Fix (C1.8):** `CheckpointManager.load()` takes a full clone of the state before calling `load_state_dict`, and restores it if an exception occurs.

### P6 — Unsafe Checkpoint Loading

**Risk:** `torch.load(..., weights_only=False)` can execute arbitrary pickle code from a malicious or corrupted `.pt` file.

**Fix (C1.3):** All `torch.load` calls in this module use `weights_only=True`. Never override this for files from external sources.

### P7 — Entropy Bias from EPS in Log

**Risk:** Computing entropy as `-sum(p * log(p + eps))` introduces a fixed bias term `eps * log(eps)` per class for peaked distributions. With `eps=1e-12` and 128 classes, this is detectable.

**Fix (N1):** All entropy computations use `F.log_softmax(logits)` (classification) or `F.logsigmoid(logits)` / `F.logsigmoid(-logits)` (multilabel) to stay in log-space without an EPS term.

### P8 — Implicit CLS Fallback in Pooling

**Risk:** If the encoder fails to produce a pooled output (e.g., a new encoder wrapper that doesn't set `pooled_output`), the old code silently fell back to `hidden[:, 0]` — producing wrong embeddings for non-CLS pooling strategies.

**Fix (A6.1):** `_extract_pooled()` raises `RuntimeError` if neither `pooled_output` nor `pooler_output` is present. The encoder must be configured with an explicit `pooling=` strategy.

### P9 — Head Returns Tensor Instead of Dict

**Risk:** Custom task heads that return a raw tensor (not a dict) will pass through undetected and crash deep inside calibration or aggregation.

**Fix (A3.4):** `MultiTaskTruthLensModel.forward()` enforces `isinstance(head_output, dict)` and `"logits" in head_output` with named error messages. Tensor-returning heads fail immediately at the head boundary.

### P10 — `task_logits` Dict Mutation De-Sync

**Risk:** Previously, `task_logits` was a separate dict populated alongside per-task entries. Mutating one de-synced the other silently.

**Fix (A3.6):** `task_logits` is now a thin comprehension view computed at the end of `forward()`: `{name: outputs[name]["logits"] for name in task_heads}`. There is one owner; no copy.

---

## 15. Example Usage

### 15.1 Build and Run the Model (Convenience Path)

```python
from src.models.multitask.multitask_truthlens_model import (
    MultiTaskTruthLensModel,
    MultiTaskTruthLensConfig,
)
from transformers import AutoTokenizer
import torch

config = MultiTaskTruthLensConfig(
    model_name="roberta-base",
    pooling="cls",
    dropout=0.1,
    init_from_config_only=True,  # skip HF download for testing
)
model = MultiTaskTruthLensModel(config=config)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
text = "Scientists discover a new treatment for cancer."
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

with torch.inference_mode():
    outputs = model(**inputs)

# Access per-task logits
print(outputs["bias"]["logits"].shape)      # (1, 2)
print(outputs["ideology"]["logits"].shape)  # (1, 3)
print(outputs["emotion"]["logits"].shape)   # (1, 11)
```

### 15.2 Full Production Inference via Predictor

```python
from src.models.registry.model_registry import ModelRegistry
from src.models.inference.predictor import Predictor
import torch

# Load model from disk
bundle = ModelRegistry.load_model("truthlens_model", device="cuda")
model = bundle["model"]
tokenizer = bundle["tokenizer"]

# Wrap in Predictor
predictor = Predictor(model, device="cuda")

# Tokenize
inputs = tokenizer(
    ["Breaking: government confirms alien contact."],
    return_tensors="pt",
    max_length=512,
    truncation=True,
    padding=True,
)

# Predict
result = predictor.predict(inputs["input_ids"], inputs["attention_mask"])

print(result["bias_probabilities"])      # Tensor [non_bias, bias]
print(result["ideology_predictions"])    # 0=left, 1=center, 2=right
print(result["emotion_probabilities"])   # Tensor [11 emotions]
```

### 15.3 Training Loop with MultiTaskLoss

```python
from src.models.loss.multitask_loss import MultiTaskLoss, TaskLossConfig
from src.models.optimization.optimizer_factory import create_optimizer
from src.models.checkpointing.checkpoint_manager import CheckpointManager

task_configs = {
    "bias":         TaskLossConfig(task_type="multiclass", weight=1.0),
    "ideology":     TaskLossConfig(task_type="multiclass", weight=1.0),
    "propaganda":   TaskLossConfig(task_type="multiclass", weight=1.0),
    "narrative":    TaskLossConfig(task_type="multilabel", weight=0.8),
    "emotion":      TaskLossConfig(task_type="multilabel", weight=0.5),
}

loss_fn = MultiTaskLoss(task_configs, normalization="active", use_normalizer=True)
optimizer = create_optimizer(model, "adamw", learning_rate=2e-5, weight_decay=0.01)
ckpt_mgr = CheckpointManager("checkpoints/run_001/")

for step, batch in enumerate(train_loader):
    optimizer.zero_grad()

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    labels = {k: batch[k] for k in task_configs}

    total_loss, per_task = loss_fn(outputs["task_logits"], labels)
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 500 == 0:
        ckpt_mgr.save(step=step, model=model, optimizer=optimizer,
                      metrics={"train_loss": total_loss.item()})
        ckpt_mgr.cleanup(keep=3)
```

### 15.4 Uncertainty Estimation

```python
from src.models.uncertainty.mc_dropout import MCDropoutPredictor

mc_predictor = MCDropoutPredictor(
    model=model,
    task_type="multiclass",
    mc_samples=20,
    device=torch.device("cuda"),
)

uncertainty = mc_predictor.predict_with_uncertainty(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    task="bias",
)

print(uncertainty["mean_probabilities"])  # shape (B, C)
print(uncertainty["mutual_information"])  # epistemic uncertainty (BALD)
print(uncertainty["entropy"])             # predictive entropy
```

### 15.5 LoRA Fine-Tuning

```python
from src.models.adapters.lora_adapter import apply_lora_to_linear

# Apply LoRA to attention layers only
model = apply_lora_to_linear(
    model,
    r=8,
    alpha=16.0,
    dropout=0.1,
    target_keywords=("query", "key", "value"),
)

# Freeze everything except LoRA parameters
for name, param in model.named_parameters():
    if ".A" not in name and ".B" not in name:
        param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable:,}")  # typically ~300K vs 125M
```

### 15.6 ONNX Export

```python
from src.models.export.onnx_export import ONNXExporter, ONNXExportConfig

exporter = ONNXExporter(ONNXExportConfig(
    opset_version=17,
    dynamic_batch=True,
    verify_export=True,
    atol=1e-4,
))

example = torch.randint(0, 50265, (1, 64))  # (B, L) roberta vocab size
exporter.export(model, example, "deployed/truthlens.onnx")
# Automatically verifies PyTorch vs ONNX Runtime outputs within atol
```

---

## 16. Simple Explanation (For Non-Technical Reviewers)

Imagine TruthLens AI as a panel of six expert analysts sitting around a table. Every article or claim that comes into the system is first read and summarized by a shared reader — the **encoder** — which compresses the text into a dense numeric fingerprint (768 numbers per article). That fingerprint is then passed simultaneously to all six analysts:

| Analyst | Question they answer |
|---|---|
| **Bias Detector** | Is this content biased or neutral? |
| **Ideology Classifier** | Is the political lean left, center, or right? |
| **Propaganda Detector** | Does this use propaganda techniques? |
| **Narrative Role Analyst** | Does the writing cast subjects as heroes, villains, or victims? |
| **Frame Analyst** | Which news framing strategy is being used? |
| **Emotion Analyzer** | What emotions does this content evoke? |

Each analyst independently gives their assessment, with a confidence score and an uncertainty estimate. The results from all six analysts are then fed to an aggregation layer that combines them into the final TruthLens credibility score.

**Key safety properties of the model system:**

- **Checkpoints are always saved atomically:** If the server crashes mid-save, no corrupted checkpoint is ever seen. Every saved file is integrity-verified with a checksum.
- **Calibration is always post-hoc:** The system's confidence scores are tuned *after* training, on data the model has never seen, so that "80% confidence" actually means "correct 80% of the time."
- **Uncertainty is always reported:** When the model is unsure (e.g., the article is ambiguous or unlike anything in training), it says so explicitly rather than guessing with false confidence.
- **No arbitrary code is ever executed from model files:** All checkpoint loading uses a safe mode that only reads numeric weights, not arbitrary Python code.
