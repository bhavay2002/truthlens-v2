# TruthLens AI — `src/models/` Production Audit

**Audit date:** 2026-05-03  
**Scope:** Every Python file under `src/models/` (encoder, multitask, registry, architectures, inference, ensemble, distillation, regularization, representation, uncertainty, interpretability, export, optimization, monitoring, metadata, utils, checkpointing, tasks, emotion, heads, loss)  
**Auditor:** Automated full-read inspection + manual reasoning  

---

## Executive Summary

Two issues were identified across the entire `src/models/` tree. One was a **critical AMP-dtype bug** that caused bf16-configured models to silently run fp16 autocast. It has been fixed. The other was a **minor numerical inconsistency** in two ensemble files where entropy was computed with an additive epsilon instead of the exact log-space formula; it has also been fixed. All remaining files are production-clean.

---

## Findings Table

| ID | Severity | Status | File | Line(s) | Description |
|----|----------|--------|------|---------|-------------|
| ENCODER-DTYPE-BUG | **Critical** | **Fixed** | `src/models/encoder/transformer_encoder.py` | 261 | `autocast_dtype` hardcoded to `torch.float16`; ignored `self.amp_dtype`. bf16-configured runs silently used fp16, risking overflow on Ampere+ hardware. Fixed: `autocast_dtype = self.amp_dtype`. |
| N1 | Minor | **Fixed** | `src/models/ensemble/stacking_ensemble.py` | 88 (old 95 after comment) | Entropy computed as `-(probs * log(probs + 1e-12)).sum()`. Additive epsilon shifts the argument off the simplex, introducing a bias when any probability is near 1. Fixed: `F.log_softmax` + `-(probs * log_probs).sum()`, matching `ensemble_model.py`. |
| N1 | Minor | **Fixed** | `src/models/ensemble/weighted_ensemble.py` | 99 (old 106 after comment) | Same `log(probs + 1e-12)` pattern as above. Fixed identically. |

---

## Detailed Findings

### ENCODER-DTYPE-BUG — Critical (Fixed)

**File:** `src/models/encoder/transformer_encoder.py`  
**Line:** 261 (pre-fix)

**Root cause:**  
Inside `TransformerEncoder.forward`, the AMP autocast context was opened with a dtype resolved from a local variable:

```python
# Before fix
autocast_dtype = torch.float16   # hardcoded — ignored self.amp_dtype
with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=self.use_amp):
    ...
```

`self.amp_dtype` (set from config at construction) was never consulted. Any caller that configured `amp_dtype = "bfloat16"` would receive `torch.float16` autocast instead — the worst possible silent failure in a precision-critical model path.

**Impact:**
- On Ampere+ (A100, H100) where bf16 is preferred for its wider dynamic range, the model ran fp16 instead, raising the risk of gradient overflow and loss NaN-out.
- The mismatch between encoder AMP dtype and `TrainingStep` AMP dtype could cause dtype-mismatch errors deep in `nn.LayerNorm` or head layers that expected bf16 inputs.

**Fix applied:**
```python
# After fix
autocast_dtype = self.amp_dtype   # resolved from config: torch.bfloat16 or torch.float16
with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=self.use_amp):
    ...
```

---

### N1 — Entropy Bias in Ensemble Files (Fixed)

**Files:**  
- `src/models/ensemble/stacking_ensemble.py`  
- `src/models/ensemble/weighted_ensemble.py`

**Root cause:**  
Both files computed Shannon entropy as:

```python
entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
```

The `+ 1e-12` epsilon guard is intended to avoid `log(0)`, but it incorrectly shifts the argument when `probs` is near 1.0:

- `log(1.0 + 1e-12) ≈ 1e-12` instead of `log(1.0) = 0`
- This makes a fully-confident prediction appear to have tiny but non-zero entropy, biasing downstream uncertainty-gated routing.

The parent class `ensemble_model.py` already used the correct log-space formula (noted explicitly with `# N1: entropy in log-space — never log(probs + EPS).`).

**Fix applied (both files):**
```python
log_probs = F.log_softmax(logits, dim=-1)  # numerically stable; reuses softmax denominator
entropy = -(probs * log_probs).sum(dim=-1)
```

`F.log_softmax` is numerically stable by construction (uses the log-sum-exp trick) and never produces `log(0)` for valid logits, so no epsilon guard is needed.

---

## File-by-File Clean Status

All files below were read in full and found to contain no production issues.

| Directory / File | Clean? | Notes |
|-----------------|--------|-------|
| `encoder/transformer_encoder.py` | Yes (after fix) | ENCODER-DTYPE-BUG fixed |
| `multitask/multitask_model.py` | Yes | — |
| `registry/model_registry.py` | Yes | — |
| `architectures/` (all files) | Yes | — |
| `inference/` (all files) | Yes | — |
| `ensemble/ensemble_model.py` | Yes | Already used log-space entropy; was the reference |
| `ensemble/stacking_ensemble.py` | Yes (after fix) | N1 fixed |
| `ensemble/weighted_ensemble.py` | Yes (after fix) | N1 fixed |
| `distillation/` (all files) | Yes | — |
| `regularization/` (all files) | Yes | — |
| `representation/` (all files) | Yes | — |
| `uncertainty/` (all files) | Yes | — |
| `interpretability/` (all files) | Yes | — |
| `export/` (all files) | Yes | — |
| `optimization/` (all files) | Yes | — |
| `monitoring/` (all files) | Yes | — |
| `metadata/` (all files) | Yes | — |
| `utils/` (all files) | Yes | — |
| `checkpointing/` (all files) | Yes | — |
| `tasks/` (all files) | Yes | — |
| `emotion/` (all files) | Yes | — |
| `heads/` (all files) | Yes | — |
| `loss/` (all files) | Yes | — |

---

## Verdict

**2 issues found. 2 fixed. 0 open.**  
`src/models/` is production-ready.
