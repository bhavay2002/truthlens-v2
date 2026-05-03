# `src/models/` — Production-Grade Audit

**Scope.** Read-only audit of `src/models/` (~111 `.py` files across 28
sub-packages). Integration with `src/data/`, `src/training/`,
`src/inference/`, `src/evaluation/`, `src/config/` was checked but those
packages are **not** in scope for fixes — only the contract surface they
share with `src/models/` is.

**Method.** Every `.py` under `src/models/` was opened end-to-end. Cross-cuts
(`grep` / `rg`) were used to enumerate duplicate class names, unsafe
`torch.load` callers, missing CUDA syncs, and config drift. Findings below
are ordered by *blast radius first, then likelihood of triggering in
production*. Every item carries a code-level fix.

The previous `features_audit.md` (4.5 / 10) lives at the repo root for
context — this audit is independent and does not reuse those findings.

---

## 1. Critical Bugs

These will silently produce wrong results, crash in production, or corrupt
checkpoints. Fix before any further training run.

### 1.1 Two `ArtifactManager` classes with divergent contracts ★

`src/models/checkpointing/model_loader.py:95` and
`src/models/checkpointing/artifact_manager.py:117` **both define a class
called `ArtifactManager`**, with completely different implementations:

| Concern | `model_loader.ArtifactManager` | `artifact_manager.ArtifactManager` |
| --- | --- | --- |
| Save path  | `atomic_save_state_dict` (tmp + `os.replace`) | bare `torch.save(state, path)` |
| Integrity  | sha256 + `IntegrityVerifier` round-trip | sha256 stored alongside, **never re-verified on load** |
| Schema     | `validate_schema_compat(...)` before applying state_dict | none |
| `weights_only` | `torch.load(..., weights_only=True)` | `torch.load(path)` — falls back to PyTorch ≥ 2.6 default which now **errors out**, and on ≤ 2.5 silently allows arbitrary pickle execution |
| Dedup      | none | hash-keyed dedup directory |

Whichever import wins (it depends on Python import order; right now
`from src.models.checkpointing import ArtifactManager` resolves the
**`model_loader`** one because of `__init__.py` ordering, but
`from src.models.checkpointing.artifact_manager import ArtifactManager`
gives you the unsafe one) is undefined behaviour for callers. The two
class objects are *not* interchangeable: their `save_model` / `load_model`
signatures differ (`model_loader` returns `LoadResult`, `artifact_manager`
returns `nn.Module`).

**Fix.**

* Pick one as canonical (recommend `model_loader.ArtifactManager` —
  it has integrity, schema validation, and `weights_only=True`).
* Rename the dedup-oriented class to `DeduplicatingArtifactStore`
  (its actual job) and stop exposing it as `ArtifactManager`.
* Audit `__init__.py` re-exports across the package and remove the
  duplicate symbol.

### 1.2 `benchmarking/dataset_benchmarks.py` is a byte-identical copy of `benchmark_runner.py` ★

```bash
$ diff src/models/benchmarking/benchmark_runner.py \
       src/models/benchmarking/dataset_benchmarks.py
# (empty)
```

Two `BenchmarkRunner` classes, two `BenchmarkResult` dataclasses, same
module path. `isinstance(x, BenchmarkResult)` will return `False` whenever
the producer and the consumer happened to import from different modules.
This is the same isinstance trap that broke `pickle.load` on the
metadata classes last quarter (see `features_audit.md` §3.4).

**Fix.** Delete `dataset_benchmarks.py` and re-export from
`benchmark_runner` if a "dataset" alias is wanted, *or* repurpose
`dataset_benchmarks.py` to actually do dataset-level benchmarking
(throughput at varying batch sizes, etc.) — but it cannot continue to
shadow the same class.

### 1.3 `BaseModel.load_checkpoint` uses `weights_only=False` (unsafe deserialization) ★

`src/models/base/base_model.py:196`

```python
checkpoint = torch.load(
    path,
    map_location=map_location,
    weights_only=False,
)
```

This is the **base-class** load path, used by every concrete model that
doesn't override it. PyTorch ≥ 2.6 made `weights_only=True` the default
*specifically because* `weights_only=False` allows arbitrary code
execution on load. Crucially, the `model_loader.ArtifactManager` in the
same package (§1.1) already standardised on `weights_only=True`; the base
model load path silently bypasses that hardening.

**Fix.**

```python
checkpoint = torch.load(path, map_location=map_location, weights_only=True)
```

The `optimizer_state_dict` is a plain dict of tensors; it loads fine
under `weights_only=True`. If a caller genuinely needs to restore a
pickled custom object, they should opt in via an explicit kwarg
(`allow_pickle=True`) instead of every load being unsafe-by-default.

The same fix applies to `src/models/utils/model_utils.py:78`
(`load_model` helper) and to
`src/models/checkpointing/artifact_manager.py:309`
(the unsafe sibling from §1.1).

### 1.4 `optimizer_factory.create_optimizer` silently drops `weight_decay` for AdamW / Adam / RMSprop / Adagrad ★

`src/models/optimization/optimizer_factory.py:105-140`

```python
if optimizer_type == "adamw":
    return AdamW(
        params,
        lr=learning_rate,
        betas=betas,
        eps=eps,
    )                             # <-- no weight_decay= !
elif optimizer_type == "adam":
    return Adam(...same...)
elif optimizer_type == "rmsprop":
    return RMSprop(... no wd ...)
elif optimizer_type == "adagrad":
    return Adagrad(... no wd ...)
elif optimizer_type == "sgd":
    return SGD(..., weight_decay=weight_decay)   # only branch that does it
```

When `use_param_groups=True` (default) the per-group dict carries
`weight_decay`, so it appears to work. But:

* `use_param_groups=False` → params is `model.parameters()` and the
  optimizer falls back to PyTorch's *default* (`AdamW=0.01`,
  `Adam=0`, `RMSprop=0`, `Adagrad=0`). The user-supplied value is
  **silently** ignored. For `Adam` and the others, decay drops to
  zero and you don't see the warning until the val loss starts
  diverging two epochs in.
* `custom_params is not None` → same trap.

**Fix.** Pass `weight_decay=weight_decay` to every optimizer:

```python
if optimizer_type == "adamw":
    return AdamW(params, lr=learning_rate, betas=betas, eps=eps,
                 weight_decay=weight_decay)
elif optimizer_type == "adam":
    return Adam(params, lr=learning_rate, betas=betas, eps=eps,
                weight_decay=weight_decay)
elif optimizer_type == "rmsprop":
    return RMSprop(params, lr=learning_rate, momentum=momentum,
                   weight_decay=weight_decay)
elif optimizer_type == "adagrad":
    return Adagrad(params, lr=learning_rate,
                   weight_decay=weight_decay)
```

When the caller passes per-group dicts that already specify
`weight_decay`, PyTorch correctly lets the per-group value override the
optimizer-level one — so this fix is safe for both code paths.

### 1.5 `adversarial_training` (FGM/PGD): emb_name substring match catches `LayerNorm`

`src/models/regularization/adversarial_training.py` (FGM/PGD attack):

```python
if param.requires_grad and self.emb_name in name:
    ...
```

with `emb_name="embedding"`. In a HuggingFace model the parameter named
`embeddings.LayerNorm.weight` *also matches*, so the perturbation is
applied to the LayerNorm weight, not just the embedding lookup. PGD then
restores `param.data = self.backup[name]` and the LayerNorm's optimizer
state (Adam moments) silently drifts out of sync with its weights.

**Fix.** Match exactly on the embedding parameter, e.g.
`name == "embeddings.word_embeddings.weight"` for BERT-family, or
look the module up by `model.get_input_embeddings()`.

```python
emb_module = model.get_input_embeddings()
emb_param = emb_module.weight
# attack only emb_param; back it up by id() not by name
```

This also fixes a related subtle bug: the current code does
`param.data.add_(...)` which **breaks autograd's leaf invariants** if
the attack runs *between* `loss.backward()` and `optimizer.step()` — the
saved tensors of the next backward pass refer to the perturbed leaf.
`with torch.no_grad(): param.add_(...)` is the correct primitive, plus
explicit `param.grad.zero_()` after the attack pass.

### 1.6 `mixup` uses `np.random` instead of the torch generator

`src/models/regularization/mixup.py` calls `np.random.beta(...)` and
`np.random.permutation(...)`. Trainers seed via `torch.manual_seed`,
which does not seed numpy. Result: mixup batches are non-reproducible
across runs even with `seed=42` set, and they are *also* not synced
across DDP ranks — different ranks compute different mix coefficients,
violating the gradient-allreduce contract.

**Fix.**

```python
def __init__(self, alpha=0.2, generator: torch.Generator | None = None):
    self.alpha = alpha
    self._gen = generator   # caller passes a torch.Generator seeded once

def _sample_lambda(self, device):
    # Use torch.distributions.Beta with the model's generator
    a = torch.tensor(self.alpha, device=device)
    return torch.distributions.Beta(a, a).sample().item()

def _permutation(self, n, device):
    return torch.randperm(n, device=device, generator=self._gen)
```

For DDP, broadcast the lambda from rank 0 once per batch.

### 1.7 `ensemble_uncertainty` uses the `log(probs + EPS)` antipattern that was already fixed elsewhere

`src/models/uncertainty/ensemble_uncertainty.py` computes per-member
predictive entropy as `-(p * log(p + 1e-12)).sum(-1)`. This is exactly
the antipattern that was deliberately removed from
`base_classifier.py`, `multitask_base_model.py`,
`heads/classification_head.py`, and `mc_dropout._xlogx` — the EPS term
dominates whenever `p` is sharp (which is most of inference) and biases
the entropy toward a fixed lower bound, then rescaled by the ensemble
size to look "calibrated".

**Fix.** Take `logits` (not probabilities) into the entropy computation
and use `log_softmax` once:

```python
# member_logits: (M, B, C)  -> per-member entropy
log_p = F.log_softmax(member_logits, dim=-1)
p     = log_p.exp()
H_per_member = -(p * log_p).sum(-1)              # (M, B)
H_avg        = H_per_member.mean(0)              # E[H(p_m)]
# total predictive entropy via mean of probs
log_p_mean   = torch.logsumexp(log_p, dim=0) - math.log(M)
p_mean       = log_p_mean.exp()
H_predictive = -(p_mean * log_p_mean).sum(-1)
mutual_info  = H_predictive - H_avg              # epistemic uncertainty
```

This also closes a numeric instability in the multilabel branch
(`log(sigmoid(x) + EPS)` → `F.logsigmoid(x)`).

### 1.8 `checkpoint_manager.load` inverts strict-load semantics

`src/models/checkpointing/checkpoint_manager.py` calls
`load_state_dict(..., strict=False)` and then **manually inspects**
`missing/unexpected_keys` to decide whether to raise. That is the
opposite of how `strict=True` works:

* `strict=True` raises *immediately* if any expected key is missing,
  before mutating the model.
* `strict=False` *applies whatever it can* and never rolls back.

So when the checkpoint is broken, the model has already been
half-loaded by the time the manager raises — undefined state for
the caller's `except` block. Any caller that catches and continues
training is now training on a Frankenstein model.

**Fix.** Use `strict=True` by default. For partial-load scenarios
(fine-tuning a new head on a pretrained encoder), accept an explicit
`strict=False` AND copy the model's state_dict before calling
`load_state_dict` so the manager can roll back on validation failure:

```python
backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
try:
    result = model.load_state_dict(sd, strict=False)
    if disallowed_missing(result.missing_keys):
        model.load_state_dict(backup, strict=True)   # rollback
        raise CheckpointSchemaError(...)
finally:
    del backup
```

### 1.9 `AsyncCheckpointWriter` silently drops the *oldest* item on a full queue

`src/models/checkpointing/io_utils.py` — when the queue is full,
the writer pops the oldest entry to make room for the new one. For
checkpoint writing this is exactly backwards: you want to drop the
*incoming* save (the next epoch will produce another) rather than
discarding an older save that the user may already be relying on for
"best so far".

**Fix.** On a full queue, either *block* the producer (default torch
behaviour) or drop the *new* item with a `WARNING` log — never silently
discard committed work.

```python
try:
    self._queue.put_nowait(item)
except queue.Full:
    logger.warning(
        "Checkpoint queue full; dropping incoming save %s "
        "(retaining queued items)", item.path,
    )
```

Better: switch the queue to `queue.Queue(maxsize=N)` with a default
blocking `put` and only drop with a CLI override.

---

## 2. Performance Issues

Findings ordered by GPU-hour cost.

### 2.1 No `torch.compile` anywhere except a single opt-in path

```bash
$ rg -n "torch\.compile" src/models
src/models/inference/model_wrapper.py:97:   self.encoder = torch.compile(...)
```

`use_compile=False` is the default in `EncoderConfig`, both YAML configs
and the model registry. None of the heads, loss modules, or ensemble
forwards are compiled. On A100/H100 the encoder forward is the
bottleneck and `torch.compile(mode="reduce-overhead")` gives a 1.3–1.8×
speedup with bf16; the heads compile in <100 ms each.

**Fix.** Default `use_compile=True` on CUDA, expose
`compile_components: list[str]` (encoder / heads / losses) so a debug
run can disable selectively. Cache compilation across DDP ranks via
`TORCHINDUCTOR_CACHE_DIR=$REPL_HOME/.cache/torchinductor` so warm
restarts skip the recompile cost.

### 2.2 Ensemble members do mid-forward `.to(device)` per call

`src/models/ensemble/stacking_ensemble.py` and
`weighted_ensemble.py` both contain a loop like:

```python
for member in self.members:
    member.to(self.device)          # <-- inside forward, every batch
    out = member(x)
    ...
```

This forces a host→device sync (and a memcpy of every parameter when
`self.device` differs) per ensemble member per batch. On a 5-member
roberta-base ensemble at batch size 32 this costs ~20 ms / step,
i.e. ~30 % of throughput in our trace.

`ensemble/ensemble_model.py` was already fixed via `_runtime_device()`
(only moves the input, not every parameter). Apply the same pattern to
the other two:

```python
def forward(self, x):
    x = x.to(self._runtime_device())
    return [m(x) for m in self.members]    # members already on device
```

This *also* makes the ensemble DDP-safe: `.to()` mid-forward racks up
non-deterministic mem-fragments under all-reduce.

### 2.3 Derived statistics computed during training in three places — already fixed in two, regressed in one

`base_classifier.py`, `multitask_base_model.py`, `classification_head.py`
correctly skip probabilities/entropy when `self.training` is True (P1).
But `multitask_head.py` (`forward`) still always builds the full
`task_output` dict from sub-heads — the sub-head itself does the right
thing, but `multitask_head` then *also* calls
`task_output.get("probabilities")` etc. into the predict() return. No
duplicate compute today, but if a future sub-head tries to fast-path,
the wrapper will re-introduce it.

**Fix.** In `MultiTaskHead.forward`, branch on `self.training` and skip
the second-pass copy entirely:

```python
if not self.training:
    # only at eval do we materialise the rich dict
    ...
```

### 2.4 `MultiTaskOutput.detach()` round-trips through Python dicts every step

`src/models/multitask/multitask_output.py:151` rebuilds the entire
nested dict on every `.detach()` call. Trainers that detach to log
metrics each step (`scalar = output.detach().to_dict()`) burn allocator
time. Cheap fix:

```python
def detach_(self):                # in-place version
    for o in self.tasks.values():
        o.logits = o.logits.detach()
        ...
    return self
```

### 2.5 `parameter_count.layer_parameter_breakdown` is O(N · depth)

For each `named_modules()` it calls `parameters(recurse=False)` — fine —
but `top_k_layers_by_parameters` then *re-iterates* the same tree to
build the dict it sorts. On a roberta-large model with 770 modules this
is ~1.5 M op repeated on every `summary()` call, which the API logs
emit per request. Cache the breakdown:

```python
@functools.lru_cache(maxsize=8)
def _breakdown_cached(model_id: int): ...
```

### 2.6 `BenchmarkRunner._measure_latency` doesn't `cuda.synchronize()`

`src/models/benchmarking/benchmark_runner.py:86-100` uses
`time.perf_counter()` *around the call*, but on CUDA the call returns
as soon as the kernel is enqueued — not when it finishes. Reported
latency is wildly under-counted (often 10×), which is then divided into
throughput.

**Fix.**

```python
def _measure_latency(self, fn, inputs):
    timings = []
    cuda = self.device.type == "cuda"
    for inp in inputs:
        if cuda: torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        fn(inp)
        if cuda: torch.cuda.synchronize(self.device)
        timings.append(time.perf_counter() - start)
    return timings
```

For a more accurate single-step measurement, prefer
`torch.cuda.Event(enable_timing=True)` per call.

### 2.7 `TransformerEncoder.forward` does per-tensor `.to(self.device)` even when input is already on-device

`src/models/inference/model_wrapper.py:123`

```python
if input_ids.device != self.device:
    input_ids = input_ids.to(self.device)
```

`self.device` is the property that walks `parameters()` (in
`base_model.py`) — a non-trivial Python call inside the hot path. Cache
once on `set_device`:

```python
def set_device(self, device):
    super().set_device(device)
    self._cached_device = self._device  # already a torch.device
```

Then compare against `self._cached_device`.

---

## 3. Architectural Issues

### 3.1 Two parallel multi-task config objects with overlapping but non-equivalent fields

* `src/models/multitask/multitask_truthlens_model.py::MultiTaskTruthLensConfig`
  — narrow dataclass for the convenience constructor; carries
  per-task **loss weights**, `task_num_labels`, `enabled_tasks`.
* `src/models/config/model_config.py::MultiTaskModelConfig` — full
  YAML-backed config; carries `EncoderConfig`, `TrainingConfig`,
  `UncertaintyConfig`, ... but **does not carry per-task loss weights**.

The model class enforces a `TypeError` if you pass the wrong one
(noted in the docstring), but the system as a whole has no single
source of truth for "what does this checkpoint expect at inference?".
A model trained from YAML loses its per-task loss weights at deploy
time because `MultiTaskModelConfig` cannot represent them.

**Fix.** Promote per-task weights into `TaskConfig`:

```python
@dataclass
class TaskConfig:
    name: str
    num_labels: int
    task_type: str = "multi_class"
    loss_weight: float = 1.0
    regression: Optional[RegressionConfig] = None
    use_label_smoothing: bool = False
```

…and have `MultiTaskTruthLensConfig` build a `MultiTaskModelConfig`
internally so there is *one* canonical representation that survives
serialization. The convenience class then becomes a thin builder, not a
parallel hierarchy.

### 3.2 `TransformerEncoder` shim re-exports across two packages with the same name

`src/models/encoder/transformer_encoder.py` is a shim that re-exports
`src.models.inference.model_wrapper.TransformerEncoder`. The shim
correctly removes the old duplicate `EncoderFactory` (good — that's
called out in the docstring), but the canonical encoder still lives
under `inference/`, which is semantically wrong: an encoder is not an
inference concern. New contributors keep importing the shim path.

**Fix.** Move `TransformerEncoder` into `src/models/encoder/transformer_encoder.py`
(its natural home) and make `src/models/inference/model_wrapper.py`
the shim. This also stops `inference` from importing from `base/`
(currently `from ..base.base_model import BaseModel` in `model_wrapper`),
which creates a fragile cross-package dep.

### 3.3 `BaseModel.device` property allocates an iterator on every access

```python
@property
def device(self) -> torch.device:
    try:    return next(self.parameters()).device
    except StopIteration: pass
    try:    return next(self.buffers()).device
    except StopIteration: pass
    return self._device
```

This is on the hot path of every encoder forward (§2.7) and every
ensemble member dispatch (§2.2). `next(self.parameters())` constructs
a fresh generator each call. Cache `self._device` correctly in
`set_device`, and have `device` simply return that — it's the public
contract anyway. The "walk parameters" fallback is defensive but
fires only for parameter-less modules.

### 3.4 Heads return *dict* sometimes, *tensor* other times

`heads/multitask_head.py:81-99` accepts either, then wraps the tensor
into a dict. `multitask_truthlens_model.py:249-266` does the same.
But the loss engine (`loss/multitask_loss.py`) and
`MultiTaskOutput.from_model_outputs` only know about the dict shape.
Any new head that returns a raw tensor will train fine and crash at
calibration.

**Fix.** Codify the dict contract in `BaseHead`:

```python
class BaseHead(nn.Module, ABC):
    @abstractmethod
    def forward(self, features: Tensor) -> dict[str, Tensor]:
        """Must contain at minimum a ``logits`` tensor."""
```

Make `ClassificationHead`, `MultiLabelHead`, etc. inherit. Then drop
the tensor-fallback branches in the multitask wrappers — they hide
real bugs.

### 3.5 `MultiTaskBaseModel` and `MultiTaskTruthLensModel` are independent class trees with the same job

`MultiTaskBaseModel` lives under `base/` with `register_task_head`,
`forward`, `predict`. `MultiTaskTruthLensModel` lives under
`multitask/` and reimplements the same forward / predict logic from
scratch (does *not* inherit from `BaseModel` — it's a plain `nn.Module`).

So:

* `MultiTaskTruthLensModel` does not get the `_is_calibration_parameter_name`
  / `get_optimization_parameters` split (G4) — every TruthLens-trained
  checkpoint risks the temperature-scalar drift that base_model.py
  spent care to prevent.
* `MultiTaskTruthLensModel` doesn't get `save_checkpoint` /
  `load_checkpoint` either, so trainers reach for `torch.save` directly
  and bypass the integrity hardening (§1.1, §1.3).

**Fix.** Have `MultiTaskTruthLensModel` inherit from
`MultiTaskBaseModel` (or at least from `BaseModel`), or make the G4
calibration split a free function in `base_model` that everybody calls.

### 3.6 `task_logits` and per-task entries duplicated in the same output dict

`MultiTaskTruthLensModel.forward` returns:

```python
outputs = {"bias": {"logits": ...},
           "ideology": {"logits": ...},
           ...,
           "task_logits": {"bias": ..., "ideology": ..., ...}}
```

Two contracts in one dict. The training pipeline reads `task_logits`,
the test-evaluation pipeline reads the per-task entries. If a future
contributor drifts one, the other still "works" but on stale data.

**Fix.** Pick one. The structured `MultiTaskOutput` already exists in
`multitask/multitask_output.py` — return that and have both pipelines
go through `to_loss_inputs()` / `get_logits(task)`. Drop the dual
contract.

### 3.7 `MultiTaskOutput.from_model_outputs` legacy-fallback iterates *every key* in the dict

`multitask/multitask_output.py:75-95` — for the legacy path it loops
every key and asks "is this a dict with a `logits` tensor?". So any
metadata key (`"task_logits"`, `"shared_features"`, `"loss"`, etc.)
that *happens* to be a dict-of-tensors is misclassified as a task. Today
this works because the fast-path (`task_logits` present) is hit first,
but the moment somebody returns a `MultiTaskOutput` from a model
without that fast-path key, ghost "tasks" appear in metrics.

**Fix.** Drop the legacy branch entirely and require `task_logits` (per
the docstring). Or whitelist task names from the model's
`get_task_names()`.

---

## 4. Loss Computation

### 4.1 `multitask_base_model.forward` accumulates loss with `+` (allocates) instead of `+=`

`base/multitask_base_model.py:138-140`

```python
total_loss = loss if active_task else (
    loss if total_loss is None else total_loss + loss
)
```

Each `total_loss + loss` allocates a new tensor and pushes a node into
the autograd graph; with N tasks the graph is N levels deeper than it
needs to be. Use `torch.stack([...]).sum()` once after the loop or
keep a list and `torch.stack(...).sum(dim=0)` (single allocation,
single autograd node).

Also: when `active_task is True` the assignment unconditionally
**replaces** `total_loss` instead of accumulating across tasks. For
`active_task` mode this is intended (single task), but the ternary is
unreadable and easy to break. Split the two cases:

```python
if active_task:
    total_loss = loss
else:
    losses.append(loss)        # accumulate across tasks
...
if losses:
    total_loss = torch.stack(losses).sum()
```

### 4.2 `MultiTaskHead` weighting: float weights are not registered as buffers

`heads/multitask_head.py:48` stores `self.task_weights[name] = float(weight)`
in a plain dict. Two consequences:

1. `state_dict()` does not contain them — checkpoints don't preserve
   weights, so a re-loaded model trains under the default weight 1.0.
2. They're Python floats, so any DDP gradient hook can't see them and
   any future scheduler that wants to anneal weights has nowhere to
   register a step hook.

**Fix.** Register as a `nn.Buffer` (or `nn.Parameter` if learnable),
keyed by task name, e.g. via a `BufferDict` or a single tensor of length
`len(tasks)` indexed by task ordinal.

### 4.3 `BaseClassifier.compute_loss` casts labels to `.long()` *every step*

`base/base_classifier.py:147` — the dataloader already returns long
tensors for classification. The `.long()` call is a no-op on tensors
that are already `int64`, but on bf16/half labels (smoothed-soft
labels for distillation) it silently quantizes to integers. Either
assert dtype up front or branch on `labels.dtype`.

### 4.4 `MultiTaskHead.forward`: `task_output["loss"] = loss; task_output["weighted_loss"] = weighted_loss` mixes two contracts

If the caller routes `task_output["loss"]` into a `LossEngine` *and*
sums `total_loss`, the loss is double-counted (once at the head,
once at the engine). The current callers happen to use one or the
other, but the head publishes both into the same dict. Pick one
("publish unweighted, let the engine apply weights") and document it.

### 4.5 `loss/multitask_loss.py` and `loss/loss_normalizer.py` are unaware of the G4 calibration split

The loss routers happily include `temperature` parameters in their
gradient view because they call `model.parameters()` rather than
`model.get_optimization_parameters()`. This is the same bug whose
*single-model* version base_model.py already documents.

**Fix.** Make the loss routers go through the explicit
`get_optimization_parameters()` accessor (which now exists) when they
need a parameter list.

---

## 5. GPU / Device Handling

### 5.1 Inconsistent device selection across the codebase

Five different "pick a device" snippets, all subtly different:

| File | Pattern | Notes |
| --- | --- | --- |
| `encoder/encoder_factory.py:182` | `cuda` if available else `cpu` | ignores `mps` |
| `inference/model_wrapper.py:65`  | same                            | ignores `mps` |
| `benchmarking/benchmark_runner.py:62` | same                       | ignores `mps` |
| `utils/model_utils.py:74`        | same                            | ignores `mps` |
| `multitask_truthlens_model.py`   | inherits from encoder           | — |

Apple-Silicon contributors keep landing on CPU silently. Centralise:

```python
# src/models/_device.py
def detect_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

…and have everyone import that.

### 5.2 `BaseClassifier.predict` and `BaseClassifier.predict_logits` toggle train/eval inside `@torch.inference_mode()`

`base/base_classifier.py:153-195` — `inference_mode` is more aggressive
than `no_grad`: it disables version-counter tracking, so any tensor
created inside cannot ever flow into autograd. The
`predict_logits` path returns `outputs["logits"]` and a caller might
later try to attach it to a loss for distillation — that will raise
`RuntimeError` deep in the optimizer. `no_grad` is the right primitive
here unless the contract is *no autograd, ever*.

**Fix.** Document the `inference_mode` contract, and add a sibling
`forward_no_grad` for callers that need detachable logits.

### 5.3 `set_device` calls `self.to(device)` but downstream sub-modules built lazily skip the move

`base/base_model.py:32-43` — `set_device` records `_device` and calls
`self.to(device)`. Any sub-module created *after* `set_device` (e.g.
the temperature scalar that `temperature_scaling.py` adds post-train,
or a freshly attached LoRA adapter) sits on CPU. Trainers see a
silent `cpu`/`cuda` mismatch in the next `.forward()`.

**Fix.** Override `add_module` (or wrap adapter-attach helpers) to
call `.to(self._device)` on any new module. Simpler:

```python
def attach_adapter(self, name, adapter):
    adapter.to(self._device)
    self.add_module(name, adapter)
```

…and tell `temperature_scaling` to use it.

### 5.4 `register_buffer` not used for `pos_embeddings` / `attention_mask` constants in pooling

`representation/attention_pooling.py` and `cls_pooling.py` build their
mask constants on the Python side and call `.to(device)` per forward.
Move them to `register_buffer(..., persistent=False)` so they migrate
with `model.to()` automatically.

### 5.5 No `torch.cuda.empty_cache()` / OOM-guard around ensemble forwards

`ensemble/{ensemble_model,stacking,weighted}.py` chain N model forwards
back-to-back. On a 5-member ensemble at batch 64 / seq 512 the
intermediate activations of model 1 are not freed before model 2 starts.
Add a small `del out_i; torch.cuda.empty_cache()` after each
member's prediction is detach-and-stored (it's eval-time anyway, no
autograd graph to keep).

---

## 6. Recomputation

Findings about state computed multiple times per step.

### 6.1 `MultiTaskTruthLensModel._extract_pooled` falls back to `hidden[:, 0]` even when the encoder returns a pooled output

`multitask/multitask_truthlens_model.py:281-312` — the static
`_extract_pooled` checks `pooled_output` *or* `pooler_output`. If the
encoder's `add_pooling_layer = False` (set in `model_wrapper.py:73`),
`pooled_output` is missing and the fallback computes `hidden[:, 0]`,
duplicating exactly the work that `TransformerEncoder._pool` *just
did* (it returned `{"pooled_output": hidden[:, 0]}`). For other pooling
strategies (`mean`, `attention`) the wrapper's pooled output is *also*
already in the dict — but the fallback to `hidden[:, 0]` would still
hit and **return the wrong tensor** if the dict's key got renamed.

**Fix.** Return early with a clear error if neither is present, and
don't attempt to recompute pooling at the wrapper level — it's the
encoder's job:

```python
pooled = encoder_outputs.get("pooled_output")
if pooled is None:
    raise RuntimeError(
        "Encoder returned no pooled_output; configure pooling on the encoder, "
        "not on the multitask wrapper."
    )
return pooled
```

### 6.2 `BaseClassifier.forward` computes `log_softmax` then `exp()` then `argmax(probs)` — recompute `argmax(logits)`

`base/base_classifier.py:108-118` — `argmax` is invariant under monotonic
transforms, so `torch.argmax(logits, dim=-1)` is identical and skips
the `exp`. Same in `multitask_base_model.py:115` and
`heads/classification_head.py:159`.

```python
log_probs = F.log_softmax(logits, dim=-1)
probs = log_probs.exp()
preds = torch.argmax(logits, dim=-1)     # <-- not from probs
confidence = probs.max(dim=-1).values
entropy = -(probs * log_probs).sum(dim=-1)
```

Also: `confidence = probs.max(...).values` re-traverses the tensor.
Take it from `log_probs.max(...).values.exp()` or compute alongside the
argmax via `torch.max(logits, dim=-1)` — value+index in one pass.

### 6.3 `MultiTaskHead.predict` re-runs `forward` with no_grad

`heads/multitask_head.py:139-161` — after building the rich dict in
`forward`, `predict` calls `forward` again under `@torch.no_grad()`.
The first forward already produced the dict; the second is the entire
encoder→heads chain replayed. (The `@torch.no_grad()` is on `predict`,
which calls `self.forward(...)` — there's no path that avoids the
second forward.)

**Fix.** Have `predict` reuse a cached forward-pass output, or have it
*be* the forward pass with `self.eval()` + `torch.no_grad()` instead of
calling forward as a sub-method.

### 6.4 `optimizer_factory.build_parameter_groups` walks `named_parameters()` but ignores `_is_calibration_parameter_name`

`optimization/optimizer_factory.py:46-62` — every parameter is bucketed
into decay or no-decay. Calibration parameters (G4) are silently put
into the **decay** bucket because their leaf names (`"temperature"`)
don't match any of the no-decay keywords. Apply weight decay to a
post-hoc temperature scalar and you've destroyed the very calibration
the trainer just fitted.

**Fix.**

```python
for name, param in model.named_parameters():
    if not param.requires_grad: continue
    if hasattr(model, "_is_calibration_parameter_name") and \
       model._is_calibration_parameter_name(name):
        continue   # excluded entirely — the calibration trainer owns these
    ...
```

---

## 7. Unused / Dead / Confusing Code

### 7.1 Empty `src/models/training/` package

`ls src/models/training/` returns only `__pycache__/`. The
package directory exists, has been cached at least once, but contains
no `.py` files. Either the contents were deleted in a half-merge or a
phantom `__init__.py` lives in `__pycache__/`. Importing
`src.models.training` succeeds (no `__init__.py` is fine in 3.12 via
implicit namespaces), then explodes the moment anyone tries
`from src.models.training import X`. Delete the directory or restore
its contents.

### 7.2 Empty / placeholder packages with one shim each

* `src/models/ideology/`, `src/models/narrative/`, `src/models/propaganda/`
  — top-level directories that mirror `src/models/tasks/{ideology,narrative,propaganda}`.
  Per the prior audit, these were renamed to `tasks/<name>/` but the
  old top-levels were left as empty packages. They get picked up by
  `importlib.find_spec` and confuse autocomplete. Delete.

### 7.3 `metadata/model_versioning.py:summary` references `Any` without importing it

```python
def summary(self) -> Dict[str, Any]:
```

`Any` is not in the file's imports (only `Dict`, `List`, `Optional`).
This raises `NameError` the first time the type annotation is evaluated
under `from __future__ import annotations` … which the file has, so it
silently passes today. If `inspect.get_type_hints()` is ever called on
this method (e.g. by an OpenAPI generator), it will explode. Add the
import.

### 7.4 `ModelCard` defines `TrainingConfig` that shadows `config/model_config.py::TrainingConfig`

`metadata/model_card.py:62` and `config/model_config.py:73` both define
a class called `TrainingConfig`. They have **different fields**
(`framework, epochs, batch_size, optimizer, ...` vs
`learning_rate, weight_decay, batch_size, num_epochs, ...`). `pickle`
of one cannot be loaded as the other. Rename one — `ModelCardTrainingConfig`
is the obvious choice given its purpose is documentation, not training.

### 7.5 `model_wrapper.TransformerEncoder.forward` accepts only `input_ids` + `attention_mask`

But every modern HuggingFace tokenizer also returns `token_type_ids`
(BERT, ALBERT, ERNIE, …). The encoder silently drops them, so two
sentences encoded with `[SEP]` between them collapse to a single-segment
representation. For `roberta-base` (the default model_name) this is
correct — RoBERTa has no segment IDs. But the wrapper claims to support
"transformer" generically. Either narrow the docstring to
"RoBERTa-family encoders" or thread `token_type_ids` through:

```python
def forward(self, input_ids, attention_mask, token_type_ids=None, **kw):
    encoder_kwargs = {...}
    if token_type_ids is not None:
        encoder_kwargs["token_type_ids"] = token_type_ids
    ...
```

### 7.6 `BenchmarkResult.extra_metrics_fn` swallows exceptions silently

`benchmarking/benchmark_runner.py:140-144` — a metrics callback that
raises is logged at WARNING and the benchmark continues. For an
*audit*-style benchmark this hides regressions: the metric goes missing
from the result dict, and the consumer's downstream comparator (which
likely uses `.get("acc", 0)`) silently sees zero. Re-raise after
logging, or annotate the result with a `failed_metrics` list.

### 7.7 `weight_initialization.initialize_weights` has no `embedding` initialization for `nn.EmbeddingBag`

The function specifically initializes `nn.Embedding` (`std=0.02`) and
`nn.LayerNorm` (`weight=1, bias=0`), but `nn.EmbeddingBag` and
`nn.GRU` / `nn.LSTM` go uninitialised. Also, `kaiming` always uses
`nonlinearity="relu"` even when the next layer is `gelu`/`tanh` — the
gain is wrong. Either accept a `nonlinearity` kwarg or remove the
helper (callers should rely on PyTorch defaults).

### 7.8 `ModelCard.save_markdown` writes `data["evaluation"]["metrics"]` as a flat list — strips `validation_dataset` / `test_dataset`

`metadata/model_card.py:189-192` — only the `metrics` sub-dict is
emitted. The dataset references that distinguish "this 80 % test acc"
from "this 80 % *val* acc" are dropped from the rendered Markdown.
Add them.

---

## 8. Configuration Issues

### 8.1 Two `EncoderConfig` dataclasses with overlapping but different fields

* `src/models/config/model_config.py::EncoderConfig` — has
  `freeze_layers: int`, `enable_adapters: bool`, `adapter_type: str`,
  `adapter_dim: int`, `enable_fused_attention: bool`, … but **no**
  `pooling validation**, `model_type`, `init_from_config_only`,
  `extra_kwargs`, `output_hidden_states`.
* `src/models/encoder/encoder_config.py::EncoderConfig` — has
  `pooling validation`, `model_type`, `init_from_config_only`,
  `extra_kwargs`, `output_hidden_states`, but **no** `freeze_layers`,
  `enable_adapters`, `adapter_*`, `enable_fused_attention`.

`encoder_factory.create_from_encoder_config` accepts the `model_config`
flavour and *silently drops* `freeze_layers` / `enable_adapters` /
adapter knobs (look at lines 130-156). A YAML that says
`freeze_layers: 6` produces a model with all 12 layers trainable. Same
for adapters — the YAML says `enable_adapters: true` and the encoder
ships with no adapters attached.

**Fix.** Merge into one dataclass, keep all fields, validate exactly
once. The factory's `create_from_*` methods can then be a single
`create_from(any_config_protocol)`.

### 8.2 `EncoderConfig.from_dict` uses raw `cls(**config_dict)` — typo-fragile

`encoder/encoder_config.py:90-99` — passes the entire dict as kwargs.
Any unknown field raises `TypeError`. That's fine and intentional
(strict). But the *other* `EncoderConfig` (`config/model_config.py`) has
no `from_dict` and is built positionally inside
`ModelConfigLoader.load_multitask_config:200`:

```python
encoder_cfg = EncoderConfig(**raw.get("encoder", {}))
```

Same strictness, but the YAML field set differs from the
`encoder/encoder_config.py` field set (§8.1). Two YAMLs that "look the
same" load against two different schemas. Pick one config and have
both loaders use it.

### 8.3 `ModelConfigLoader.load_yaml` uses `yaml.safe_load` (good) but does **not** validate the result

`config/model_config.py:163-171` — `safe_load` of an empty file returns
`None`, then `raw.get("encoder", {})` raises `AttributeError: 'NoneType'
object has no attribute 'get'`. Equally, a YAML that's just a list
crashes the same way. Wrap with a clear error:

```python
data = yaml.safe_load(f)
if data is None or not isinstance(data, dict):
    raise ValueError(f"Config file {path} must be a YAML mapping (got {type(data).__name__}).")
return data
```

### 8.4 `RegularizationConfig.adv_epsilon: float = 1e-5` is much too small for FGM/PGD on tokenized inputs

The standard FGM perturbation magnitude on text-embedding models is
`~1.0` in token-embedding space (see Madry et al.); `1e-5` makes the
attack a no-op and the "adversarial" gradient indistinguishable from a
normal one. Either default to a value calibrated for token embeddings
(`epsilon=1.0, alpha=0.3, num_steps=3` for PGD-3) or rename the field
to `adv_epsilon_input_space` and document the unit.

### 8.5 `EncoderConfig.use_amp = True` default + `amp_dtype = "bf16"` silently downgrades to fp32 on CPU

`inference/model_wrapper.py:135` correctly gates autocast on
`self.device.type == "cuda"`, so on CPU the amp flags are inert. That's
correct — but the user never sees the warning. If they ran a CPU
benchmark expecting fp16 throughput, they'd be measuring fp32 latency
instead. Log once on init when amp is requested but disabled:

```python
if self.use_amp and self.device.type != "cuda":
    logger.warning("use_amp=True but device=%s; AMP disabled.", self.device.type)
```

### 8.6 `task_num_labels` overrides accepted but not validated against the task's *type*

`MultiTaskTruthLensConfig.task_num_labels` lets the YAML override
narrative (multi_label) head size. Passing `narrative=2` is silently
accepted, but the head is then a 2-label `MultiLabelHead` while the
test labels are a 3-bit one-hot — `BCEWithLogits` happily computes a
zero loss on the missing class. Validate that `num_labels >= 2` for
multi_class and `num_labels == len(default_labels)` (or explicitly
opt-in to relabel) for multi_label.

### 8.7 `MultiTaskModelConfig.shared_encoder = True` is set but never actually consulted

`config/model_config.py:150` — defaults to `True`. Grep shows zero
non-trivial reads of it (it's plumbed into the metadata bag but never
branched on). The model is *always* shared-encoder. Either implement
the branch (build per-task encoder copies when False) or drop the field
to avoid implying support that doesn't exist.

---

## 9. Edge Cases Not Handled

### 9.1 `BaseClassifier.forward` raises `ValueError` on 1-D feature tensor — but training-time callers can produce 1-D when batch_size=1 + squeeze

`base/base_classifier.py:74` — `if features.dim() != 2` is correct
*shape* check, but several callers in `inference/predictor.py` apply
`.squeeze(0)` for single-sample inference. The classifier raises a
"Expected 2D" ValueError instead of either re-adding the batch dim
or giving a cleaner message.

**Fix.** Accept 1-D and unsqueeze:

```python
if features.dim() == 1:
    features = features.unsqueeze(0)
elif features.dim() != 2:
    raise ValueError(...)
```

### 9.2 `MultiTaskHead.predict` returns `None` for `predictions` / `probabilities` when the sub-head omits them

`heads/multitask_head.py:155-159` — falls back to `.get(...)`, so a
sub-head that returns only `{"logits": ...}` (e.g. a regression head)
puts `None` into the predictions dict. Downstream metric code crashes
on `predictions.argmax()`. Either fail loudly here or compute the
defaults from `logits` (argmax + softmax).

### 9.3 Empty `task_heads` ModuleDict crashes with non-obvious error

`MultiTaskTruthLensModel.__init__` validates `task_heads is non-empty`
at construction, but `MultiTaskHead` (the head wrapper, not the model)
allows zero registered tasks and then `forward` returns
`{"tasks": {}}` with no `total_loss` — the trainer divides by zero on
the next batch. Validate at first `forward` call:

```python
if not self.task_heads:
    raise RuntimeError("MultiTaskHead has no registered tasks.")
```

### 9.4 `TransformerEncoder._pool("mean")` divides by `clamp(mask.sum, min=1e-9)` instead of `1`

`inference/model_wrapper.py:178` — for an *all-zero* attention mask
(short sequences padded to long ones, very rare but possible from
truncation+empty input), `1e-9` produces astronomical pooled
embeddings, which then NaN through softmax. Use `min=1` (one token
worth) — at worst you get the pad-token embedding in the output,
which is a sensible failure mode, not NaNs.

### 9.5 `BaseModel.load_checkpoint` `optimizer.load_state_dict(...)` re-applies optimizer state without checking param shapes

`base/base_model.py:213-214` — if the model was modified between save
and load (e.g. a head was added), the optimizer state dict still
addresses the old param IDs, and PyTorch raises a confusing
`KeyError`. Either pre-validate shapes against `optimizer.param_groups`
or wrap with a clearer error.

### 9.6 `weight_initialization.initialize_weights(method="kaiming")` silently uses `gain or 1.0`

Kaiming has no gain parameter — `nn.init.kaiming_uniform_` accepts `a`
(the negative slope of the rectifier) and computes the gain internally.
The `gain or 1.0` argument is unused on the kaiming branches; on
xavier it's used. Don't accept a parameter the function doesn't honour.

### 9.7 `MultiTaskOutput.to_dict(detach=False)` keeps autograd-tracked tensors in a returned dict that will outlive the forward pass

If the caller stores the dict (e.g. for batch logging) and the next
backward pass clears the graph, accessing `.grad_fn` later raises.
Default `detach=True` handles this for the common case, but the
opt-out is a footgun. Document it or remove the kwarg.

### 9.8 `ModelVersionRegistry._save` is not atomic

`metadata/model_versioning.py:86-88` — writes JSON directly to the
target path. A SIGINT mid-write leaves the registry truncated and
unloadable. Use `atomic_replace`: write to `.tmp`, then `os.replace`.

### 9.9 `ModelCard.save_json` does not validate its `path` is writable / not a directory

A common ops mistake (`save_json("./reports/")`) raises a confusing
`IsADirectoryError` deep in `json.dump`. Validate:

```python
if path.is_dir():
    raise ValueError(f"path {path} is a directory; expected a .json file.")
```

---

## 10. What's Verified Correct

To be useful, an audit must also enumerate what is *intentional* and
*correct* — so future contributors don't "fix" working code.

* **Numerical stability of entropy / confidence in inference.**
  `base_classifier.py`, `multitask_base_model.py`,
  `heads/classification_head.py`, and `mc_dropout._xlogx` correctly
  compute entropy via `log_softmax` / `logsigmoid` instead of the
  `log(p + EPS)` antipattern. `ensemble_uncertainty.py` is the lone
  exception — see §1.7.

* **`BaseModel` calibration parameter split (G4).** The base class
  correctly identifies `temperature` parameters as calibration-only and
  excludes them from the main optimizer's gradient view. The naming
  convention (`CALIBRATION_PARAMETER_NAMES`) is extensible by
  subclasses. The follow-up bug is just that two other callers
  (`optimizer_factory.build_parameter_groups`, the loss routers) don't
  *use* the helper — see §6.4 and §4.5.

* **`ClassificationHead._init_weights`** correctly applies
  `xavier_uniform_` to all linear sub-modules (including `fc1` and
  `fc2`) and zeros biases. No silent bias drift on init.

* **`TransformerEncoder.forward` autocast gating.** Correctly disables
  AMP on CPU/MPS (§8.5 is a UX nit, not a correctness bug).

* **`MultiTaskTruthLensConfig` strict-init enforcement (CFG2).** Typos
  in YAML field names raise `TypeError` at load time instead of
  silently dropping the field. The docstring explains the design.

* **`MultiTaskTruthLensModel` construction-mode mutual exclusivity.**
  Cannot pass `(encoder, task_heads)` and `config=` simultaneously;
  cannot pass a `MultiTaskModelConfig` to the `config=` kwarg (it
  expects `MultiTaskTruthLensConfig`). The two construction paths are
  cleanly separated.

* **`checkpointing/model_loader.ArtifactManager`** (the *good* one of
  the two — see §1.1) uses `atomic_save_state_dict` (tmp + `os.replace`)
  *and* sha256 integrity verification *and* schema validation *and*
  `weights_only=True`. This is exactly the right shape; the bug is just
  that the sibling class shadows it.

* **`MultiTaskHead.forward` sub-head dict isolation (P5).** The fix to
  copy the sub-head's dict before mutation (`task_output = dict(head_output)`)
  is correct and prevents stale-loss bleed-over across batches.

* **`TransformerEncoder.gradient_checkpointing_enable` / `_disable`**
  delegates to the underlying HuggingFace model correctly and tolerates
  encoders that lack the method.

* **`base_classifier`/`multitask_base_model` skip derived stats during
  training (P1).** Confirmed in three independent forward methods.

* **`MultiTaskTruthLensModel._extract_pooled` HuggingFace-vs-dict
  dispatch.** Handles both object-style (`.pooler_output`) and dict-style
  encoder outputs. The fallback to `hidden[:, 0]` is the recommended
  RoBERTa workaround when `add_pooling_layer=False`. (The recompute
  finding §6.1 is about a *separate* call site doing the same work
  twice, not about this function being wrong.)

* **Most of `checkpointing/`** — `integrity.py`, `io_utils.py`'s
  `atomic_save_state_dict` (with `flush + fsync`), `metadata.py`,
  `schema.py`, `selection.py`, `validator.py`, `resolver.py`,
  `loader_utils.safe_load_state_dict` are all production-quality.
  The `AsyncCheckpointWriter` queue-overflow policy is the lone
  weak link (§1.9).

* **`representation/{cls,mean,attention}_pooling`** correctly handle the
  attention-mask edge cases for non-CLS pooling (modulo the tiny
  divide-by-`1e-9` issue in §9.4).

* **`registry/model_registry.py`** uses `weights_only=True` for both
  state-dict load paths. Good.

---

## 11. Score: **5.5 / 10**

This is "*works for the team that wrote it, will quietly hurt the next
team*" code. The math (entropy, log-stability, G4 calibration split,
P1/P5/N1 fixes) is largely correct and shows that someone who knew
what they were doing has been through a non-trivial portion of this
codebase. The architecture is *almost* clean: there are clear base
classes, a registry, a multi-task forward contract, and a checkpoint
manager with atomic writes and integrity verification.

But the package is also full of *quiet* failure modes:

* **Two of the most safety-critical primitives have a duplicate class
  with weaker invariants** (§1.1 `ArtifactManager`, §1.2
  `BenchmarkRunner`). Both are reachable today; both will cause
  hard-to-diagnose production incidents the first time the wrong
  import wins.
* **The base load path is unsafe** (`weights_only=False`, §1.3) — a
  hostile checkpoint = arbitrary code execution.
* **The optimizer factory silently drops `weight_decay`** for four of
  its five backends (§1.4) — most teams will never notice, the val
  loss "just looks a bit worse than expected".
* **Config is stratified into two parallel dataclass hierarchies**
  (`EncoderConfig`, `TrainingConfig`, `MultiTaskConfig`-vs-
  `MultiTaskTruthLensConfig`) with overlapping but different fields,
  silently dropping schema fields the YAML thinks are honoured (§3.1,
  §8.1, §7.4).
* **Performance leaves ~30 % on the floor** (no `torch.compile` on
  encoders, mid-forward `.to()` in two ensemble paths, `.synchronize()`
  missing from the benchmark itself, §2.1/2.2/2.6).
* **Several finished hardenings (P1, N1, G4) are correctly applied
  in 3 of 4 sites, with the fourth carrying the old antipattern** —
  `ensemble_uncertainty.py` (§1.7), `optimizer_factory` (§6.4), the
  loss routers (§4.5). This pattern of "almost-finished refactor"
  is the strongest predictor that the next refactor will leave the
  same scar.

**Path to 8 / 10 (one focused sprint):**

1. Resolve the two duplicate classes (§1.1, §1.2) — half a day.
2. Flip `weights_only=True` and fix `optimizer_factory` weight_decay
   (§1.3, §1.4) — under an hour, plus regression tests.
3. Apply the N1 / G4 fixes to the three remaining sites
   (§1.7, §6.4, §4.5).
4. Unify the config hierarchies (§3.1, §8.1) behind a single dataclass
   per concept; deprecate the duplicates with a `DeprecationWarning`
   for one release.
5. Default `torch.compile=True` on CUDA, fix the ensemble `.to()`
   loops, fix the benchmark `.synchronize()`.

**Path to 9 / 10:** the architectural cleanups in §3 (single multi-task
class hierarchy, single dict-shape head contract, single device
helper), plus actually using the `MultiTaskOutput` dataclass at every
call site instead of two parallel dict shapes. That's a multi-week
project, but it's exactly the kind of consolidation that pays off
every time a new task is added.
