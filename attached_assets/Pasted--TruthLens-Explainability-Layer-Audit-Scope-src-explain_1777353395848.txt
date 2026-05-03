# TruthLens Explainability-Layer Audit

Scope: `src/explainability/*` — 22 files audited (orchestrator, explainability_pipeline, common_schema, model_explainer, shap_explainer, lime_explainer, attention_rollout, attention_visualizer, bias_explainer, emotion_explainer, propaganda_explainer, explanation_aggregator, explanation_cache, explanation_calibrator, explanation_consistency, explanation_metrics, explanation_monitor, explanation_report_generator, explanation_visualizer, token_alignment, utils_validation, __init__).

Format: v9 strict — no fixes performed, report only. All findings carry file:line references and code-level fixes.

---

## 1. CRITICAL EXPLANATION BUGS

### CRIT-1 — `bias_explainer.compute_shap` / `compute_ig` / `compute_attention_rollout` assume single-head model; multitask model has no `out.logits`

- **File / Function:** `src/explainability/bias_explainer.py:68-116`.
- **Issue:** all three functions call `model(...).logits` or `model(...).logits.max().backward()`. The TruthLens model is a multitask `nn.Module` with `model.heads = ModuleDict({task: head})` (per `src/inference/analyze_article.py` and the `_integrated_gradients` path in `score_explainer.py:88-94` which uses `model.encoder(...)` then `model.heads[task](cls)`). Calling `model(**enc)` returns either a dict or raises depending on the wrapper, and `.logits` does not exist.
- **Why explanation is misleading:** the `_run("bias", ...)` wrapper in `orchestrator.py:122-125` swallows the `AttributeError` and returns `None`. The orchestrator then stores `bias_explanation=None`, and the aggregator/consistency silently skip it. The user sees "bias_explanation: null" without any indication that the entire bias path is dead.
- **Exact fix:**
  ```python
  def compute_ig(model, tokenizer, text, *, task: str = "bias", target_idx: int = 0, steps: int = 32):
      device = next(model.parameters()).device
      enc = tokenizer(text, return_tensors="pt").to(device)
      emb = model.encoder.embeddings(enc["input_ids"])
      baseline = torch.zeros_like(emb)
      alphas = torch.linspace(0, 1, steps, device=device).view(-1, 1, 1, 1)
      scaled = (baseline + alphas * (emb - baseline)).flatten(0, 1).requires_grad_(True)
      attn = enc["attention_mask"].repeat(steps, 1)
      out = model.encoder(inputs_embeds=scaled, attention_mask=attn)
      logits = model.heads[task](out.last_hidden_state[:, 0])
      logits[:, target_idx].sum().backward()
      grads = scaled.grad.view(steps, *emb.shape).mean(dim=0)
      ig = ((emb - baseline) * grads).sum(-1).detach()[0]
      return _normalize(ig.cpu().numpy())
  ```
  Apply the same multitask-aware pattern to `compute_shap` and `compute_attention_rollout`.

---

### CRIT-2 — `emotion_explainer.fuse(lexicon, gradients)` mixes word-level and subword-level vectors of different lengths

- **File / Function:** `src/explainability/emotion_explainer.py:91-128, 175-205`.
- **Issue:**
  - `tokens = tokenize(text)` → word-level regex `\b[a-z]+\b` (length = N_words).
  - `gradients = compute_gradients(...)` → subword-level from `tokenizer(text)` (length = N_subwords ≠ N_words).
  - `fuse(lexicon, gradients)` → `0.6 * normalize(lexicon) + 0.4 * normalize(gradients)`. NumPy will either **raise `ValueError`** for non-broadcastable shapes or, if N_words happens to equal N_subwords, **silently misalign** every position.
- **Why explanation is misleading:** the per-token scores are claimed to align with `tokens` (word level), but 40 % of the value comes from arbitrarily-indexed subword gradients. Token-position semantics are destroyed.
- **Exact fix:** project gradients to word level using offset alignment from `tokenizer(text, return_offsets_mapping=True)`:
  ```python
  enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
  offsets = enc.pop("offset_mapping")[0].tolist()
  emb = model.encoder.embeddings(enc["input_ids"]).requires_grad_(True)
  ...                                             # backward
  subword_imp = emb.grad.abs().sum(-1)[0].cpu().numpy()
  word_imp = align_subwords_to_words(subword_imp, offsets, text, word_tokens=tokens)
  return word_imp
  ```
  Or — simpler and faster — drop the gradient term entirely and use the lexicon plus a `model.predict_proba`-derived global emotion confidence multiplier.

---

### CRIT-3 — `explanation_aggregator.aggregate` sorts tokens alphabetically, destroying positional order

- **File / Function:** `src/explainability/explanation_aggregator.py:110-113`.
- **Issue:** `tokens = sorted(set().union(*[set(s.keys()) for s in sources.values()]))`. The output `AggregatedExplanation.tokens` is now alphabetically sorted, with `final_token_importance[i]` aligned to the sorted token at position `i`, **not** to the original text position.
- **Why explanation is misleading:** the report generator and visualizer (`explanation_visualizer.plot_token_heatmap`, `explanation_report_generator._highlight_tokens`) render tokens in the order they receive them. Users see an alphabetic strip of tokens, not the original sentence. The downstream `explanation_metrics.faithfulness` then constructs `" ".join(tokens)` (`explanation_metrics.py:60, 62-65`) producing a string in alphabetical order — a sentence the model has never seen — and computes faithfulness on **that**.
- **Exact fix:** preserve order from the first source whose tokens are in input order (which is all explainers except SHAP `_process_shap_values`, which already returns in input order). Use a stable accumulator:
  ```python
  ordered_tokens = []
  seen = set()
  for src in sources.values():
      for tok in src.keys():
          if tok not in seen:
              ordered_tokens.append(tok); seen.add(tok)
  for tok in graph_node_importance:
      if tok not in seen:
          ordered_tokens.append(tok); seen.add(tok)
  tokens = ordered_tokens
  ```
  And add a unit test that asserts `agg.tokens == shap.tokens` when only SHAP is enabled.

---

### CRIT-4 — `explanation_aggregator.aggregate` collapses duplicate tokens via `dict(zip(...))` → loses every repeated word

- **File / Function:** `src/explainability/explanation_aggregator.py:82, 86, 90, 94`.
- **Issue:** `dict(zip(shap.tokens, shap.importance))` keeps only the **last** occurrence of each token. For an article that says "the truth is the truth", the final aggregated explanation has **one** `the` instead of three.
- **Why explanation is misleading:** propaganda relies heavily on **repetition** (encoded explicitly in `propaganda_explainer.PROPAGANDA_PATTERNS["repetition"]`). The aggregator silently kills this signal. Token-level faithfulness metrics applied to the aggregated output then operate on the wrong sequence length.
- **Exact fix:** index by `(position, token)` rather than `token`:
  ```python
  sources["shap"] = {(i, t): v for i, (t, v) in enumerate(zip(shap.tokens, shap.importance))}
  ```
  All downstream lookups and `tokens` enumeration use position as the key, with `t` carried alongside.

---

### CRIT-5 — `common_schema.ExplanationOutput.normalize_importance` validator silently re-normalizes the importance list, but `structured` keeps the un-normalized values → numerical disagreement

- **File / Function:** `src/explainability/common_schema.py:60-78` (`mode="before"` validator).
- **Issue:** every explainer calls `calibrate_explanation(...)` which already L1-normalizes scores, then constructs:
  ```python
  ExplanationOutput(
      tokens=tokens,
      importance=scores.tolist(),                                # passes through normalize_importance
      structured=[TokenImportance(token=t, importance=float(s)) for t, s in zip(tokens, scores)],
      ...
  )
  ```
  The validator runs on `importance` (re-normalizes again — `abs() / sum`) but `structured[i].importance` is the raw `s` value passed in. After construction, `out.importance[i] != out.structured[i].importance` for any non-trivial input. Worse: `TokenImportance.importance` is `Field(ge=0.0, le=1.0)` — if a calibrator returns a value > 1.0 (numerically possible when sharpening + EPS), construction raises `ValidationError` even though the L1-norm of the same value would have been < 1.
- **Why explanation is misleading:** consumers read either `out.importance` (re-normalized) or `out.structured[i].importance` (raw) and silently disagree on the magnitude of every token's contribution. The aggregator uses `shap.importance` (re-normalized), the consistency module uses `shap.structured[i].importance` (raw) — different inputs.
- **Exact fix:** make the schema a strict pass-through and let callers normalize **once**:
  ```python
  class ExplanationOutput(BaseModel):
      ...
      @field_validator("importance")
      @classmethod
      def validate_finite(cls, v):
          arr = np.asarray(v, dtype=float)
          if not np.all(np.isfinite(arr)):
              raise ValueError("importance must be finite")
          return v
      @field_validator("structured")
      @classmethod
      def validate_aligned(cls, v, info):
          tokens = info.data.get("tokens", [])
          importance = info.data.get("importance", [])
          if len(v) != len(tokens) or len(importance) != len(tokens):
              raise ValueError("tokens, importance, and structured must align")
          for s, i in zip(v, importance):
              if abs(s.importance - i) > 1e-6:
                  raise ValueError("structured importance must equal flat importance")
          return v
  ```
  Then drop `TokenImportance.Field(ge=0.0, le=1.0)` since explainers may legitimately produce signed attributions.

---

### CRIT-6 — Two `ExplainabilityResult` classes with divergent fields → silent schema confusion

- **Files:**
  - `src/explainability/common_schema.py:162-179` (with `extra="forbid"`, includes `propaganda_explanation`, `explanation_quality_score`).
  - `src/explainability/explainability_pipeline.py:28-50` (with `extra="forbid"`, **no** `propaganda_explanation`, **no** `explanation_quality_score`).
- **Issue:** `run_explainability_pipeline` constructs the **second** `ExplainabilityResult` (the local one). If the orchestrator includes `propaganda_explanation` or `explanation_quality_score` in its output dict, **the second class is built without them** — silent data loss. Conversely, if a downstream consumer imports from `common_schema`, it expects fields the pipeline never populates.
- **Why explanation is misleading:** `model_explainer.py:51-94` constructs a third hand-rolled dict (not a Pydantic model at all). Three mutually inconsistent representations of "the explanation result" coexist.
- **Exact fix:** delete `explainability_pipeline.ExplainabilityResult` and `model_explainer.py` outright. Use `common_schema.ExplainabilityResult` as the single source of truth and have `run_explainability_pipeline` and any backward-compat function build that one.

---

### CRIT-7 — `model_explainer.py` and `explainability_pipeline.py` are duplicated, divergent entry points

- **Files:** `src/explainability/model_explainer.py` (139 lines) and `src/explainability/explainability_pipeline.py` (175 lines) both define `explain_prediction_full` and `explain_fast` with **different behavior**:
  - `model_explainer.explain_prediction_full` enables `cache_enabled=True`, `use_attention_rollout=True`.
  - `explainability_pipeline.explain_prediction_full` enables `cache_enabled=False`, `use_attention_rollout=False`.
- **Issue:** depending on which file the caller imports, the same function name produces a different explanation. `__init__.py` exports neither, so callers must import explicitly — and there is no signal which is canonical.
- **Why explanation is misleading:** users running the same input twice (once via each path) get different importance vectors with no failure mode visible.
- **Exact fix:** delete `model_explainer.py`. Re-export `run_explainability_pipeline`, `explain_prediction_full`, `explain_fast` from `__init__.py`. Add a deprecation shim if any external import is detected.

---

### CRIT-8 — `orchestrator.explain` discards integrated_gradients computed inside `bias_explainer` → IG never reaches the aggregator or consistency module

- **File / Function:** `src/explainability/orchestrator.py:230-241, 263-271`.
- **Issue:**
  - The aggregator call passes `integrated_gradients=None` even though `bias.integrated_gradients` (a length-N float list) was computed.
  - The consistency call passes `integrated_gradients=None` for the same reason.
- **Why explanation is misleading:** the aggregator advertises a 0.25 weight for IG (`AggregationWeights.integrated_gradients=0.25`); since IG is never supplied, that weight is silently redistributed to the others — but `_normalize` at construction (`explanation_aggregator.py:42-50`) hardcoded the normalization at init. So the 0.25 share goes to nothing, leaving sum < 1.0 effectively (other weights still sum to 0.75 in the dict). The fusion formula `weighted_sum += val * w * c` then under-weights every token.
- **Exact fix:**
  ```python
  ig_out = None
  if isinstance(bias_out := explanation.get("bias_explanation"), dict):
      tokens_ig = bias_out.get("tokens")
      ig_vals = bias_out.get("integrated_gradients")
      if tokens_ig and ig_vals:
          ig_out = ExplanationOutput(
              method="integrated_gradients",
              tokens=tokens_ig,
              importance=ig_vals,
              structured=[TokenImportance(token=t, importance=float(v))
                          for t, v in zip(tokens_ig, ig_vals)],
          )
  ...
  agg = self.aggregator.aggregate(
      shap=shap_out, integrated_gradients=ig_out,
      attention=attention_out, lime=lime_out, graph_explanation=graph_expl,
  )
  ```

---

### CRIT-9 — `propaganda_explainer.explain_propaganda` is text-only (no model), but is presented in the same `ExplanationOutput` schema as faithful explainers

- **File / Function:** `src/explainability/propaganda_explainer.py:122-182`.
- **Issue:** the function does pure regex + lexicon scoring with no model call. It is then wrapped as `ExplanationOutput(method="propaganda", ...)` and merged into the same downstream pipeline as SHAP/LIME/IG/attention.
- **Why explanation is misleading:** consumers reading `result.shap_explanation`, `result.lime_explanation`, `result.propaganda_explanation` cannot distinguish model-faithful from heuristic-only signals. The aggregator does not distinguish either — propaganda scores influence `final_token_importance` even though the model never saw any of those tokens.
- **Exact fix:**
  - Add a `faithful: bool` flag to `ExplanationOutput`:
    ```python
    faithful: bool = True   # False for heuristic / lexicon-only
    ```
  - In the propaganda explainer, set `faithful=False`.
  - In `ExplanationAggregator.aggregate`, gate inclusion of non-faithful sources by an explicit policy:
    ```python
    if propaganda and self.weights.get("include_heuristic", False):
        sources["propaganda"] = ...
    ```

---

### CRIT-10 — `explanation_consistency._spearman` uses `np.corrcoef(np.argsort(a), np.argsort(b))` — that is **not** Spearman

- **File / Function:** `src/explainability/explanation_consistency.py:73-75`.
- **Issue:** `np.argsort(a)` returns the **indices** that would sort `a`, not the **ranks** of each element. Spearman correlation requires ranks (i.e., `scipy.stats.rankdata` or `np.argsort(np.argsort(a))`). The current implementation correlates index permutations and produces nonsense values that happen to lie in `[-1, 1]`.
- **Why explanation is misleading:** the value reported as `spearman` in `consistency_metrics` is mathematically incorrect. Consumers comparing explainer agreement see a number that has no statistical interpretation.
- **Exact fix:**
  ```python
  @staticmethod
  def _spearman(a, b):
      ra = np.argsort(np.argsort(a))   # ranks of a
      rb = np.argsort(np.argsort(b))   # ranks of b
      if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
          return 0.0
      return float(np.corrcoef(ra, rb)[0, 1])
  ```
  Or import `scipy.stats.spearmanr` and use it directly.

---

### CRIT-11 — `explanation_metrics.faithfulness` (and `comprehensiveness`/`sufficiency`/`deletion`/`insertion`) constructs ablated text by joining tokens with `" "`

- **File / Function:** `src/explainability/explanation_metrics.py:56-75, 80-89, 95-104, 110-124, 130-143`.
- **Issue:** tokens passed in are subword tokens (`Ġevery`, `Ġone`, `##the`, `<s>`, `</s>`) **or** sorted-alphabetic tokens from the aggregator (CRIT-3). `" ".join(tokens)` produces strings like `"</s> <s> Ġabandoned Ġabhorrent ..."` that the model has never seen. The model's response on that string has no relationship to its response on the original sentence.
- **Why explanation is misleading:** every faithfulness metric reports a value computed on garbage text. The "explanation_quality_score" is meaningless.
- **Exact fix:** ablate at the **input text** level using offsets, not at the token list level:
  ```python
  def faithfulness(self, text, tokens, offsets, scores, predict_fn):
      base = predict_fn([text])[0]["fake_probability"]
      ablated = []
      for (start, end) in offsets:
          ablated.append(text[:start] + "[MASK]" + text[end:])
      preds = self._extract_fake_prob_batch(predict_fn(ablated))
      deltas = base - preds
      ...
  ```
  And require the orchestrator to pass `offsets` end-to-end.

---

### CRIT-12 — `explanation_metrics.evaluate` runs **3·N + 2 forward passes per article** and orchestrator wraps with `_make_batch_predict_fn` that loops in Python

- **File / Function:** `src/explainability/explanation_metrics.py:159-192` and `src/explainability/orchestrator.py:33-38`.
- **Issue:**
  - `faithfulness`: 1 base + N ablations.
  - `comprehensiveness`: 1 base + 1 perturbed (already cached if same `text`).
  - `sufficiency`: 1 base + 1 kept.
  - `deletion_score`: N progressive deletions.
  - `insertion_score`: N progressive insertions.
  - Total ≥ **3N + 2** forwards per article.
  - Each batch call goes through `_make_batch_predict_fn` which is `[predict_fn(t) for t in texts]` — Python loop, no batching.
- **Why explanation is misleading:** for a 200-token article, that is 600+ sequential forwards just for one metrics call. A single explanation call on a moderate article takes tens of seconds; "fast mode" still triggers `lime` which adds another 256 forwards.
- **Exact fix:** propagate true batched prediction:
  ```python
  def _make_batch_predict_fn(predict_fn) -> Callable:
      batch_fn = getattr(predict_fn, "batch_predict", None)
      if callable(batch_fn):
          return batch_fn
      def _batch(texts: List[str], chunk: int = 32) -> List[Dict]:
          out = []
          for i in range(0, len(texts), chunk):
              out.extend(predict_fn(t) for t in texts[i:i+chunk])  # still per-text but at least documented
          return out
      return _batch
  ```
  And require `predict_fn` to expose `batch_predict` for production use.

---

## 2. PERFORMANCE BOTTLENECKS

### PERF-1 — Sequential, redundant model forward passes across explainers

- **Location:** `orchestrator.py:158-226`.
- **Issue:** SHAP, LIME, IG (in `bias_explainer`), attention rollout (in `bias_explainer`), gradient (in `emotion_explainer`), and `predict_fn` itself all do **independent** forward passes on the **same** text. For one article: ≥ 6 model forwards plus the `explanation_metrics` (`3N+2` more). No tensors shared.
- **Fix:** introduce a `ModelForwardCache` that caches `(input_ids, attention_mask) → (logits_dict, hidden_states, attentions, embeddings)` on the first pass and re-uses for SHAP/LIME wrapping (they need the prediction), IG (needs embeddings), attention rollout (needs attentions). For the orchestrator:
  ```python
  fwd = self._cached_forward(text)      # one model call
  shap_out = self._shap_from_cache(fwd, text) if self.config.use_shap else None
  ig_out = self._ig_from_cache(fwd, model, tokenizer, task="bias") if self.config.use_bias_emotion else None
  attn_out = self._rollout_from_cache(fwd, tokens) if self.config.use_attention_rollout else None
  ```
- **Expected speedup:** **5–10×** end-to-end per article.

---

### PERF-2 — `lime_explainer.num_samples=256` default with per-sample non-batched `predict_fn`

- **Location:** `lime_explainer.py:152-175` and `lime_predict_wrapper:88-125`.
- **Issue:** LIME generates 256 perturbation samples by default; `lime_predict_wrapper` tries `predict_fn.batch_predict` first but falls back to a Python `for t in text_list: predict_fn(t)`. If the wrapped `predict_fn` lacks `batch_predict`, LIME does 256 sequential forwards.
- **Fix:** require batched prediction at the orchestrator boundary (PERF-1 fix); reduce LIME default `num_samples` to 64 with explicit override; add `batch_size=32` chunking to `_batched_predict` (already present, but only kicks in if outer wrapper batched first).
- **Expected speedup:** **8–16×** on the LIME path.

---

### PERF-3 — `bias_explainer` builds a fresh `shap.Explainer` on every call

- **Location:** `bias_explainer.py:79-80`.
- **Issue:** `shap.Explainer(predict, tokenizer)` is constructed inside `compute_shap` per article — no caching like `shap_explainer.get_explainer`. SHAP explainer construction is expensive (builds the `Text` masker tokenizer state).
- **Fix:** use the cached `get_explainer` from `shap_explainer.py`, or maintain a module-level `OrderedDict` mirror inside `bias_explainer.py`.
- **Expected speedup:** **2-3×** on the bias path.

---

### PERF-4 — `attention_rollout._add_residual` uses `clamp_min(EPS)` then `attention.sum(dim=-1, keepdim=True)` per layer per call → many small ops

- **Location:** `attention_rollout.py:86-91, 130`.
- **Issue:** for each of L layers, we do `+ identity`, `.sum(...)`, `clamp_min(EPS)`, `/`. Then `multi_dot(processed[::-1])` is a cascade of L matrix multiplies of `(seq_len, seq_len)` matrices. For seq_len=512, L=12: 12 × 512² = 3.1M ops per residual normalization, 11 × 512³ = 1.5B ops in `multi_dot` — substantial on CPU if `attentions[i].device == "cpu"`.
- **Fix:** stack first, then vectorize:
  ```python
  attn = torch.stack([self._aggregate_heads(a, sample_index).float()
                      for a in attentions])           # [L, S, S]
  identity = torch.eye(attn.shape[-1], device=attn.device).expand_as(attn)
  attn = attn + identity
  attn = attn / attn.sum(-1, keepdim=True).clamp_min(EPS)
  rollout = attn[0]
  for i in range(1, attn.shape[0]):
      rollout = attn[i] @ rollout
  ```
  Or use `torch.linalg.multi_dot(list(attn.flip(0)))` once.
- **Expected speedup:** **2-4×** on rollout.

---

### PERF-5 — `explanation_aggregator.aggregate` per-token Python loop with dict lookups

- **Location:** `explanation_aggregator.py:142-174`.
- **Issue:** for N tokens × 4 explainers, each iteration does dict lookup, multiplication, list append. For a 500-token article, this is ~2k Python operations.
- **Fix:** vectorize via numpy. Build a `[n_methods, n_tokens]` matrix indexed by token position once, then `final_scores = (W * C * importance_matrix).sum(0)`:
  ```python
  imp = np.zeros((len(method_names), len(tokens)), dtype=np.float32)
  for mi, name in enumerate(method_names):
      imp[mi] = [sources[name].get(t, 0.0) for t in tokens]
  weights = np.array([self.weights[m] * confidences[m] for m in method_names])
  final = (weights[:, None] * imp).sum(0)
  ```
- **Expected speedup:** **3-5×** on aggregation.

---

### PERF-6 — `orchestrator` instantiated on every `run_explainability_pipeline` call

- **Location:** `explainability_pipeline.py:71-73` and `model_explainer.py:46`.
- **Issue:** `orchestrator = ExplainabilityOrchestrator(config=config)` constructs `ExplanationCache`, `AttentionRollout`, `ExplanationAggregator`, `ExplanationConsistency`, `ExplanationMetrics`, `ExplanationMonitor`, `GraphExplainer` on every call. `GraphExplainer.__init__` itself loads spaCy/regex tables.
- **Fix:** keep a module-level singleton keyed by `config` hash, or make `run_explainability_pipeline` accept `orchestrator: Optional[ExplainabilityOrchestrator] = None`.
- **Expected speedup:** **30-100ms** saved per call.

---

## 3. FAITHFULNESS ISSUES

### FAITH-1 — Attention rollout used as a faithful explanation without validation

- **Files:** `attention_rollout.py` (full module), `orchestrator.py:206-214`.
- **Issue:** attention has well-known issues as an explanation (Jain & Wallace 2019, "Attention is not Explanation"; Wiegreffe & Pinter 2019, "Attention is not not Explanation"). The current implementation feeds raw rollout values directly into `AggregatedExplanation` with **no validation that rollout correlates with model output sensitivity**. The default `AggregationWeights.attention=0.20` is a fixed 20 % weight regardless of whether attention actually predicts the target.
- **Fix:**
  1. Compute attention rollout AND a sensitivity check (gradient × input on the same tokens). If `pearson(rollout, |grad·x|) < 0.3`, downweight or drop.
  2. Add config knob `attention_faithfulness_threshold` and skip the attention term when the per-call agreement is below it:
     ```python
     if attention_out and ig_out:
         agreement = np.corrcoef(attention_out.importance, ig_out.importance)[0,1]
         if agreement < self.config.attention_faithfulness_threshold:
             attention_out = None
     ```

---

### FAITH-2 — `bias_explainer.compute_ig` is not integrated gradients — it is single-step gradient × embedding

- **File / Function:** `bias_explainer.py:92-102`.
- **Issue:**
  ```python
  emb = model.get_input_embeddings()(inputs["input_ids"]).detach().requires_grad_(True)
  out = model(inputs_embeds=emb)
  out.logits.max().backward()
  grads = emb.grad.abs().sum(dim=-1)[0]
  ```
  This is **gradient-input**, not integrated gradients. Real IG integrates over an interpolation path from a baseline to the input (per `score_explainer._integrated_gradients`). Single-step gradient is highly sensitive to local saturation and produces unfaithful attributions for nonlinear models.
- **Fix:** copy the path-integration loop from `src/aggregation/score_explainer.py:74-107` (or use the corrected version proposed in the aggregation audit GPU-2) and remove the `.abs()` to preserve sign:
  ```python
  ig = ((emb - baseline) * mean_grad).sum(-1)[0]   # signed
  return _normalize(ig.cpu().numpy())
  ```

---

### FAITH-3 — `explanation_calibrator.calibrate_by_method` applies arbitrary `power 0.8` (lime) and `power 1.2` (attention) with no theoretical basis

- **File / Function:** `explanation_calibrator.py:79-98`.
- **Issue:** these "shaping" exponents are not motivated by any calibration theory; they bias LIME toward sharper distributions and attention toward flatter ones, then renormalize. The transformation is irreversible and changes the relative ordering of low-importance tokens.
- **Fix:** drop the per-method shaping. Calibration of explanations should mean "make the magnitudes comparable across explainers" (e.g. quantile-mapping to a common reference distribution), not "apply a hardcoded exponent". Use a learned monotone calibration trained on a labeled faithfulness dataset, or no calibration at all.

---

### FAITH-4 — `propaganda_explainer` produces explanations divorced from model behavior

- See CRIT-9.

---

### FAITH-5 — `emotion_explainer` lexicon-derived "explanation" is not tied to model predictions

- **File:** `emotion_explainer.py:175-205`.
- **Issue:** even if `compute_gradients` were fixed (CRIT-2), `fuse(lexicon, gradients)` weights lexicon at 0.6 and gradients at 0.4 — most of the "explanation" is a fixed lexicon lookup that has no relationship to what the model actually attended to.
- **Fix:** treat the lexicon path as a separate "feature signal" not as an explanation. Return two outputs: `model_attribution` (gradient-only, faithful) and `lexicon_signal` (heuristic, marked as such).

---

### FAITH-6 — Orchestrator silently swallows explainer failures and returns `None` outputs

- **File:** `orchestrator.py:116-125`.
- **Issue:** `_run("name", fn)` catches every exception and returns `(None, latency, False)`. The aggregator then runs with whatever survived. The aggregated explanation is still presented to the user as if all explainers contributed, but `metadata.modules.shap=False` is the only signal — which the report generator does not surface.
- **Fix:** propagate failures into the final result schema so consumers can check:
  ```python
  result["module_failures"] = [k for k, ok in metadata["modules"].items() if not ok]
  if len(result["module_failures"]) > len(metadata["modules"]) // 2:
      raise RuntimeError(f"Majority of explainers failed: {result['module_failures']}")
  ```

---

## 4. TOKEN ALIGNMENT ISSUES

### ALIGN-1 — Mixed tokenization across the pipeline

- **Files:** `bias_explainer.py:152` (`tokenizer.tokenize(text)` — subword), `emotion_explainer.py:57-58` (regex word-level), `propaganda_explainer.py:70-71` (regex word-level), `shap_explainer.py:202` (SHAP's `Text` masker tokens — splits by whitespace by default), `lime_explainer.py:177` (LIME's word-level tokens from `as_list()`), `attention_rollout` (subword tokens supplied externally).
- **Issue:** the aggregator's `tokens = sorted(set(...))` (CRIT-3) merges these incompatible token spaces by string equality. `Ġworld` ≠ `world` ≠ `World` ≠ `WORLD` — the same surface form gets fragmented across methods and the aggregator's per-token lookup misses every match.
- **Fix:** introduce a single canonical token space at the orchestrator boundary:
  ```python
  enc = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
  offsets = enc["offset_mapping"][0].tolist()
  tokens = [text[s:e] for s, e in offsets if (e - s) > 0]
  ```
  Pass `tokens, offsets` to every explainer; require explainers to output in this canonical token space (provide an alignment helper that maps subword IG → word level using offsets, and word-level lexicon scores → subword level by repetition).

---

### ALIGN-2 — `align_tokens` in `token_alignment.py` only handles `##` (WordPiece) and `▁` (SentencePiece) — doesn't handle RoBERTa BPE `Ġ`

- **File:** `src/explainability/token_alignment.py:94-126`.
- **Issue:** the model is RoBERTa-based (per `src/aggregation/score_explainer.py:88` and the project stack). RoBERTa BPE uses `Ġ` for word-initial tokens. `align_tokens` does not recognize it, so subword merging never happens — every token is treated as a standalone word.
- **Fix:**
  ```python
  if tokenizer_type == "bpe":
      if token.startswith("Ġ"):
          if parts:
              val, var = agg(vals); merged_tokens.append("".join(parts))
              merged_scores.append(val); merged_variance.append(var)
          parts = [token[1:]]; vals = [score]
      else:
          parts.append(token); vals.append(score)
  ```
  Add `"bpe"` to the `tokenizer_type` literal and update default to detect the family from `tokenizer.__class__.__name__`.

---

### ALIGN-3 — `bias_explainer` calls `align_tokens(tokens, base)` with `base` derived from one explainer; `shap_vals`, `ig_vals`, `attn_vals` are kept un-aligned

- **File / Function:** `bias_explainer.py:159-166`.
- **Issue:**
  ```python
  base = next((v for v in [shap_vals, ig_vals, attn_vals] if v is not None), None)
  tokens, base = align_tokens(tokens, base)
  shap_vals = shap_vals if shap_vals is not None else np.zeros_like(base)
  ig_vals   = ig_vals   if ig_vals   is not None else np.zeros_like(base)
  attn_vals = attn_vals if attn_vals is not None else np.zeros_like(base)
  ```
  Only `base` is aligned to the merged token list. The other two arrays remain at the **original** subword length. `fuse_methods` then computes `weights["shap"] * shap_vals + weights["ig"] * ig_vals + ...` with mismatched shapes → broadcast error or silent misalignment.
- **Fix:** align all three arrays jointly:
  ```python
  shap_vals = align_array(tokens_in, shap_vals, tokens) if shap_vals is not None else np.zeros(len(tokens))
  ig_vals   = align_array(tokens_in, ig_vals,   tokens) if ig_vals   is not None else np.zeros(len(tokens))
  attn_vals = align_array(tokens_in, attn_vals, tokens) if attn_vals is not None else np.zeros(len(tokens))
  ```
  Where `align_array` collapses subwords using the same merge function as `align_tokens`.

---

### ALIGN-4 — Truncation in `attention_rollout._validate_inputs` allows `len(tokens) ≤ seq_len` but doesn't enforce equality

- **File:** `attention_rollout.py:69-70`.
- **Issue:** if `len(tokens) < seq_len` (caller stripped CLS/SEP), the rollout matrix is `seq_len × seq_len` but `tokens[i]` aligns to `attention[i+1]` (off-by-one for stripped CLS). The `mask_tokens` loop in `compute_rollout:139-142` then checks tokens by index without correcting.
- **Fix:** require `len(tokens) == seq_len` and have callers pass the full subword sequence including special tokens; mask CLS/SEP via `mask_tokens=["<s>","</s>","[CLS]","[SEP]"]` instead of stripping.

---

### ALIGN-5 — `explanation_metrics.faithfulness` joins tokens with " " and re-tokenizes implicitly via predict_fn

- See CRIT-11.

---

## 5. SCALE / NORMALIZATION ISSUES

### SCALE-1 — Triple normalization: explainer normalizes → schema validator re-normalizes → aggregator re-normalizes

- **Files:** every explainer calls `calibrate_explanation` (L1 normalize). `ExplanationOutput.normalize_importance` validator L1 normalizes again. `ExplanationAggregator._normalize` L1 normalizes the fused scores again.
- **Issue:** three independent normalization steps with no shared state. After the first normalization, every importance vector sums to 1 — so the **relative scale across methods is destroyed**. SHAP's "this token contributed +0.7 to the prediction" becomes "this token has 0.012 of the total mass" indistinguishable from any other method's 0.012.
- **Fix:**
  - Explainers return **raw signed magnitudes** (no calibration).
  - `ExplanationOutput` validates finiteness only (CRIT-5 fix).
  - `ExplanationAggregator` normalizes **once**, after weighted fusion, with method-specific rescaling derived from offline calibration on a labeled faithfulness dataset (e.g., temperature per method estimated by maximum-likelihood faithfulness fit).

---

### SCALE-2 — Mixing signed (gradients, SHAP) and non-negative (attention, lexicon) signals via L1 of `abs`

- **Files:** `explanation_calibrator.normalize_scores:34` does `np.abs(arr) / sum(abs(arr))`. `bias_explainer._normalize:53-54` does `np.maximum(x, 0); x / sum(x)`. Inconsistent.
- **Issue:** SHAP returns signed values (positive = pushes prediction toward "fake", negative = pushes toward "real"). Taking `abs()` collapses both into magnitudes, losing direction. `np.maximum(x, 0)` drops the entire negative half. The two utilities behave differently on the same input.
- **Fix:** preserve sign. Maintain two separate fields on `TokenImportance`:
  ```python
  importance: float          # |attribution|, in [0,1]
  direction: Literal["positive","negative","neutral"]
  ```
  And let the visualizer/report colorize accordingly.

---

### SCALE-3 — `explanation_calibrator.compute_confidence` normalizes entropy by `log(N+EPS)`, where N is the number of input scores — confounds scale with vocabulary size

- **File:** `explanation_calibrator.py:55-72`.
- **Issue:** for two articles with very different token counts (50 vs 500), the same uniform distribution gets reported as confidence 0 for both — but the confidence should reflect how peaked the distribution is, not the article length. Currently `confidence = 1 - entropy / log(N)` — the denominator changes per call, so confidences are not comparable across articles.
- **Fix:** report entropy in nats and confidence as a fixed-reference quantity:
  ```python
  effective_size = math.exp(entropy)              # perplexity-equivalent
  confidence = 1.0 - min(effective_size / 50.0, 1.0)  # 50 = chosen reference
  ```
  Or expose the raw entropy and let consumers decide.

---

### SCALE-4 — `explanation_aggregator` per-token confidence formula `1 - std(vals)` can go negative

- **File:** `explanation_aggregator.py:171`.
- **Issue:** `vals` are already L1-normalized importance values. For 4 methods that completely disagree, `std` can be > 1 if the normalization differs across articles (different N tokens). `1 - 1.5 = -0.5` is then `np.clip(conf, 0.0, 1.0) = 0` — perfect-disagreement and complete-collapse-to-zero are reported identically.
- **Fix:** normalize std by max-possible std for the given methods:
  ```python
  max_std = 0.5   # for L1-normalized vectors over 4 methods, theoretical max ≈ 0.5
  conf = float(np.clip(1.0 - np.std(vals) / max_std, 0.0, 1.0))
  ```
  Or use cosine similarity over method-vectors instead of std.

---

### SCALE-5 — `explanation_metrics._normalize` uses min-max on 5 metric values then averages — destroys magnitude

- **File:** `explanation_metrics.py:34-44, 184-191`.
- **Issue:** the 5 raw metrics (`faithfulness, comprehensiveness, sufficiency, deletion_score, insertion_score`) are min-max normalized **per call** to `[0,1]`, then averaged. So the largest metric is always 1, smallest always 0, and `overall_score = mean = 0.5` for any input. The `overall_score` is a constant.
- **Fix:** use a population-fitted normalizer (loaded from disk) so per-call normalization is comparable across calls. Or simply average the raw metrics with documented domain semantics:
  ```python
  result = {
      **weighted,
      "variance": self.variance(scores),
      "overall_score": float((weighted["faithfulness"]
                              + weighted["comprehensiveness"]
                              + (1.0 - weighted["sufficiency"])) / 3.0),
  }
  ```

---

## 6. GPU / TORCH ISSUES

### GPU-1 — `attention_rollout.compute_rollout` casts `bf16/fp16 → fp32` per-layer in a Python loop; no batched cast

- **File:** `attention_rollout.py:120-123`.
- **Issue:** for each layer, `attn = attn.to(torch.float32)` allocates a new tensor on device. For 12 layers × seq_len² = 12 × 512² × 4 bytes = 12 MB of allocations per article.
- **Fix:** stack first, cast once:
  ```python
  attn_stack = torch.stack([self._aggregate_heads(a, sample_index) for a in attentions]).to(torch.float32)
  ```

---

### GPU-2 — `attention_rollout.compute_rollout` does `.detach().cpu().numpy()` mid-pipeline, then `np.maximum`, then numpy ops — kills device parallelism

- **File:** `attention_rollout.py:132-142`.
- **Issue:** the rollout is computed on GPU, then immediately moved to CPU for `np.nan_to_num`, `np.maximum`, the mask loop, and the `calibrate_explanation` call. The remainder of the pipeline (top-k, structured assembly) runs on CPU.
- **Fix:** keep on device until all tensor ops are done:
  ```python
  scores = rollout[source_token_index].clamp_min(0)
  scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
  if mask_tokens:
      mask_idx = torch.tensor([i for i, t in enumerate(tokens) if t in mask_tokens],
                              device=scores.device, dtype=torch.long)
      scores[mask_idx] = 0
  scores = scores.cpu().numpy().astype(np.float32)   # single transfer
  ```

---

### GPU-3 — `bias_explainer.compute_ig` runs a single forward pass on default device, ignoring whether `model` is on CUDA

- **File:** `bias_explainer.py:92-102`.
- **Issue:** `inputs = tokenizer(text, return_tensors="pt")` returns CPU tensors. `emb = model.get_input_embeddings()(inputs["input_ids"])` then triggers `Tensor.to(model.device)` implicitly only if the embeddings layer detects mismatch — older PyTorch raises `RuntimeError: Tensor on CPU passed to module on cuda`.
- **Fix:**
  ```python
  device = next(model.parameters()).device
  inputs = {k: v.to(device) for k, v in tokenizer(text, return_tensors="pt").items()}
  ```

---

### GPU-4 — `lime_explainer` and `shap_explainer` run their `predict` callback in a Python loop over 256 perturbed strings

- See PERF-1 and PERF-2. From a GPU perspective: the GPU sits idle 90 % of the time because each forward is single-text.
- **Fix:** require `predict_fn.batch_predict(texts: List[str]) -> List[Dict]` that internally batches to GPU. Without it, `lime_predict_wrapper` falls back to per-text calls.

---

### GPU-5 — `explanation_visualizer` and `attention_visualizer` import `matplotlib` at module level

- **Files:** `explanation_visualizer.py:7`, `attention_visualizer.py:11`.
- **Issue:** `import matplotlib.pyplot as plt` is heavyweight (~200 ms) and unused on the API hot path. Always-imported even when no plotting requested.
- **Fix:** lazy-import inside plotting methods:
  ```python
  def plot_token_heatmap(self, ...):
      import matplotlib.pyplot as plt
      ...
  ```

---

## 7. RECOMPUTATION ISSUES

### REC-1 — Same forward pass repeated by every explainer (see PERF-1)

SHAP, LIME, IG, attention rollout, gradient (in emotion), and `predict_fn` all forward the same input independently. Fix via shared `ModelForwardCache`.

---

### REC-2 — `ExplanationCache` not consulted when sub-modules are computed

- **File:** `orchestrator.py:145-148`.
- **Issue:** the orchestrator's top-level cache stores **the whole assembled explanation**. But `shap_explainer._VALUE_CACHE`, `lime_explainer._EXPLANATION_CACHE`, and the orchestrator cache are three independent caches with three independent eviction policies and three independent keys. A cache hit at the top level skips everything; a miss at top level re-runs every sub-explainer even if `_VALUE_CACHE` would have hit.
- **Fix:** unify under one cache layer keyed by `(text, model_version, config_hash)`. Sub-explainers receive a cache view and store/look up under their own sub-keys.

---

### REC-3 — `orchestrator.explain` runs `predict_fn(text)` inside `explanation_metrics.evaluate` but had already run it in `explainability_pipeline.run_explainability_pipeline` (line 77)

- **Files:** `explainability_pipeline.py:77`, `explanation_metrics.faithfulness:60`, `comprehensiveness:82`, `sufficiency:97`.
- **Issue:** `prediction = predict_fn(text)` is computed at the top, then thrown away after wrapping. Inside metrics, `base = predict_fn([" ".join(tokens)])[0]["fake_probability"]` is computed three more times with slight variants of the input.
- **Fix:** thread `prediction` and `base_text_proba` through to `metrics.evaluate(...)`:
  ```python
  metrics = self.metrics.evaluate(
      tokens, scores, batch_predict_fn,
      base_proba=prediction.get("fake_probability"),
      original_text=text,
  )
  ```

---

### REC-4 — `explanation_consistency._compare` re-normalizes every source dict each time `compute()` is called

- **File:** `explanation_consistency.py:163-177`.
- **Issue:** `self._normalize(shap_m)`, `self._normalize(ig_m)` etc. are computed once per `compute()` call but the sources are unchanged across calls (orchestrator runs consistency only once per article). OK in isolation. But `_token_consistency` then iterates `for t in tokens: vals = [src[t] for src in sources.values() if t in src]` — Python loop over tokens × sources.
- **Fix:** stack into a numpy matrix and compute std along axis 0:
  ```python
  tokens = sorted(set().union(*[set(s) for s in sources.values()]))
  mat = np.full((len(sources), len(tokens)), np.nan)
  for mi, src in enumerate(sources.values()):
      for ti, t in enumerate(tokens):
          if t in src: mat[mi, ti] = src[t]
  per_token_std = np.nanstd(mat, axis=0)
  token_scores = dict(zip(tokens, np.clip(1.0 - per_token_std, 0, 1)))
  ```

---

## 8. UNUSED EXPLAINERS

| Explainer / Module | Status | Action |
|---|---|---|
| `model_explainer.py` (entire file) | DUPLICATE of `explainability_pipeline.py`, divergent defaults | Delete |
| `explanation_visualizer.ExplanationVisualizer` (`explanation_visualizer.py:18-244`) | Defined, never imported by orchestrator or report generator | Either wire from `explanation_report_generator.save_html` or delete |
| `attention_visualizer.AttentionVisualizer` (`attention_visualizer.py:19-218`) | Imports `matplotlib` at module load, never instantiated by orchestrator | Make optional; lazy-import |
| `attention_visualizer.analyze` (`attention_visualizer.py:198-217`) | Provides full attention extraction → rollout pipeline; orchestrator's attention path requires caller to manually compute attentions instead | Wire as the canonical attention path: `orchestrator.explain(...)` should accept `model` and call `AttentionVisualizer(model).analyze(...)` if `attentions is None` |
| `propaganda_explainer.detect_techniques` (`propaganda_explainer.py:189-198`) | Defined, no caller | Either expose via `__init__.py` or fold into `explain_propaganda` output |
| `bias_explainer.BiasExplanation` dataclass | Returned as `.__dict__`, type info lost downstream | Return dataclass or `ExplanationOutput`; never `.__dict__` |
| `emotion_explainer.EmotionExplanation` dataclass | Same as above | Same fix |
| `explanation_aggregator.AggregationWeights.graph` | Configured (default 0.10) but `graph_explanation` only injected from `GraphExplainer.explain_from_text(text)` which may not return `node_importance` for short inputs | Either guarantee graph_explanation contract or remove graph weight |
| `lime_explainer._EXPLANATION_CACHE` (`lime_explainer.py:31`) | Unbounded `dict` (not OrderedDict, no eviction) → memory leak | Replace with `OrderedDict` + `_MAX_CACHE_SIZE` eviction (already done for explainer cache) |
| `lime_explainer.save_explanation_html` (`lime_explainer.py:226-251`) | Public, no caller | Expose via `__init__.py` or delete |
| `shap_explainer.plot_explanation`, `save_explanation_html` (`shap_explainer.py:253-277`) | Public, no caller | Same |
| `explanation_calibrator.calibrate_by_method` per-method shaping | See FAITH-3 — arbitrary, no theory | Drop the power transforms |
| `explanation_metrics.evaluate_batch` (`explanation_metrics.py:200-232`) | Public, no caller | Acceptable; keep but expose via `__init__.py` |
| `explanation_metrics.variance` (`explanation_metrics.py:149-153`) | Trivial, exposed as a method | Inline at the call site |
| `token_alignment.return_variance, return_structured` modes | Defined, no caller | Either wire into orchestrator (variance is a faithfulness signal!) or delete |
| `utils_validation.auto_fix, return_fixed, allow_duplicates` modes | Defined, no caller | Same |
| `__init__.py` exports `ExplainabilityConfig` and `ExplainabilityOrchestrator` only | `run_explainability_pipeline`, `explain_prediction_full`, `explain_fast`, `ExplainabilityResult`, `AggregatedExplanation`, `TokenImportance`, etc. all hidden | Expand `__all__` |

---

## 9. CONFIG ISSUES

### CFG-1 — Hardcoded explainer choices in `bias_explainer` and `emotion_explainer`

- **Files:** `bias_explainer.py:147-189`, `emotion_explainer.py:175-205`.
- **Issue:** these functions hardcode "compute SHAP, IG, attention rollout" with no config. The `ExplainabilityConfig.use_shap` / `use_attention_rollout` / etc. flags are ignored inside `explain_bias`.
- **Fix:** thread the config through:
  ```python
  def explain_bias(model, tokenizer, text, *, config: ExplainabilityConfig):
      shap_vals = compute_shap(...) if config.use_shap else None
      ig_vals   = compute_ig(...)   if config.use_ig else None
      attn_vals = compute_attention_rollout(...) if config.use_attention_rollout else None
      ...
  ```

---

### CFG-2 — `ExplainabilityConfig` lacks `use_graph_explainer` flag, but graph explainer always runs

- **Files:** `orchestrator.py:108, 219-226`.
- **Issue:** `self.graph_explainer = GraphExplainer()` is unconditional; the explain method always invokes it. There is no way to disable.
- **Fix:** add the flag:
  ```python
  use_graph_explainer: bool = True
  ...
  self.graph_explainer = GraphExplainer() if config.use_graph_explainer else None
  ...
  if self.graph_explainer:
      graph_expl, t, ok = self._run("graph_explainer", lambda: self.graph_explainer.explain_from_text(text))
  ```

---

### CFG-3 — `AggregationWeights` not loaded from YAML config

- **Files:** `orchestrator.py:63-65` defaults to `AggregationWeights()` (the dataclass defaults).
- **Issue:** users editing `config/config.yaml` cannot override `shap=0.35, integrated_gradients=0.25, attention=0.20, lime=0.10, graph=0.10`.
- **Fix:** add to `AggregationConfig` (or a new `ExplainabilityConfig` Pydantic model in `aggregation_config.py`):
  ```yaml
  explainability:
    aggregation_weights:
      shap: 0.35
      integrated_gradients: 0.25
      attention: 0.20
      lime: 0.10
      graph: 0.10
  ```

---

### CFG-4 — `lime_explainer.num_samples=256` and `num_features=10` hardcoded as defaults; not in config

- **File:** `lime_explainer.py:154-155`.
- **Fix:** expose:
  ```python
  @dataclass
  class LimeConfig:
      num_features: int = 10
      num_samples: int = 256
      batch_size: int = 32
  ```
  And read from `ExplainabilityConfig.lime`.

---

### CFG-5 — `attention_rollout` `top_k` is per-call, but no global default in config

Acceptable, but inconsistent with `lime`/`shap` which have module-level defaults. Consolidate.

---

### CFG-6 — `explanation_cache_max_size`, `cache_dir`, `ttl_seconds`, `enable_compression` configurable on `ExplanationCache` but not on `ExplainabilityConfig`

- **File:** `orchestrator.py:77-84`.
- **Issue:** `ttl_seconds` and `enable_compression` not exposed in `ExplainabilityConfig`.
- **Fix:** add `cache_ttl_seconds: Optional[int] = None`, `cache_compression: bool = True`.

---

## 10. EDGE CASE FAILURES

| Edge case | File / line | Failure | Fix |
|---|---|---|---|
| Empty text | `explainability_pipeline.py:68-69` raises `ValueError`; OK. But `bias_explainer.explain_bias:149-150` also raises. `emotion_explainer.explain_emotion` does NOT validate → `tokenize("")=[]` → `compute_lexicon([])=array([])` → `np.mean(fused)` returns `nan` | partial | Add `if not text.strip(): raise ValueError` to all explainers |
| All padding tokens | `attention_rollout.compute_rollout` produces all-zero scores after `np.maximum(0)`; `calibrate_explanation` returns `np.zeros_like(arr)` (correct); but `ExplanationOutput.normalize_importance` then returns `[0.0]*N`; `TokenImportance.importance` field is `ge=0.0, le=1.0` so OK | partial | Fine as-is |
| Long sequences > 512 | Caller truncates at tokenizer; rollout sees 512 tokens; tokens list externally provided may be longer → `_validate_inputs:69-70` raises `"tokens exceed seq_len"` | OK | Add explicit truncation in orchestrator to seq_len before passing |
| Batch size = 1 | `attention_rollout` works; SHAP/LIME work; explanation_metrics.faithfulness `if len(deltas) < 2: return 0.0` → returns 0 silently for 1-token text | poor | Surface as a metadata flag rather than 0 |
| `predict_fn` raises | `_run` catches; `shap_out=None`; aggregator runs without SHAP; metrics fail because `agg.tokens=[]` if all explainers failed | poor | Add `if not agg or not agg.tokens: skip metrics` and report `error_summary` |
| `predict_fn` returns missing `fake_probability` | `_extract_fake_probability` raises `KeyError`; `_run` catches; SHAP returns None; same downstream | partial | Allow alternative key via config (`predict_fn_target_key: str = "fake_probability"`) |
| Tokenizer returns 0 tokens for special-only text (e.g. `"<s></s>"`) | `shap_explainer:209-217` handles via `if not filtered`; `lime_explainer:179-185` handles via `if not raw_features`; `attention_rollout._validate_inputs:37-38` raises | OK |
| Numerical NaN/Inf in scores | `calibrate_explanation` does `nan_to_num`; `attention_rollout` does `nan_to_num`; `bias_explainer.compute_shap` does NOT (see fix) | partial | Add `nan_to_num` in `_normalize` for bias and emotion |
| `model` on `cuda`, inputs on `cpu` | `bias_explainer.compute_ig`, `emotion_explainer.compute_gradients` will raise `RuntimeError` | broken | GPU-3 fix |
| Cache key collision across model versions | `lime_explainer._make_cache_key` ignores model version; `explanation_cache._make_key` includes `model_version="default"` | partial | Plumb `model_version` from orchestrator into all sub-caches |
| `orchestrator.cache.get(text)` returns dict | The dict is what the orchestrator emits, but `ExplainabilityResult` consumer expects pydantic model | partial | Cache the validated model, not the raw dict |
| Single-token text | `validate_tokens_scores:122-126` raises `"scores have near-zero variance"` for single-token (variance = 0) — blocks legitimate single-word inputs | bad | Skip variance check when `len(scores) <= 1` |
| LIME fails to converge | `explainer.explain_instance` may return empty `as_list`; `lime_explainer:179-185` returns empty `ExplanationOutput`; aggregator skips the missing source | OK |
| `shap_values.values[0]` shape unexpected | `_process_shap_values` handles `ndim==3` and `ndim==2,shape[1]==1`; other shapes pass through unchanged → may break downstream | partial | Add `assert values.ndim == 1` after the branches and raise informative error |
| `attentions` on different devices | `_aggregate_heads` uses tensor's own device; `_add_residual` builds identity on `attention.device` — OK | OK |
| `tokens` contain non-string | `attention_rollout._validate_inputs:37-38` only checks list; rollout proceeds; `mask_tokens in tokens` may fail downstream | poor | Add `if not all(isinstance(t, str) for t in tokens): raise TypeError` |
| `confidence == None` propagated to aggregator | `confidences[name] = shap.confidence or 0.5` — fine | OK |
| `explanation_aggregator.aggregate(shap=None, ig=None, attn=None, lime=None, graph_explanation=None)` | `if not sources and not graph_node_importance: raise ValueError("No sources provided")` — OK |
| Cache disk write fails (permission denied) | `explanation_cache.set:178-183` catches and ignores → silent data loss; consumer thinks cache write succeeded | poor | Log a warning at minimum |

---

## 11. VERIFIED COMPONENTS

| Component | Status |
|---|---|
| Pydantic schemas (`common_schema.TokenImportance`, `ExplanationOutput`, `AggregatedExplanation`) validate types and finiteness | OK |
| `EPS = 1e-12` consistently used for division-safe normalization | OK (all 22 files) |
| `np.nan_to_num` guards on probability arrays (`shap_explainer`, `attention_rollout`, `explanation_calibrator`, `token_alignment`) | OK |
| `RLock` for thread-safe SHAP / LIME / cache mutation | OK |
| `OrderedDict` + LRU eviction on `_EXPLAINER_CACHE` and `_VALUE_CACHE` in `shap_explainer` | OK |
| `attention_rollout._validate_inputs` checks 4D shape, square seq, batch consistency, sample-index range | OK |
| `explanation_cache.CACHE_VERSION` validates disk-cache compatibility on read | OK |
| `explanation_cache._is_expired` honors TTL | OK |
| `explanation_cache.stats` reports hit/miss rate | OK |
| `propaganda_explainer.PROPAGANDA_PATTERNS` weights documented per technique | OK |
| `attention_rollout` uses `multi_dot` (the right primitive for serial matmul cascade) | OK |
| `attention_rollout` correctly reverses processed list before `multi_dot` | OK |
| `attention_rollout._add_residual` clamps row-sum with `EPS` to prevent zero-division | OK |
| `shap_explainer.SPECIAL_TOKENS` filter strips CLS/SEP/PAD/UNK before output | OK |
| `shap_explainer._stable_predict_fn_key` attempts to stabilize cache key across reloads | OK (partial) |
| `explanation_metrics.faithfulness` handles `< 2` deltas by returning 0.0 | OK (defensive) |
| `explanation_metrics.deletion_score` and `insertion_score` use ranked-importance ordering | OK |
| `explanation_consistency._pearson` guards against zero-variance inputs | OK |
| `explanation_consistency._cosine` uses EPS in denominator | OK |
| `token_alignment` handles WordPiece (`##`) and SentencePiece (`▁`) prefixes | OK (partial — BPE missing per ALIGN-2) |
| `explainability_pipeline.run_explainability_pipeline` validates non-empty text | OK |
| `explanation_aggregator._normalize` uses `np.abs` + EPS for division safety | OK |
| `explanation_monitor.update` clamps history to `max_history` | OK |

---

## 12. FINAL SCORE

**Score: 4.0 / 10**

### Why not 10

1. **`bias_explainer` and `emotion_explainer` cannot run on the multitask model** — `out.logits` doesn't exist (CRIT-1, CRIT-2). All bias/emotion explanations return `None`, silently masked by `_run`.
2. **Aggregator sorts tokens alphabetically** (CRIT-3) and **collapses duplicates** (CRIT-4) — destroys positional and repetition signals critical for propaganda detection.
3. **Schema validator silently re-normalizes `importance` while `structured` keeps raw values** (CRIT-5) — `out.importance[i] != out.structured[i].importance`.
4. **Two `ExplainabilityResult` classes and two duplicate entry-point modules** (CRIT-6, CRIT-7) — schema confusion across the codebase.
5. **IG output computed in `bias_explainer` is never wired to aggregator or consistency** (CRIT-8) — 25 % of aggregation weight goes to nothing.
6. **`propaganda_explainer` is text-only (no model) but presented as a faithful explanation** (CRIT-9, FAITH-4).
7. **`_spearman` is mathematically wrong** (CRIT-10) — uses `argsort` instead of ranks.
8. **`explanation_metrics` ablates by joining tokens with `" "`** (CRIT-11) — computes faithfulness on text the model has never seen, especially when tokens come from the alphabetic aggregator (CRIT-3).
9. **3·N + 2 sequential forward passes per article in metrics, plus 256 in LIME, plus 6+ across other explainers** (CRIT-12, PERF-1, PERF-2) — single article takes tens of seconds.
10. **Triple normalization** (SCALE-1) — explainer → schema → aggregator each L1-normalize independently, destroying cross-method magnitude semantics.
11. **`compute_ig` in `bias_explainer` is single-step gradient × input, not integrated gradients** (FAITH-2).
12. **Calibrator's per-method `power 0.8` / `power 1.2` shaping has no theoretical basis** (FAITH-3).
13. **Tokenization mismatch across explainers** (ALIGN-1, ALIGN-2, ALIGN-3) — word-regex, BPE, WordPiece, LIME-internal, SHAP-Text all collide in the aggregator.
14. **GPU underutilization**: matplotlib loaded eagerly (GPU-5), mid-pipeline `.cpu().numpy()` (GPU-2), per-call orchestrator instantiation with full graph-explainer load (PERF-6).
15. **Three independent caches** (orchestrator, SHAP, LIME) with no coordination (REC-2).
16. **`explainability_pipeline.run_explainability_pipeline` re-instantiates orchestrator every call** (PERF-6).
17. **No `use_graph_explainer` config flag** (CFG-2) — graph explainer always runs.
18. **`AggregationWeights` not loaded from YAML** (CFG-3) — config knobs ignored.
19. **`lime_explainer._EXPLANATION_CACHE` is unbounded** (UNUSED) — memory leak.

### Steps to reach ≥ 9.5

1. **Fix multitask model integration** in `bias_explainer.compute_shap/_ig/_attention_rollout` and `emotion_explainer.compute_gradients`. Pattern: `model.encoder(...)` → `model.heads[task](cls)`. (CRIT-1, CRIT-2, FAITH-2, GPU-3)
2. **Replace alphabetic-sort + dict-zip in aggregator with positional indexing** preserving original token order and duplicates. (CRIT-3, CRIT-4)
3. **Make `ExplanationOutput.normalize_importance` a pass-through validator** that only checks finiteness and length consistency; assert `structured[i].importance == importance[i]`. (CRIT-5)
4. **Delete `model_explainer.py` and `explainability_pipeline.ExplainabilityResult`**; consolidate on `common_schema.ExplainabilityResult`. (CRIT-6, CRIT-7)
5. **Wire IG output from `bias_explainer` into aggregator and consistency**. (CRIT-8)
6. **Mark `propaganda_explainer` outputs with `faithful=False`** and gate aggregator inclusion via config. (CRIT-9, FAITH-4)
7. **Fix `_spearman`** to use `np.argsort(np.argsort(a))` or `scipy.stats.rankdata`. (CRIT-10)
8. **Re-architect `explanation_metrics` to ablate text** using `(start,end)` offsets, not token-string joining. Pass original `text` and `offsets` end-to-end. (CRIT-11, ALIGN-1)
9. **Introduce `ModelForwardCache`** — one model forward per article shared by SHAP, LIME, IG, attention rollout, gradient. (PERF-1, REC-1, REC-2, REC-3)
10. **Require `predict_fn.batch_predict`** at the orchestrator boundary; fail loudly if missing in production. (PERF-2, CRIT-12, GPU-4)
11. **Single normalization point** in the aggregator, with method-specific calibration learned offline rather than hardcoded power transforms. (SCALE-1, FAITH-3)
12. **Add BPE (`Ġ`) handling to `align_tokens`** and detect tokenizer family automatically. (ALIGN-2)
13. **Align all three arrays in `bias_explainer.fuse_methods`** to a common token space, not just `base`. (ALIGN-3)
14. **Add `use_graph_explainer` config flag** and honor it in `orchestrator.__init__` and `explain`. (CFG-2)
15. **Plumb `AggregationWeights`, `LimeConfig`, `cache_ttl_seconds`, `cache_compression` from `config/config.yaml`** through `ExplainabilityConfig`. (CFG-3, CFG-4, CFG-6)
16. **Replace `lime_explainer._EXPLANATION_CACHE` `dict` with bounded `OrderedDict` + LRU eviction** (mirroring `_EXPLAINER_CACHE`). (UNUSED, leak)
17. **Lazy-import `matplotlib`** in plotting methods; do not load at module import. (GPU-5)
18. **Stop `.detach().cpu().numpy()` mid-pipeline in `attention_rollout`**; keep on device until final assembly. (GPU-2)
19. **Make orchestrator a singleton** keyed by config hash; stop re-instantiating per call. (PERF-6)
20. **Surface `module_failures` in the final result** and raise when more than half the explainers fail. (FAITH-6)
21. **Replace `.__dict__` returns with `dataclasses.asdict` or proper `ExplanationOutput`** in bias/emotion explainers. (UNUSED)

### After fix and error solving: **9.5 / 10**

The remaining 0.5 reflects the inherent fragility of multi-explainer fusion: even with all fixes, attention-as-explanation remains contested in the literature (FAITH-1), and the per-method calibration weights are policy choices that need empirical validation on a labeled faithfulness dataset.
