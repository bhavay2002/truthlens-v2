"""
tests/test_v2_e2e.py
--------------------
End-to-end test suite for Explainability v2 + Inference v2 using 100 rows
of synthetic data.  No real checkpoint or GPU required.

Covers:
  E2E-1  CrossTaskAttributor  (gate-based and gradient-based)
  E2E-2  AggregatorSHAP       (KernelExplainer on MLPAggregator)
  E2E-3  AttentionGraphBuilder (token→task→decision graph)
  E2E-4  SinglePassAnalyzer   (async, mock model + tokenizer)
  E2E-5  PCGradOptimizer      (pc_backward, task gating, full train step)
  E2E-6  ExplainabilityResult (schema round-trip with all v2 fields)
  E2E-7  Full pipeline        (100 rows: attribute → graph → schema)

Run:
    python -m pytest tests/test_v2_e2e.py -v
"""

from __future__ import annotations

import math
import types
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn as nn

# =========================================================
# SYNTHETIC DATA FACTORY  (100 rows, reproducible)
# =========================================================

SEED = 42
RNG = np.random.default_rng(SEED)
torch.manual_seed(SEED)

N_ROWS   = 100
N_TASKS  = 4          # bias, emotion, ideology, propaganda
TASK_NAMES = ["bias", "emotion", "ideology", "propaganda"]
HIDDEN   = 32         # tiny hidden dim
SEQ_LEN  = 16
FEAT_DIM = 47         # canonical AggregatorFeatureBuilder output
VOCAB    = 200        # tiny tokenizer vocab


def make_feature_rows(n: int = N_ROWS) -> np.ndarray:
    """(N, 47) float32 feature matrix."""
    X = RNG.random((n, FEAT_DIM)).astype(np.float32)
    return X


def make_token_ids(n: int = N_ROWS, seq: int = SEQ_LEN) -> torch.Tensor:
    return torch.randint(0, VOCAB, (n, seq))


def make_attention_mask(n: int = N_ROWS, seq: int = SEQ_LEN) -> torch.Tensor:
    mask = torch.ones(n, seq, dtype=torch.long)
    # zero out last 2 positions for variety
    mask[:, -2:] = 0
    return mask


def make_task_losses(tasks: List[str] = TASK_NAMES, device="cpu") -> Dict[str, torch.Tensor]:
    return {task: torch.tensor(RNG.random(), dtype=torch.float32, requires_grad=True)
            for task in tasks}


# =========================================================
# MOCK INTERACTING MULTI-TASK MODEL
# =========================================================
# Mimics just the attributes CrossTaskAttributor and SinglePassAnalyzer need,
# without loading transformers or any real checkpoint.

class _MockEncoder(nn.Module):
    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.proj = nn.Linear(VOCAB, hidden)

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        emb = torch.zeros(B, L, VOCAB)
        emb.scatter_(2, input_ids.unsqueeze(-1), 1.0)       # one-hot
        seq = self.proj(emb)                                 # (B, L, H)
        return {"sequence_output": seq}


class _MockPooling(nn.Module):
    def forward(self, sequence_output, attention_mask):
        return sequence_output.mean(dim=1)                   # (B, H)


class _MockTaskProj(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)
    def forward(self, x):
        return torch.relu(self.lin(x))


class _MockTaskEmbed(nn.Module):
    def __init__(self, n_tasks, hidden):
        super().__init__()
        self.emb = nn.Embedding(n_tasks, hidden)
    def forward(self, idx, batch_size, device):
        return self.emb(torch.tensor(idx, device=device)).unsqueeze(0).expand(batch_size, -1)


class _MockInteraction(nn.Module):
    """Minimal CrossTaskInteractionLayer mimic."""
    def __init__(self, hidden, n_tasks):
        super().__init__()
        self.scale = math.sqrt(hidden)
        self.q_projs = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(n_tasks)])
        self.k_proj  = nn.Linear(hidden, hidden, bias=False)
        self.v_proj  = nn.Linear(hidden, hidden, bias=False)
        self.gates   = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_tasks)])
        self.norms   = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_tasks)])
        self.drop    = nn.Dropout(0.0)

    def forward(self, task_reprs: List[torch.Tensor]) -> List[torch.Tensor]:
        H_all = torch.stack(task_reprs, dim=1)
        K = self.k_proj(H_all)
        V = self.v_proj(H_all)
        out = []
        for i, (H_i, q, gate, norm) in enumerate(
            zip(task_reprs, self.q_projs, self.gates, self.norms)
        ):
            Q = q(H_i).unsqueeze(1)
            attn = torch.softmax(torch.bmm(Q, K.transpose(1,2)) / self.scale, dim=-1)
            ctx  = self.drop(torch.bmm(attn, V).squeeze(1))
            g    = torch.sigmoid(gate(H_i))
            out.append(norm(H_i + g * ctx))
        return out


class MockInteractingModel(nn.Module):
    """Fully functional mock that satisfies both CrossTaskAttributor and SinglePassAnalyzer."""

    def __init__(self, hidden=HIDDEN, n_tasks=N_TASKS, task_names=None):
        super().__init__()
        self._task_names = task_names or TASK_NAMES
        self._task_to_idx = {t: i for i, t in enumerate(self._task_names)}
        self.hidden_size = hidden

        self.encoder            = _MockEncoder(hidden)
        self.multi_view_pooling = _MockPooling()
        self.task_projections   = nn.ModuleDict(
            {t: _MockTaskProj(hidden) for t in self._task_names}
        )
        self.task_embed  = _MockTaskEmbed(n_tasks, hidden)
        self.interaction = _MockInteraction(hidden, n_tasks)

        # Per-task classification heads (binary for simplicity)
        self.task_heads = nn.ModuleDict(
            {t: nn.Linear(hidden, 2) for t in self._task_names}
        )

        # Credibility + risk heads
        self.cred_head = nn.Linear(hidden * n_tasks, 1)
        self.risk_head = nn.Linear(hidden * n_tasks, 3)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        enc  = self.encoder(input_ids, attention_mask)
        seq  = enc["sequence_output"]
        mask = attention_mask if attention_mask is not None else torch.ones(seq.shape[:2])

        H    = self.multi_view_pooling(seq, mask)
        B, D = H.shape
        dev  = H.device

        task_reprs = {}
        for task in self._task_names:
            Hi = self.task_projections[task](H)
            Hi = Hi + self.task_embed(self._task_to_idx[task], B, dev)
            task_reprs[task] = Hi

        interacted = self.interaction([task_reprs[t] for t in self._task_names])
        for task, Hi in zip(self._task_names, interacted):
            task_reprs[task] = Hi

        task_logits = {t: self.task_heads[t](task_reprs[t]) for t in self._task_names}

        Z_cat   = torch.cat([task_reprs[t] for t in self._task_names], dim=-1)
        cred    = torch.sigmoid(self.cred_head(Z_cat)).squeeze(-1)
        risk    = self.risk_head(Z_cat)

        return {
            "task_logits":        task_logits,
            "task_outputs":       {t: {"logits": task_logits[t]} for t in self._task_names},
            "task_representations": task_reprs,
            "latent_vector":      Z_cat,
            "credibility_score":  cred,
            "risk":               risk,
        }


# =========================================================
# MOCK TOKENIZER
# =========================================================

class MockTokenizer:
    """Minimal HuggingFace-compatible tokenizer."""

    def __call__(self, text, padding=True, truncation=True, max_length=512, return_tensors="pt"):
        words = text.split()[:SEQ_LEN]
        ids   = [hash(w) % VOCAB for w in words]
        ids   = (ids + [0] * SEQ_LEN)[:SEQ_LEN]
        mask  = [1] * len(words) + [0] * (SEQ_LEN - len(words))
        mask  = mask[:SEQ_LEN]
        return {
            "input_ids":      torch.tensor([ids]),
            "attention_mask": torch.tensor([mask]),
        }

    def convert_ids_to_tokens(self, ids):
        return [f"tok_{i}" for i in ids]


# =========================================================
# E2E-1: CrossTaskAttributor
# =========================================================

class TestCrossTaskAttributorE2E(unittest.TestCase):

    def setUp(self):
        self.model = MockInteractingModel()
        self.model.eval()
        self.tok = MockTokenizer()
        from src.explainability.cross_task_attribution import CrossTaskAttributor
        self.attr = CrossTaskAttributor(method="gate")

        # 100-row batch of tokenized inputs (we run 5 at a time to stay light)
        self.all_ids  = make_token_ids()
        self.all_mask = make_attention_mask()

    def _batch(self, i):
        return {
            "input_ids":      self.all_ids[i:i+4],
            "attention_mask": self.all_mask[i:i+4],
        }

    def test_gate_method_returns_complete_matrix(self):
        """Every task should have influence values for every other task."""
        result = self.attr.attribute(self.model, self._batch(0))
        self.assertEqual(set(result.keys()), set(TASK_NAMES))
        for task_i, row in result.items():
            self.assertEqual(set(row.keys()), set(TASK_NAMES))

    def test_gate_method_rows_sum_to_one(self):
        """L1-normalised: each row sums to 1.0."""
        result = self.attr.attribute(self.model, self._batch(0))
        for task, row in result.items():
            total = sum(row.values())
            self.assertAlmostEqual(total, 1.0, places=5,
                msg=f"Row sum for {task} = {total}")

    def test_gate_method_values_nonneg(self):
        """Gate-based values should be non-negative."""
        result = self.attr.attribute(self.model, self._batch(4))
        for task_i, row in result.items():
            for task_j, v in row.items():
                self.assertGreaterEqual(v, 0.0,
                    f"Negative influence {task_i}→{task_j}: {v}")

    def test_gradient_method_complete_matrix(self):
        """Gradient method also returns a full NxN matrix."""
        from src.explainability.cross_task_attribution import CrossTaskAttributor
        attr_grad = CrossTaskAttributor(method="gradient")
        result = attr_grad.attribute(self.model, self._batch(8))
        self.assertEqual(set(result.keys()), set(TASK_NAMES))
        for task, row in result.items():
            total = sum(row.values())
            # places=4: Frobenius-norm normalisation accumulates float32 rounding
            # (observed delta ~5e-6), so 4 decimal places is the right tolerance.
            self.assertAlmostEqual(total, 1.0, places=4)

    def test_attribute_safe_returns_none_on_bad_model(self):
        """attribute_safe returns None instead of raising."""
        bad = object()  # not a real model
        result = self.attr.attribute_safe(bad, self._batch(0))
        self.assertIsNone(result)

    def test_100_rows_all_complete(self):
        """All 100 rows produce valid matrices (running in batches of 4)."""
        ok = 0
        for i in range(0, N_ROWS, 4):
            result = self.attr.attribute(self.model, self._batch(i))
            for task, row in result.items():
                self.assertEqual(set(row.keys()), set(TASK_NAMES))
                self.assertAlmostEqual(sum(row.values()), 1.0, places=5)
            ok += 1
        self.assertEqual(ok, N_ROWS // 4)


# =========================================================
# E2E-2: AggregatorSHAP
# =========================================================

class TestAggregatorSHAPE2E(unittest.TestCase):

    def setUp(self):
        from src.aggregation.neural_aggregator import MLPAggregator
        from src.explainability.aggregator_shap import AggregatorSHAP
        self.agg = MLPAggregator(input_dim=FEAT_DIM, hidden_dim=64, dropout=0.0)
        self.agg.eval()
        self.X   = make_feature_rows(N_ROWS)           # (100, 47)
        self.explainer = AggregatorSHAP(self.agg, background_size=8)
        self.explainer.set_background(self.X[:20])

    def test_explain_single_row_credibility(self):
        """Single row → credibility head has 47 feature importances."""
        result = self.explainer.explain(self.X[0], heads=["credibility"])
        self.assertEqual(len(result.global_explanation), FEAT_DIM)

    def test_feature_names_match_canonical(self):
        from src.explainability.aggregator_shap import _CANONICAL_FEATURE_NAMES
        result = self.explainer.explain(self.X[1], heads=["credibility"])
        names = [fi.feature_name for fi in result.global_explanation]
        self.assertIn("bias_biased", names)
        self.assertIn("cross_bias_x_emotion", names)

    def test_global_explanation_sorted_desc(self):
        """global_explanation is sorted by |shap_value| descending."""
        result = self.explainer.explain(self.X[2], heads=["credibility"])
        vals = [abs(fi.shap_value) for fi in result.global_explanation]
        self.assertEqual(vals, sorted(vals, reverse=True))

    def test_all_heads(self):
        """Explaining all 4 heads produces per_head with 4 keys."""
        result = self.explainer.explain(self.X[3])
        self.assertIn("credibility", result.per_head)
        self.assertIn("risk_low",    result.per_head)
        self.assertIn("risk_medium", result.per_head)
        self.assertIn("risk_high",   result.per_head)
        self.assertIn("credibility", result.base_values)

    def test_to_dict_serialisable(self):
        """AggregatorSHAPResult serialises to a plain dict."""
        result = self.explainer.explain(self.X[4], heads=["credibility"])
        d = result.to_dict()
        self.assertIn("global_explanation", d)
        self.assertIn("per_head", d)
        self.assertIn("base_values", d)
        self.assertIsInstance(d["global_explanation"][0]["shap_value"], float)

    def test_finite_shap_values_on_all_100_rows(self):
        """SHAP values must be finite for all 100 rows."""
        for i in range(0, N_ROWS, 10):
            result = self.explainer.explain(self.X[i], heads=["credibility"])
            for fi in result.global_explanation:
                self.assertTrue(math.isfinite(fi.shap_value),
                    f"row {i}: non-finite SHAP for {fi.feature_name}: {fi.shap_value}")

    def test_batch_explain(self):
        """Batch of 8 rows produces a result with 47 importances."""
        result = self.explainer.explain(self.X[:8], heads=["credibility"])
        self.assertEqual(len(result.global_explanation), FEAT_DIM)


# =========================================================
# E2E-3: AttentionGraphBuilder
# =========================================================

class TestAttentionGraphBuilderE2E(unittest.TestCase):

    def setUp(self):
        from src.explainability.attention_graph import AttentionGraphBuilder
        self.builder = AttentionGraphBuilder(top_k_tokens=8)
        self.tokens = [f"tok_{i}" for i in range(SEQ_LEN)]

    def _make_scores(self) -> Dict[str, List[float]]:
        return {
            task: RNG.random(SEQ_LEN).tolist()
            for task in TASK_NAMES
        }

    def _make_task_weights(self) -> Dict[str, float]:
        w = RNG.random(N_TASKS)
        w = w / w.sum()
        return dict(zip(TASK_NAMES, w.tolist()))

    def test_graph_structure(self):
        """Graph has correct node types and edge types."""
        g = self.builder.build(self.tokens, self._make_scores(), self._make_task_weights())
        self.assertEqual(len(g.token_nodes()),  SEQ_LEN)
        self.assertEqual(len(g.task_nodes()),   N_TASKS)
        self.assertIsNotNone(g.decision_node())

    def test_top_k_token_edges(self):
        """Each task has at most top_k_tokens=8 token→task edges."""
        g = self.builder.build(self.tokens, self._make_scores(), self._make_task_weights())
        for task in TASK_NAMES:
            edges = g.top_token_edges(task)
            self.assertLessEqual(len(edges), 8)

    def test_all_edge_weights_nonneg(self):
        """All edge weights are >= 0."""
        g = self.builder.build(self.tokens, self._make_scores(), self._make_task_weights())
        for e in g.edges:
            self.assertGreaterEqual(e.weight, 0.0)

    def test_to_dict_roundtrip(self):
        """to_dict() produces expected keys."""
        g = self.builder.build(self.tokens, self._make_scores(), self._make_task_weights())
        d = g.to_dict()
        self.assertIn("nodes", d)
        self.assertIn("edges", d)
        self.assertIn("metadata", d)
        for node in d["nodes"]:
            self.assertIn("node_id", node)
            self.assertIn("node_type", node)
            self.assertIn("label", node)

    def test_from_rollout_and_attribution(self):
        """Factory method from_rollout_and_attribution wires cross-task correctly."""
        from src.explainability.attention_graph import AttentionGraphBuilder
        influence = {
            task: {other: 1.0/N_TASKS for other in TASK_NAMES}
            for task in TASK_NAMES
        }
        scores = self._make_scores()
        g = AttentionGraphBuilder.from_rollout_and_attribution(
            tokens=self.tokens,
            rollout_scores=scores,
            cross_task_influence=influence,
        )
        self.assertEqual(len(g.task_nodes()), N_TASKS)

    def test_100_rows_all_valid(self):
        """100 different synthetic graphs all have correct structure."""
        for _ in range(N_ROWS):
            g = self.builder.build(
                self.tokens, self._make_scores(), self._make_task_weights()
            )
            self.assertEqual(len(g.token_nodes()),  SEQ_LEN)
            self.assertEqual(len(g.task_nodes()),   N_TASKS)
            self.assertIsNotNone(g.decision_node())
            d = g.to_dict()
            self.assertIn("nodes", d)
            self.assertIn("edges", d)

    def test_missing_task_scores_uses_zero(self):
        """If a task is missing from token_scores, it gets zeros and still builds."""
        partial = {"bias": RNG.random(SEQ_LEN).tolist()}  # only 1 of 4
        weights = self._make_task_weights()
        g = self.builder.build(self.tokens, partial, weights)
        self.assertEqual(len(g.task_nodes()), N_TASKS)


# =========================================================
# E2E-4: SinglePassAnalyzer
# =========================================================

class TestSinglePassAnalyzerE2E(unittest.TestCase):

    def setUp(self):
        from src.inference.single_pass_analyzer import SinglePassAnalyzer, SinglePassConfig
        from src.explainability.cross_task_attribution import CrossTaskAttributor
        from src.explainability.attention_graph import AttentionGraphBuilder

        self.model    = MockInteractingModel()
        self.model.eval()
        self.tokenizer = MockTokenizer()

        cfg = SinglePassConfig(
            device="cpu",
            max_length=SEQ_LEN,
            return_cross_task=True,
            return_attention_graph=True,
            async_timeout=10.0,
        )
        self.analyzer = SinglePassAnalyzer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=cfg,
            attributor=CrossTaskAttributor(method="gate"),
            graph_builder=AttentionGraphBuilder(top_k_tokens=5),
        )

    def test_analyze_returns_correct_tasks(self):
        result = self.analyzer.analyze("The president denied the allegations.")
        task_names_out = {p.task for p in result.task_predictions}
        self.assertEqual(task_names_out, set(TASK_NAMES))

    def test_credibility_in_unit_interval(self):
        result = self.analyzer.analyze("Breaking: scientists discover water on Mars.")
        self.assertGreaterEqual(result.credibility_score, 0.0)
        self.assertLessEqual(result.credibility_score, 1.0)

    def test_risk_label_valid(self):
        result = self.analyzer.analyze("Fake news spreads on social media.")
        self.assertIn(result.risk_label, ["low", "medium", "high"])
        self.assertEqual(len(result.risk_probabilities), 3)
        self.assertAlmostEqual(sum(result.risk_probabilities), 1.0, places=5)

    def test_cross_task_influence_populated(self):
        result = self.analyzer.analyze("Government officials confirm vaccine mandate.")
        self.assertIsNotNone(result.cross_task_influence)
        for task, row in result.cross_task_influence.items():
            self.assertEqual(set(row.keys()), set(TASK_NAMES))
            self.assertAlmostEqual(sum(row.values()), 1.0, places=5)

    def test_attention_graph_populated(self):
        result = self.analyzer.analyze("Opposition leaders reject the bill.")
        self.assertIsNotNone(result.attention_graph)
        self.assertIn("nodes", result.attention_graph)
        self.assertIn("edges", result.attention_graph)

    def test_latency_positive(self):
        result = self.analyzer.analyze("Short text.")
        self.assertGreater(result.latency_ms, 0.0)

    def test_to_dict_complete(self):
        result = self.analyzer.analyze("Climate scientists issue urgent warning.")
        d = result.to_dict()
        for key in ("text", "task_predictions", "credibility_score",
                    "risk_label", "risk_probabilities", "latency_ms"):
            self.assertIn(key, d)

    def test_task_confidence_in_unit_interval(self):
        result = self.analyzer.analyze("Poll shows record support for incumbents.")
        for pred in result.task_predictions:
            self.assertGreaterEqual(pred.confidence, 0.0)
            self.assertLessEqual(pred.confidence, 1.0)
            self.assertAlmostEqual(sum(pred.probabilities), 1.0, places=5)

    def test_100_texts_all_complete(self):
        """100 synthetic texts all produce valid SinglePassResult objects.

        We run analyze() synchronously which spins an event loop per call;
        limit to 20 texts to keep the test under the CI timeout while still
        exercising enough diversity (text length varies from 3 to 12 words).
        """
        N_SAMPLE = 20
        texts = [
            f"Article {i}: {'word ' * (i % 10 + 3)}news story about event {i}."
            for i in range(N_SAMPLE)
        ]
        ok = 0
        for text in texts:
            result = self.analyzer.analyze(text)
            self.assertIn(result.risk_label, ["low", "medium", "high"])
            self.assertGreaterEqual(result.credibility_score, 0.0)
            self.assertLessEqual(result.credibility_score, 1.0)
            self.assertEqual(len(result.task_predictions), N_TASKS)
            ok += 1
        self.assertEqual(ok, N_SAMPLE)


# =========================================================
# E2E-5: PCGradOptimizer
# =========================================================

class TestPCGradOptimizerE2E(unittest.TestCase):

    def setUp(self):
        from src.training.pcgrad import PCGradOptimizer
        self.model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )
        base = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.opt = PCGradOptimizer(base, gate_threshold=0.15)

    def _losses(self, x):
        out = self.model(x)                                  # (B, 8)
        return {
            "bias":       out[:, :2].sum(),
            "emotion":    out[:, 2:4].mean(),
            "ideology":   out[:, 4:6].norm(),
            "propaganda": out[:, 6:].std(),
        }

    def test_pc_backward_sets_all_grads(self):
        """After pc_backward every shared parameter has a grad."""
        x = torch.randn(8, 16)
        self.opt.zero_grad()
        losses = self._losses(x)
        self.opt.pc_backward(losses)
        for p in self.model.parameters():
            self.assertIsNotNone(p.grad, "Grad missing after pc_backward")

    def test_step_changes_params(self):
        """params change after opt.step()."""
        x    = torch.randn(8, 16)
        snap = [p.data.clone() for p in self.model.parameters()]
        self.opt.zero_grad()
        losses = self._losses(x)
        self.opt.pc_backward(losses)
        self.opt.step()
        for p, old in zip(self.model.parameters(), snap):
            self.assertFalse(torch.allclose(p.data, old),
                "Parameter unchanged after step — optimizer may be broken")

    def test_task_gating_excludes_low_confidence_task(self):
        """A task with gate_factor < gate_threshold must not appear in the grad."""
        # Run with all 4 tasks, record grad magnitude for 'propaganda'
        x = torch.randn(4, 16)
        self.opt.zero_grad()
        self.opt.set_gate_factors({t: 1.0 for t in TASK_NAMES})
        losses = self._losses(x)
        self.opt.pc_backward(losses)
        grad_all = self.model[0].weight.grad.clone()

        # Now gate propaganda out
        self.opt.zero_grad()
        self.opt.set_gate_factors({"bias": 0.9, "emotion": 0.8, "ideology": 0.7,
                                   "propaganda": 0.05})   # 0.05 < 0.15 → gated out
        losses2 = self._losses(x)
        self.opt.pc_backward(losses2)
        grad_gated = self.model[0].weight.grad.clone()

        # Gradients should differ when a task is gated out
        self.assertFalse(torch.allclose(grad_all, grad_gated))

    def test_gating_all_out_warns_but_no_grad(self):
        """If all tasks are gated out pc_backward returns without setting grads."""
        self.opt.zero_grad()
        self.opt.set_gate_factors({t: 0.0 for t in TASK_NAMES})
        x = torch.randn(4, 16)
        losses = self._losses(x)
        import logging
        with self.assertLogs("src.training.pcgrad", level="WARNING") as cm:
            self.opt.pc_backward(losses)
        self.assertTrue(any("gated out" in m for m in cm.output))
        for p in self.model.parameters():
            if p.grad is not None:
                self.assertTrue(
                    (p.grad == 0).all() or p.grad is None,
                    "Expected zero/no grad when all tasks gated out"
                )

    def test_100_training_steps(self):
        """100 full training steps complete without NaN/Inf gradients."""
        X = torch.randn(N_ROWS, 16)
        for step in range(N_ROWS):
            x = X[step:step+4]
            self.opt.zero_grad()
            self.opt.set_gate_factors({t: 1.0 for t in TASK_NAMES})
            losses = self._losses(x)
            self.opt.pc_backward(losses)
            # Check no NaN/Inf in gradients
            for p in self.model.parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all(),
                        f"Non-finite grad at step {step}")
            self.opt.step()

    def test_project_conflicting_zeroes_opposite_vectors(self):
        """Core PCGrad math: g_i · g_j < 0 → project out the conflicting component."""
        from src.training.pcgrad import _project_conflicting, _pcgrad_project
        g1 = torch.tensor([1.0, 0.0])
        g2 = torch.tensor([-1.0, 0.0])
        out = _project_conflicting(g1, g2)
        self.assertTrue(torch.allclose(out, torch.zeros(2), atol=1e-5))

    def test_pcgrad_project_averages_nonconflicting(self):
        from src.training.pcgrad import _pcgrad_project
        g_a = torch.tensor([2.0, 0.0])
        g_b = torch.tensor([0.0, 2.0])
        proj = _pcgrad_project([g_a, g_b])
        self.assertTrue(torch.allclose(proj, torch.tensor([1.0, 1.0]), atol=1e-5))


# =========================================================
# E2E-6: ExplainabilityResult schema round-trip
# =========================================================

class TestExplainabilityResultSchemaE2E(unittest.TestCase):

    def _influence(self):
        return {
            task: {other: 1.0/N_TASKS for other in TASK_NAMES}
            for task in TASK_NAMES
        }

    def _global_explanation(self):
        from src.explainability.aggregator_shap import _CANONICAL_FEATURE_NAMES
        return [
            {"feature_name": name, "feature_index": i,
             "shap_value": float(RNG.random() - 0.5), "head": "credibility"}
            for i, name in enumerate(_CANONICAL_FEATURE_NAMES)
        ]

    def _graph(self):
        from src.explainability.attention_graph import AttentionGraphBuilder
        b = AttentionGraphBuilder()
        g = b.build(
            tokens=[f"tok_{i}" for i in range(8)],
            token_scores={t: [0.1]*8 for t in TASK_NAMES},
            task_to_decision={t: 0.25 for t in TASK_NAMES},
        )
        return g.to_dict()

    def test_all_v2_fields_populate(self):
        from src.explainability.common_schema import ExplainabilityResult
        r = ExplainabilityResult(
            prediction={"label": 1, "confidence": 0.87},
            cross_task_influence=self._influence(),
            global_explanation=self._global_explanation(),
            attention_graph=self._graph(),
        )
        self.assertIsNotNone(r.cross_task_influence)
        self.assertIsNotNone(r.global_explanation)
        self.assertIsNotNone(r.attention_graph)
        self.assertEqual(len(r.global_explanation), 47)

    def test_backward_compat_fields_still_present(self):
        from src.explainability.common_schema import ExplainabilityResult
        r = ExplainabilityResult(prediction={"label": 0})
        self.assertIsNone(r.shap_explanation)
        self.assertIsNone(r.lime_explanation)
        self.assertIsNone(r.cross_task_influence)
        self.assertIsNone(r.global_explanation)
        self.assertIsNone(r.attention_graph)
        self.assertEqual(r.module_failures, [])

    def test_extra_fields_ignored(self):
        from src.explainability.common_schema import ExplainabilityResult
        r = ExplainabilityResult(
            prediction={"label": 0},
            totally_new_unknown_field="should be ignored",
        )
        self.assertFalse(hasattr(r, "totally_new_unknown_field"))

    def test_100_rows_schema_round_trip(self):
        """100 rows: build → populate → validate with all v2 fields."""
        from src.explainability.common_schema import ExplainabilityResult
        for i in range(N_ROWS):
            r = ExplainabilityResult(
                prediction={"label": int(RNG.integers(0, 2)), "confidence": float(RNG.random())},
                cross_task_influence=self._influence(),
                global_explanation=self._global_explanation(),
                attention_graph=self._graph(),
                module_failures=[] if i % 10 != 0 else ["shap"],
            )
            self.assertIsNotNone(r.cross_task_influence)
            self.assertEqual(len(r.global_explanation), 47)
            self.assertIn("nodes", r.attention_graph)


# =========================================================
# E2E-7: Full pipeline — attribute → graph → schema (100 rows)
# =========================================================

class TestFullPipelineE2E(unittest.TestCase):
    """Run all three v2 explainability steps in sequence on 100 rows."""

    def setUp(self):
        from src.explainability.cross_task_attribution import CrossTaskAttributor
        from src.explainability.attention_graph import AttentionGraphBuilder
        from src.explainability.common_schema import ExplainabilityResult

        self.model   = MockInteractingModel()
        self.model.eval()
        self.ids     = make_token_ids()
        self.masks   = make_attention_mask()
        self.attr    = CrossTaskAttributor(method="gate")
        self.builder = AttentionGraphBuilder(top_k_tokens=6)
        self.tokens  = [f"w{j}" for j in range(SEQ_LEN)]
        self.Result  = ExplainabilityResult

    def test_100_rows_full_pipeline(self):
        """100 rows: attribution → graph → ExplainabilityResult, all valid."""
        from src.explainability.aggregator_shap import _CANONICAL_FEATURE_NAMES
        failures = 0

        for i in range(0, N_ROWS, 4):
            batch = {
                "input_ids":      self.ids[i:i+4],
                "attention_mask": self.masks[i:i+4],
            }

            # Step 1: cross-task attribution
            try:
                influence = self.attr.attribute(self.model, batch)
            except Exception as e:
                failures += 1
                continue

            # Validate matrix
            for task, row in influence.items():
                self.assertAlmostEqual(sum(row.values()), 1.0, places=5,
                    msg=f"row {i}: row sum for {task}")

            # Step 2: attention graph
            rollout_scores = {
                task: RNG.random(SEQ_LEN).tolist()
                for task in TASK_NAMES
            }
            g = self.builder.build(
                tokens=self.tokens,
                token_scores=rollout_scores,
                task_to_decision={
                    task: sum(influence[task].values()) for task in TASK_NAMES
                },
            )
            graph_dict = g.to_dict()
            self.assertEqual(len(g.token_nodes()), SEQ_LEN)
            self.assertEqual(len(g.task_nodes()),  N_TASKS)

            # Step 3: populate ExplainabilityResult
            fake_global = [
                {"feature_name": _CANONICAL_FEATURE_NAMES[k],
                 "feature_index": k,
                 "shap_value": float(RNG.random() - 0.5),
                 "head": "credibility"}
                for k in range(47)
            ]
            result = self.Result(
                prediction={"label": int(RNG.integers(0,2))},
                cross_task_influence=influence,
                global_explanation=fake_global,
                attention_graph=graph_dict,
            )
            self.assertIsNotNone(result.cross_task_influence)
            self.assertEqual(len(result.global_explanation), 47)
            self.assertIn("nodes", result.attention_graph)

        self.assertEqual(failures, 0, f"{failures} pipeline rows failed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
