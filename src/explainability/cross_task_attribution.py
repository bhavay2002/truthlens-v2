"""
CrossTaskAttributor — Explainability v2 §1 (Cross-task attribution).

Two methods are provided:

Method B — Gate-based (default, no autograd required)
    Reads the sigmoid gate values G_i (B, D) from CrossTaskInteractionLayer
    and the attention weights attn_i (B, 1, T) produced during a forward
    pass.  For each target task i, the influence of source task j is:

        influence(j → i) = mean_over_batch( attn_i[:, 0, j] * mean(G_i) )

    This is cheap, deterministic, and runs under torch.no_grad.

Method A — Gradient-based (opt-in via method="gradient")
    Computes the Jacobian of each task's latent representation with
    respect to all other tasks' latent representations via autograd.
    Requires a model forward with retain_graph=True and grad-enabled.

Public API
----------
    from src.explainability.cross_task_attribution import CrossTaskAttributor

    attributor = CrossTaskAttributor()
    influence = attributor.attribute(
        model,
        inputs,
        method="gate",   # or "gradient"
    )
    # influence: Dict[str, Dict[str, float]]
    #   influence["bias"]["propaganda"] = 0.32
    #   means propaganda influenced bias with weight 0.32
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

import torch

logger = logging.getLogger(__name__)

_EPS = 1e-12

InfluenceMatrix = Dict[str, Dict[str, float]]


# =========================================================
# HOOK CONTAINERS
# =========================================================

class _InteractionCapture:
    """Captures attention weights and gate tensors from a forward pass."""

    def __init__(self) -> None:
        self.attn_weights: List[torch.Tensor] = []   # per-task (B, 1, T)
        self.gate_values: List[torch.Tensor] = []    # per-task (B, D)
        self._handles: list = []

    def _make_attn_hook(self, idx: int):
        def hook(module, args, output):  # noqa: ARG001
            self.attn_weights.append(output.detach().cpu())
        return hook

    def register(self, interaction_layer) -> "_InteractionCapture":
        """Attach hooks to a CrossTaskInteractionLayer."""
        self.clear()
        for i, q_proj in enumerate(interaction_layer.q_projs):
            h = q_proj.register_forward_hook(self._make_qproj_hook(i, interaction_layer))
            self._handles.append(h)
        return self

    def _make_qproj_hook(self, idx: int, interaction_layer):
        """Hook on q_proj to intercept attention weights computed inside forward."""
        def hook(module, args, output):  # noqa: ARG001
            pass
        return hook

    def clear(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.attn_weights.clear()
        self.gate_values.clear()


# =========================================================
# GATE-BASED ATTRIBUTION  (Method B)
# =========================================================

def _gate_based_attribution(
    model,
    inputs: Dict[str, torch.Tensor],
) -> InfluenceMatrix:
    """
    Re-run the interaction layer under no_grad with manual extraction.

    We call the CrossTaskInteractionLayer directly, intercepting:
      - attn[:, 0, j]  — how much task i attends to task j's K/V
      - gate values G_i — element-wise gate strength for task i

    The combined influence of task j on task i is:

        influence(j → i) = E_batch[ attn_i(j) ] * E_batch[ mean(G_i) ]

    Both factors are in [0, 1], so the product is also in [0, 1].
    We L1-normalise each row so the influence weights sum to 1.
    """
    interaction = getattr(model, "interaction", None)
    if interaction is None:
        raise AttributeError(
            "Model has no 'interaction' attribute. "
            "Gate-based attribution requires InteractingMultiTaskModel."
        )

    task_names: List[str] = getattr(model, "_task_names", None)
    if task_names is None:
        raise AttributeError("Model missing '_task_names'.")

    num_tasks = len(task_names)

    with torch.no_grad():
        encoder_inputs = {k: inputs[k] for k in ("input_ids", "attention_mask") if k in inputs}
        encoder_outputs = model.encoder(**encoder_inputs)
        sequence_output = encoder_outputs["sequence_output"]
        attention_mask = inputs["attention_mask"]

        H_shared = model.multi_view_pooling(sequence_output, attention_mask)

        B = H_shared.size(0)
        device = H_shared.device

        task_reprs: Dict[str, torch.Tensor] = {}
        for task in task_names:
            H_i = model.task_projections[task](H_shared)
            idx = model._task_to_idx[task]
            H_i = H_i + model.task_embed(idx, B, device)
            task_reprs[task] = H_i

        ordered_reprs = [task_reprs[t] for t in task_names]
        H_all = torch.stack(ordered_reprs, dim=1)          # (B, T, D)
        K = interaction.k_proj(H_all)                      # (B, T, D)
        V = interaction.v_proj(H_all)                      # noqa: F841

        influence: InfluenceMatrix = {t: {} for t in task_names}

        for i, (task_i, q_proj, gate_proj) in enumerate(
            zip(task_names, interaction.q_projs, interaction.gates)
        ):
            H_i = ordered_reprs[i]
            Q_i = q_proj(H_i).unsqueeze(1)                # (B, 1, D)
            attn = torch.bmm(Q_i, K.transpose(1, 2)) / interaction.scale  # (B, 1, T)
            attn = torch.softmax(attn, dim=-1)             # (B, 1, T)

            gate_strength = torch.sigmoid(gate_proj(H_i)).mean(dim=-1)  # (B,)
            mean_gate = gate_strength.mean().item()

            for j, task_j in enumerate(task_names):
                raw_attn = attn[:, 0, j].mean().item()
                influence[task_i][task_j] = raw_attn * (mean_gate if i != j else 1.0)

        influence = _row_normalise(influence, task_names)

    return influence


# =========================================================
# GRADIENT-BASED ATTRIBUTION  (Method A)
# =========================================================

def _gradient_based_attribution(
    model,
    inputs: Dict[str, torch.Tensor],
) -> InfluenceMatrix:
    """
    Gradient-based cross-task attribution.

    For each target task i, we compute the L2 norm of the Jacobian
    of H_i' with respect to H_j (the pre-interaction representation
    of task j), summed over the batch.

        influence(j → i) = ||∂H_i' / ∂H_j||_F  (Frobenius)

    This is more expensive than gate-based attribution but model-faithful.
    """
    task_names: List[str] = getattr(model, "_task_names", None)
    if task_names is None:
        raise AttributeError("Model missing '_task_names'.")

    encoder_inputs = {k: inputs[k] for k in ("input_ids", "attention_mask") if k in inputs}

    with torch.enable_grad():
        encoder_outputs = model.encoder(**encoder_inputs)
        sequence_output = encoder_outputs["sequence_output"]
        attention_mask = inputs["attention_mask"]

        H_shared = model.multi_view_pooling(sequence_output, attention_mask)
        B = H_shared.size(0)
        device = H_shared.device

        task_reprs_pre: Dict[str, torch.Tensor] = {}
        for task in task_names:
            H_i = model.task_projections[task](H_shared)
            idx = model._task_to_idx[task]
            H_i = H_i + model.task_embed(idx, B, device)
            H_i = H_i.requires_grad_(True)
            task_reprs_pre[task] = H_i

        ordered_pre = [task_reprs_pre[t] for t in task_names]
        interacted = model.interaction(ordered_pre)

        influence: InfluenceMatrix = {t: {} for t in task_names}

        for i, task_i in enumerate(task_names):
            H_i_prime = interacted[i]                       # (B, D)
            scalar = H_i_prime.sum()

            grads = torch.autograd.grad(
                scalar, ordered_pre,
                retain_graph=True,
                allow_unused=True,
            )

            for j, task_j in enumerate(task_names):
                g = grads[j]
                if g is None:
                    influence[task_i][task_j] = 0.0
                else:
                    influence[task_i][task_j] = float(g.norm(p="fro").item())

        influence = _row_normalise(influence, task_names)

    return influence


# =========================================================
# NORMALISATION HELPER
# =========================================================

def _row_normalise(matrix: InfluenceMatrix, task_names: List[str]) -> InfluenceMatrix:
    """L1-normalise each row so influence weights sum to 1 per target task."""
    out: InfluenceMatrix = {}
    for task_i in task_names:
        row = matrix[task_i]
        total = sum(abs(v) for v in row.values()) + _EPS
        out[task_i] = {task_j: v / total for task_j, v in row.items()}
    return out


# =========================================================
# PUBLIC ATTRIBUTOR
# =========================================================

class CrossTaskAttributor:
    """Compute a cross-task influence matrix from a trained InteractingMultiTaskModel.

    Parameters
    ----------
    method : "gate" | "gradient"
        Attribution method. "gate" is cheap (no autograd); "gradient" is
        model-faithful but requires enabled gradients and retain_graph.
    """

    def __init__(self, method: Literal["gate", "gradient"] = "gate") -> None:
        self.method = method

    def attribute(
        self,
        model: Any,
        inputs: Dict[str, torch.Tensor],
        *,
        method: Optional[Literal["gate", "gradient"]] = None,
    ) -> InfluenceMatrix:
        """Compute the cross-task influence matrix.

        Parameters
        ----------
        model  : InteractingMultiTaskModel (or any model with .interaction,
                 .multi_view_pooling, .task_projections, .task_embed attributes)
        inputs : tokenizer output dict with at least "input_ids", "attention_mask"
        method : override the instance default if provided

        Returns
        -------
        InfluenceMatrix : Dict[target_task, Dict[source_task, float]]
            influence["bias"]["propaganda"] = 0.32 means propaganda
            influenced the bias representation with normalised weight 0.32.
        """
        m = method or self.method
        try:
            if m == "gate":
                return _gate_based_attribution(model, inputs)
            elif m == "gradient":
                return _gradient_based_attribution(model, inputs)
            else:
                raise ValueError(f"Unknown method {m!r}. Choose 'gate' or 'gradient'.")
        except Exception as exc:
            logger.exception("CrossTaskAttributor.attribute failed (method=%s)", m)
            raise RuntimeError(f"Cross-task attribution failed: {exc}") from exc

    def attribute_safe(
        self,
        model: Any,
        inputs: Dict[str, torch.Tensor],
        *,
        method: Optional[Literal["gate", "gradient"]] = None,
        fallback: Optional[InfluenceMatrix] = None,
    ) -> Optional[InfluenceMatrix]:
        """Like attribute() but returns ``fallback`` on failure instead of raising."""
        try:
            return self.attribute(model, inputs, method=method)
        except Exception as exc:
            logger.warning("CrossTaskAttributor.attribute_safe: %s", exc)
            return fallback
