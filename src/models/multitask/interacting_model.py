"""
InteractingMultiTaskModel — cross-task interacting heads with a unified
latent truth representation.

Architecture
------------

    Input Text
       ↓
    TransformerEncoder  (RoBERTa)
       ↓
    MultiViewPooling   (CLS + mean + attention → Linear → H_shared)
       ↓
    TaskProjectionBlock × T  (H_shared → H_i  per task)
    TaskEmbeddings           (H_i  += E_task[i])
       ↓
    CrossTaskInteractionLayer  (gated cross-attention; H_i attends all H_j)
       ↓
    Task-Specific Heads  (existing ClassificationHead / MultiLabelHead)
       ↓
    LatentFusionHead   (concat(H_i') → LayerNorm → Linear → GELU → Z)
       ↓
    credibility_score  (sigmoid(Linear(Z)))

Forward output
--------------
{
    "<task>": {"logits": Tensor, ...},   # per-task head dicts
    "task_logits": {<task>: Tensor},     # LossEngine / training loop compat
    "latent_vector": Tensor (B, D),      # fused latent representation
    "credibility_score": Tensor (B,),    # sigmoid credibility in [0, 1]
}

Compatibility guarantees
------------------------
* Inherits MultiTaskTruthLensModel so self.encoder, self.task_heads,
  self.heads (view), get_input_embeddings(), freeze_encoder(), etc.
  all work without any changes to the training loop or inference layer.
* Output dict is a strict superset of MultiTaskTruthLensModel's output —
  existing consumers that only read "task_logits" or per-task entries are
  unaffected.
* Fully bf16 / fp16 mixed-precision safe (no float-only ops in forward).
* CPU fallback: no CUDA-specific primitives.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.multitask.multitask_truthlens_model import (
    MultiTaskTruthLensModel,
    MultiTaskTruthLensConfig,
    _DEFAULT_TASK_SPEC,
)

logger = logging.getLogger(__name__)


# =========================================================
# 1. MULTI-VIEW POOLING
# =========================================================

class MultiViewPooling(nn.Module):
    """Three-way pooled sentence representation.

    Combines CLS, masked mean-pooling, and learned attention pooling into
    a single (B, D) vector that captures global, average, and key-token
    semantics simultaneously.

    CLS alone is insufficient for long documents and narrative / discourse
    tasks — mean + attention pooling supply the complementary signal.

    Parameters
    ----------
    hidden_size:
        Encoder hidden dimension (D).
    dropout:
        Applied after the projection layer.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.attn_weight = nn.Linear(hidden_size, 1, bias=False)
        self.proj = nn.Linear(hidden_size * 3, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        sequence_output: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        sequence_output : (B, L, D)
        attention_mask  : (B, L)  — 1 for real tokens, 0 for padding

        Returns
        -------
        (B, D) pooled representation.
        """
        # ── CLS ──────────────────────────────────────────────────────
        h_cls = sequence_output[:, 0]                           # (B, D)

        # ── Masked mean ──────────────────────────────────────────────
        mask = attention_mask.unsqueeze(-1).to(sequence_output.dtype)
        h_mean = (sequence_output * mask).sum(1) / mask.sum(1).clamp_min(1e-9)

        # ── Learned attention pooling ─────────────────────────────────
        scores = self.attn_weight(sequence_output).squeeze(-1)  # (B, L)
        # mask out padding tokens before softmax so they get ~0 weight
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)                 # (B, L)
        h_attn = (sequence_output * weights.unsqueeze(-1)).sum(1)  # (B, D)

        # ── Concat + project ─────────────────────────────────────────
        h = torch.cat([h_cls, h_mean, h_attn], dim=-1)         # (B, 3D)
        h = self.drop(self.norm(self.proj(h)))                  # (B, D)

        return h


# =========================================================
# 2. TASK PROJECTION BLOCK
# =========================================================

class TaskProjectionBlock(nn.Module):
    """Per-task linear projection from the shared representation.

    H_i = LayerNorm( Dropout( GELU( Linear(H_shared) ) ) )

    Each task gets its own projection so the shared encoder's output can
    be specialised per-task BEFORE the cross-task interaction step. This
    avoids all tasks competing for exactly the same feature directions.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, D) → (B, D)"""
        return self.norm(self.drop(self.act(self.proj(x))))


# =========================================================
# 3. TASK EMBEDDINGS
# =========================================================

class TaskEmbeddings(nn.Module):
    """Learnable task identity embeddings injected into each task representation.

    Encodes *task identity* explicitly in embedding space so the model can
    learn which tasks are semantically related (e.g. bias ↔ ideology close,
    propaganda ↔ narrative close).

    Usage::
        emb = TaskEmbeddings(num_tasks=5, hidden_size=768)
        H_bias = H_bias + emb(task_idx=0)  # (B, D) + (D,) broadcast
    """

    def __init__(self, num_tasks: int, hidden_size: int) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(num_tasks, hidden_size)
        # Small-std init — similar to BERT token-type embeddings
        nn.init.normal_(self.embeddings.weight, std=0.02)

    def forward(self, task_idx: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return (B, D) task embedding for broadcast addition."""
        idx = torch.tensor(task_idx, device=device)
        return self.embeddings(idx).unsqueeze(0).expand(batch_size, -1)


# =========================================================
# 4. CROSS-TASK INTERACTION LAYER
# =========================================================

class CrossTaskInteractionLayer(nn.Module):
    """Gated multi-task cross-attention.

    For each task i, computes a cross-attention context from all task
    representations and gates how much of that context is added back:

        Q_i   = W_Q_i(H_i)              shape: (B, D)
        K     = W_K(stack(H_j))         shape: (B, T, D)
        V     = W_V(stack(H_j))         shape: (B, T, D)
        attn  = softmax(Q_i·Kᵀ / √D)   shape: (B, T)
        ctx_i = attn · V                shape: (B, D)
        g_i   = sigmoid(W_g_i(H_i))     shape: (B, D)
        H_i'  = LayerNorm(H_i + Dropout(g_i ⊙ ctx_i))

    Key properties
    --------------
    * Residual connection prevents gradient vanishing / explosion.
    * Per-task gating allows each task to independently control how much
      it borrows from other tasks, rather than applying a global blend.
    * LayerNorm after each update stabilises activation scale.
    * Shared K/V projections reduce parameter count while still letting
      each task query with its own projection.
    """

    def __init__(
        self,
        hidden_size: int,
        num_tasks: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.scale = math.sqrt(hidden_size)

        # Per-task query projections
        self.q_projs = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_tasks)]
        )

        # Shared key / value projections (all task representations → K, V)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Per-task scalar+vector gate (learned)
        self.gates = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_tasks)]
        )

        # Per-task post-interaction layer norms
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(num_tasks)]
        )

        self.drop = nn.Dropout(dropout)

        for q in self.q_projs:
            nn.init.xavier_uniform_(q.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        for g in self.gates:
            nn.init.xavier_uniform_(g.weight)
            nn.init.zeros_(g.bias)

    def forward(self, task_reprs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Parameters
        ----------
        task_reprs : List of T tensors, each (B, D)

        Returns
        -------
        List of T tensors, each (B, D) — refined representations.
        """
        # Stack all task representations: (B, T, D)
        H_all = torch.stack(task_reprs, dim=1)              # (B, T, D)

        K = self.k_proj(H_all)                              # (B, T, D)
        V = self.v_proj(H_all)                              # (B, T, D)

        outputs: List[torch.Tensor] = []

        for i, (H_i, q_proj, gate_proj, norm) in enumerate(
            zip(task_reprs, self.q_projs, self.gates, self.norms)
        ):
            # Query for task i: (B, 1, D)
            Q_i = q_proj(H_i).unsqueeze(1)

            # Scaled dot-product attention: (B, 1, T)
            attn = torch.bmm(Q_i, K.transpose(1, 2)) / self.scale
            attn = torch.softmax(attn, dim=-1)              # (B, 1, T)

            # Context vector: (B, D)
            ctx = torch.bmm(attn, V).squeeze(1)            # (B, D)
            ctx = self.drop(ctx)

            # Gated residual update
            g = torch.sigmoid(gate_proj(H_i))              # (B, D)
            H_i_out = norm(H_i + g * ctx)

            outputs.append(H_i_out)

        return outputs


# =========================================================
# 5. LATENT FUSION HEAD
# =========================================================

class LatentFusionHead(nn.Module):
    """Unified latent truth representation + credibility score.

    Concatenates all per-task representations, projects them into a
    shared D-dimensional latent space, then produces a scalar credibility
    score via sigmoid:

        Z_cat   = concat([H_bias', H_emotion', ...])  (B, T·D)
        Z       = Dropout(GELU(Linear(LayerNorm(Z_cat))))  (B, D)
        score   = sigmoid(Linear(Z, 1))               (B,)

    Returns both Z (for downstream use, e.g. calibration) and the score.
    """

    def __init__(
        self,
        hidden_size: int,
        num_tasks: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        concat_dim = hidden_size * num_tasks

        self.input_norm = nn.LayerNorm(concat_dim)
        self.proj = nn.Linear(concat_dim, hidden_size)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.score_head = nn.Linear(hidden_size, 1)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        # Score head: zero bias → 0.5 initial credibility (sigmoid(0))
        nn.init.xavier_uniform_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)

    def forward(
        self,
        task_reprs: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        task_reprs : List of T tensors, each (B, D)

        Returns
        -------
        Z     : (B, D) latent vector
        score : (B,)  credibility score in [0, 1]
        """
        Z_cat = torch.cat(task_reprs, dim=-1)           # (B, T·D)
        Z_cat = self.input_norm(Z_cat)

        Z = self.proj(Z_cat)                            # (B, D)
        Z = self.act(Z)
        Z = self.drop(Z)

        score = torch.sigmoid(self.score_head(Z)).squeeze(-1)  # (B,)

        return Z, score


# =========================================================
# 6. CONFIG
# =========================================================

@dataclass
class InteractingMultiTaskConfig:
    """Config for ``InteractingMultiTaskModel``.

    Extends ``MultiTaskTruthLensConfig`` semantics with the new components
    introduced by the cross-task architecture.
    """

    # ── Encoder ──────────────────────────────────────────────────────
    model_name: str = "roberta-base"
    dropout: float = 0.1
    device: Optional[str] = None
    init_from_config_only: bool = False

    # ── Task selection ───────────────────────────────────────────────
    task_num_labels: Optional[Dict[str, int]] = None
    enabled_tasks: Optional[List[str]] = None

    # ── Per-task loss weights ────────────────────────────────────────
    bias_weight: float = 1.0
    ideology_weight: float = 1.0
    propaganda_weight: float = 1.0
    narrative_weight: float = 1.0
    emotion_weight: float = 1.0

    # ── Cross-task interaction ───────────────────────────────────────
    # Number of tasks that can attend to each other in the interaction
    # layer. Inferred automatically from enabled_tasks; override only
    # for custom task sets.
    num_tasks: Optional[int] = None

    # ── Hybrid aggregation ───────────────────────────────────────────
    # When hybrid_alpha is not None, the credibility score returned by the
    # neural fusion head is blended with a rule-based score passed in at
    # inference time:
    #     final = α * neural + (1 - α) * rule
    # Set to None to use the pure neural score.
    hybrid_alpha: Optional[float] = None

    # Reserved
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


# =========================================================
# 7. INTERACTING MULTI-TASK MODEL
# =========================================================

class InteractingMultiTaskModel(MultiTaskTruthLensModel):
    """Cross-task interacting model with a unified latent truth vector.

    Extends ``MultiTaskTruthLensModel`` with:

    1. ``MultiViewPooling`` — replaces CLS-only pooling with a three-way
       (CLS + mean + attention) pooled representation.

    2. Per-task ``TaskProjectionBlock`` + ``TaskEmbeddings`` — each task
       projects the shared representation into its own subspace and
       receives a learnable task-identity embedding.

    3. ``CrossTaskInteractionLayer`` — gated cross-attention lets each
       task borrow relevant signal from all other tasks' intermediate
       representations before the final heads are applied.

    4. ``LatentFusionHead`` — concatenates all refined task
       representations into a unified latent vector Z and produces a
       scalar credibility score.

    Backward compatibility
    ----------------------
    * All ``MultiTaskTruthLensModel`` attributes (``self.encoder``,
      ``self.task_heads``, ``self.heads``, ``get_input_embeddings``,
      ``freeze_encoder``, …) are inherited unchanged.
    * The forward output is a strict superset — existing code that reads
      only ``outputs["task_logits"]`` or per-task entries keeps working.
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        task_heads: Optional[Dict[str, nn.Module]] = None,
        *,
        config: Optional[InteractingMultiTaskConfig] = None,
        dropout: float = 0.1,
    ) -> None:

        # -----------------------------------------------------------------
        # Config-construction path: build encoder + heads from the config
        # dataclass, then fall through to the raw-modules path.
        # -----------------------------------------------------------------
        if config is not None:
            if encoder is not None or task_heads is not None:
                raise ValueError(
                    "Pass either (encoder, task_heads) or `config=`, not both."
                )

            # Re-use MultiTaskTruthLensModel's build helpers by constructing
            # an equivalent MultiTaskTruthLensConfig and delegating.
            base_cfg = MultiTaskTruthLensConfig(
                model_name=config.model_name,
                pooling="cls",          # overridden by MultiViewPooling below
                dropout=config.dropout,
                device=config.device,
                init_from_config_only=config.init_from_config_only,
                bias_weight=config.bias_weight,
                ideology_weight=config.ideology_weight,
                propaganda_weight=config.propaganda_weight,
                narrative_weight=config.narrative_weight,
                emotion_weight=config.emotion_weight,
                task_num_labels=config.task_num_labels,
                enabled_tasks=config.enabled_tasks,
            )
            encoder, task_heads = MultiTaskTruthLensModel._build_from_truthlens_config(
                base_cfg
            )
            dropout = config.dropout
            self._hybrid_alpha: Optional[float] = config.hybrid_alpha
        else:
            self._hybrid_alpha = None

        # -----------------------------------------------------------------
        # Parent init: registers self.encoder + self.task_heads + label
        # class attributes (BIAS_LABELS, etc.). We pass config=None so the
        # parent does NOT re-run its config-construction path.
        # -----------------------------------------------------------------
        super().__init__(encoder=encoder, task_heads=task_heads)

        hidden_size: int = int(getattr(encoder, "hidden_size", 768))
        task_names: List[str] = list(self.task_heads.keys())
        num_tasks: int = len(task_names)

        # Store task ordering so forward() can map names → interaction indices
        self._task_names: List[str] = task_names

        # -----------------------------------------------------------------
        # New components
        # -----------------------------------------------------------------

        self.multi_view_pooling = MultiViewPooling(hidden_size, dropout)

        self.task_projections = nn.ModuleDict(
            {task: TaskProjectionBlock(hidden_size, dropout) for task in task_names}
        )

        self.task_embed = TaskEmbeddings(num_tasks, hidden_size)

        # Map task name → embedding index (deterministic, insertion order)
        self._task_to_idx: Dict[str, int] = {t: i for i, t in enumerate(task_names)}

        self.interaction = CrossTaskInteractionLayer(hidden_size, num_tasks, dropout)

        self.fusion = LatentFusionHead(hidden_size, num_tasks, dropout)

        logger.info(
            "InteractingMultiTaskModel initialized | tasks=%s | hidden=%d",
            task_names,
            hidden_size,
        )

    # =====================================================================
    # FORWARD
    # =====================================================================

    def forward(self, **inputs: Any) -> Dict[str, Any]:
        """Multi-task forward with cross-task interaction.

        Parameters
        ----------
        inputs:
            Batch dict from the collate function. Must contain
            ``input_ids`` and ``attention_mask``; other keys are ignored
            by the encoder boundary.

        Returns
        -------
        dict with:
            ``"<task>"``         : per-task head output dict (has "logits")
            ``"task_logits"``    : {task: logits_tensor}
            ``"latent_vector"``  : (B, D) unified latent representation Z
            ``"credibility_score"``: (B,) sigmoid credibility in [0, 1]
        """

        # ── 1. Encode ────────────────────────────────────────────────────
        encoder_inputs = {
            k: inputs[k]
            for k in ("input_ids", "attention_mask")
            if k in inputs
        }
        encoder_outputs = self.encoder(**encoder_inputs)

        # Both keys are guaranteed by TransformerEncoder._EncoderOutput
        sequence_output: torch.Tensor = encoder_outputs["sequence_output"]  # (B, L, D)
        attention_mask: torch.Tensor = inputs["attention_mask"]             # (B, L)

        # ── 2. Multi-view pooling ────────────────────────────────────────
        H_shared = self.multi_view_pooling(sequence_output, attention_mask)  # (B, D)

        # ── 3. Per-task projection + task embedding ───────────────────────
        B = H_shared.size(0)
        device = H_shared.device

        task_reprs: Dict[str, torch.Tensor] = {}
        for task in self._task_names:
            H_i = self.task_projections[task](H_shared)         # (B, D)
            idx = self._task_to_idx[task]
            H_i = H_i + self.task_embed(idx, B, device)         # (B, D)
            task_reprs[task] = H_i

        # ── 4. Cross-task interaction ─────────────────────────────────────
        # Build ordered list → interaction → map back to dict
        ordered_reprs = [task_reprs[t] for t in self._task_names]
        interacted = self.interaction(ordered_reprs)             # List[(B, D)]
        for task, H_i_prime in zip(self._task_names, interacted):
            task_reprs[task] = H_i_prime

        # ── 5. Task heads ────────────────────────────────────────────────
        outputs: Dict[str, Any] = {}
        for task in self._task_names:
            try:
                head_out = self.task_heads[task](task_reprs[task])
            except Exception as exc:
                raise RuntimeError(
                    f"Head '{task}' forward failed: {exc}"
                ) from exc

            if not isinstance(head_out, dict):
                raise TypeError(
                    f"Head '{task}' must return a dict "
                    f"(got {type(head_out).__name__})"
                )
            if "logits" not in head_out:
                raise RuntimeError(
                    f"Head '{task}' dict missing 'logits' "
                    f"(keys={list(head_out)})"
                )
            outputs[task] = head_out

        # ── 6. task_logits (LossEngine / training loop compat) ───────────
        outputs["task_logits"] = {
            t: outputs[t]["logits"] for t in self._task_names
        }

        # ── 7. Latent fusion + credibility score ─────────────────────────
        ordered_refined = [task_reprs[t] for t in self._task_names]
        Z, neural_score = self.fusion(ordered_refined)          # (B, D), (B,)

        outputs["latent_vector"] = Z
        outputs["credibility_score"] = neural_score

        return outputs

    # =====================================================================
    # HYBRID AGGREGATION
    # =====================================================================

    def blend_credibility(
        self,
        neural_score: torch.Tensor,
        rule_score: torch.Tensor,
        alpha: Optional[float] = None,
    ) -> torch.Tensor:
        """Blend neural credibility with a rule-based score.

        ``final = α * neural + (1 − α) * rule``

        Parameters
        ----------
        neural_score : (B,) or scalar — output of the fusion head.
        rule_score   : (B,) or scalar — rule-based aggregation score
                       (e.g. from ``AggregationPipeline``), already in [0, 1].
        alpha        : blend weight for neural score. Falls back to
                       ``self._hybrid_alpha`` if ``None``. If both are
                       ``None`` returns ``neural_score`` unchanged.

        Returns
        -------
        (B,) or scalar blended credibility score.
        """
        a = alpha if alpha is not None else self._hybrid_alpha
        if a is None:
            return neural_score

        a = float(a)
        if not (0.0 <= a <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {a}")

        return a * neural_score + (1.0 - a) * rule_score.to(neural_score.device)

    # =====================================================================
    # FACTORIES
    # =====================================================================

    @classmethod
    def from_interacting_config(
        cls,
        config: InteractingMultiTaskConfig,
    ) -> "InteractingMultiTaskModel":
        """Convenience constructor — builds the model from an
        ``InteractingMultiTaskConfig`` dataclass."""
        return cls(config=config)

    @classmethod
    def from_model_config(cls, model_config: Any) -> "InteractingMultiTaskModel":
        """Build from a YAML-backed ``MultiTaskModelConfig`` (registry path).

        Delegates the encoder + head construction to the parent class's
        ``from_model_config``, then wraps the result in the interacting
        architecture.
        """
        base: MultiTaskTruthLensModel = super().from_model_config(model_config)
        return cls(encoder=base.encoder, task_heads=dict(base.task_heads))

    # =====================================================================
    # UTILITIES (extend parent)
    # =====================================================================

    def get_task_names(self) -> List[str]:
        return list(self._task_names)

    def extra_repr(self) -> str:
        return (
            f"tasks={self._task_names}, "
            f"hybrid_alpha={self._hybrid_alpha}"
        )
