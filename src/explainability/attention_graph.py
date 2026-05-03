"""
AttentionGraphBuilder — Explainability v2 §3 (Attention graph).

Builds a directed weighted graph capturing the information flow:

    token nodes  →  task nodes  →  decision node

Edges
-----
token → task
    Weight = attention rollout score of that token for that task's head.
    Computed from the encoder's self-attention layers (via AttentionRollout)
    aggregated per-task by taking the mean of the first layer CLS weights.
    When per-task attention is unavailable the shared rollout is broadcast.

task → decision
    Weight = cross-task influence of that task on the final credibility
    decision, derived from the gate/attention weights of CrossTaskInteractionLayer
    averaged over all tasks (i.e., "how much does this task contribute to
    the latent fusion vector").

Public API
----------
    from src.explainability.attention_graph import AttentionGraphBuilder

    builder = AttentionGraphBuilder()
    graph = builder.build(
        tokens=["The", "president", "said", ...],
        token_scores={"bias": [0.1, 0.4, ...], "propaganda": [0.3, 0.2, ...]},
        task_to_decision={"bias": 0.6, "propaganda": 0.3, ...},
    )
    # graph.to_dict() → {"nodes": [...], "edges": [...]}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_EPS = 1e-12


# =========================================================
# GRAPH SCHEMA
# =========================================================

@dataclass
class GraphNode:
    """A node in the attention graph."""
    node_id: str
    node_type: str          # "token" | "task" | "decision"
    label: str
    index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "label": self.label,
        }
        if self.index is not None:
            d["index"] = self.index
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class GraphEdge:
    """A directed weighted edge in the attention graph."""
    source: str             # node_id
    target: str             # node_id
    weight: float           # normalised to [0, 1]
    edge_type: str          # "token_to_task" | "task_to_decision"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "edge_type": self.edge_type,
        }


@dataclass
class AttentionGraph:
    """Complete token→task→decision graph for one inference sample."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "metadata": self.metadata,
        }

    # Convenience selectors

    def token_nodes(self) -> List[GraphNode]:
        return [n for n in self.nodes if n.node_type == "token"]

    def task_nodes(self) -> List[GraphNode]:
        return [n for n in self.nodes if n.node_type == "task"]

    def decision_node(self) -> Optional[GraphNode]:
        nodes = [n for n in self.nodes if n.node_type == "decision"]
        return nodes[0] if nodes else None

    def top_token_edges(
        self,
        task: str,
        top_k: int = 10,
    ) -> List[GraphEdge]:
        """Return the top-k token→task edges for the given task, sorted by weight desc."""
        task_id = f"task:{task}"
        edges = [
            e for e in self.edges
            if e.target == task_id and e.edge_type == "token_to_task"
        ]
        return sorted(edges, key=lambda e: e.weight, reverse=True)[:top_k]


# =========================================================
# NORMALISATION
# =========================================================

def _l1_normalise(values: List[float]) -> List[float]:
    total = sum(abs(v) for v in values) + _EPS
    return [v / total for v in values]


def _minmax_normalise(values: List[float]) -> List[float]:
    if not values:
        return values
    lo, hi = min(values), max(values)
    span = hi - lo + _EPS
    return [(v - lo) / span for v in values]


# =========================================================
# BUILDER
# =========================================================

class AttentionGraphBuilder:
    """Build an AttentionGraph from per-task token scores and task→decision weights.

    Parameters
    ----------
    normalise_tokens : bool
        Whether to L1-normalise token scores within each task (default True).
    normalise_tasks : bool
        Whether to min-max normalise task→decision weights (default True).
    top_k_tokens : int | None
        Keep only the top-k tokens per task to avoid graph bloat. None = all.
    """

    DECISION_NODE_ID = "decision"
    DECISION_LABEL   = "credibility_decision"

    def __init__(
        self,
        normalise_tokens: bool = True,
        normalise_tasks: bool = True,
        top_k_tokens: Optional[int] = None,
    ) -> None:
        self.normalise_tokens = normalise_tokens
        self.normalise_tasks  = normalise_tasks
        self.top_k_tokens     = top_k_tokens

    # -------------------------------------------------------
    # MAIN BUILD
    # -------------------------------------------------------

    def build(
        self,
        tokens: List[str],
        token_scores: Dict[str, List[float]],
        task_to_decision: Dict[str, float],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AttentionGraph:
        """Build the attention graph.

        Parameters
        ----------
        tokens : list of token strings (length T)
        token_scores : {task_name: [score_0, ..., score_T-1]}
            Per-task attention/rollout scores for each token.
            Missing tasks get zero scores.
        task_to_decision : {task_name: float}
            How much each task contributes to the final decision.
        metadata : optional dict attached to the graph

        Returns
        -------
        AttentionGraph
        """
        task_names = list(task_to_decision.keys())
        num_tokens = len(tokens)

        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []

        # ── Token nodes ───────────────────────────────────────────────
        token_node_ids: List[str] = []
        for i, tok in enumerate(tokens):
            nid = f"token:{i}"
            nodes.append(GraphNode(
                node_id=nid,
                node_type="token",
                label=tok,
                index=i,
            ))
            token_node_ids.append(nid)

        # ── Task nodes ────────────────────────────────────────────────
        for task in task_names:
            nid = f"task:{task}"
            nodes.append(GraphNode(
                node_id=nid,
                node_type="task",
                label=task,
                metadata={"decision_weight": float(task_to_decision[task])},
            ))

        # ── Decision node ─────────────────────────────────────────────
        nodes.append(GraphNode(
            node_id=self.DECISION_NODE_ID,
            node_type="decision",
            label=self.DECISION_LABEL,
        ))

        # ── Token → Task edges ────────────────────────────────────────
        for task in task_names:
            raw_scores = token_scores.get(task, [0.0] * num_tokens)
            scores = list(raw_scores[:num_tokens])
            if len(scores) < num_tokens:
                scores += [0.0] * (num_tokens - len(scores))

            if self.normalise_tokens:
                scores = _l1_normalise(scores)

            # Optionally keep only top-k
            if self.top_k_tokens is not None and self.top_k_tokens < num_tokens:
                indexed = sorted(
                    enumerate(scores), key=lambda iv: abs(iv[1]), reverse=True
                )[:self.top_k_tokens]
                keep = {i for i, _ in indexed}
            else:
                keep = set(range(num_tokens))

            task_nid = f"task:{task}"
            for i, score in enumerate(scores):
                if i not in keep:
                    continue
                edges.append(GraphEdge(
                    source=token_node_ids[i],
                    target=task_nid,
                    weight=float(max(score, 0.0)),
                    edge_type="token_to_task",
                ))

        # ── Task → Decision edges ─────────────────────────────────────
        task_weights = list(task_to_decision.values())
        if self.normalise_tasks:
            task_weights = _minmax_normalise(task_weights)

        for task, weight in zip(task_names, task_weights):
            edges.append(GraphEdge(
                source=f"task:{task}",
                target=self.DECISION_NODE_ID,
                weight=float(max(weight, 0.0)),
                edge_type="task_to_decision",
            ))

        return AttentionGraph(
            nodes=nodes,
            edges=edges,
            metadata=metadata or {},
        )

    # -------------------------------------------------------
    # CONVENIENCE: build from InteractingMultiTaskModel outputs
    # -------------------------------------------------------

    @classmethod
    def from_model_outputs(
        cls,
        tokens: List[str],
        model_outputs: Dict[str, Any],
        cross_task_influence: Optional[Dict[str, Dict[str, float]]] = None,
        *,
        normalise_tokens: bool = True,
        normalise_tasks: bool = True,
        top_k_tokens: Optional[int] = None,
    ) -> AttentionGraph:
        """Build graph directly from InteractingMultiTaskModel.forward() output.

        Parameters
        ----------
        tokens : tokenizer token list (len = sequence length)
        model_outputs : output dict from InteractingMultiTaskModel.forward()
        cross_task_influence : optional pre-computed influence matrix from
            CrossTaskAttributor. Used to weight task→decision edges.
            Falls back to uniform weights if not provided.
        """
        builder = cls(
            normalise_tokens=normalise_tokens,
            normalise_tasks=normalise_tasks,
            top_k_tokens=top_k_tokens,
        )

        task_names: List[str] = list(
            model_outputs.get("task_outputs", {}).keys()
        )
        if not task_names:
            task_names = [
                k for k in model_outputs
                if k not in ("task_logits", "task_outputs", "task_representations",
                             "latent_vector", "credibility_score", "risk")
                and isinstance(model_outputs[k], dict)
            ]

        # task→decision weights: sum of influence each task receives from all others
        # (i.e., row sums of influence matrix) as a proxy for centrality
        if cross_task_influence is not None:
            task_to_decision: Dict[str, float] = {
                task: sum(cross_task_influence.get(task, {}).values())
                for task in task_names
            }
        else:
            task_to_decision = {task: 1.0 for task in task_names}

        # token scores: uniform for now (caller should provide actual rollout scores)
        token_scores: Dict[str, List[float]] = {
            task: [1.0 / max(len(tokens), 1)] * len(tokens)
            for task in task_names
        }

        return builder.build(
            tokens=tokens,
            token_scores=token_scores,
            task_to_decision=task_to_decision,
            metadata={
                "source": "model_outputs",
                "num_tasks": len(task_names),
                "num_tokens": len(tokens),
            },
        )

    @classmethod
    def from_rollout_and_attribution(
        cls,
        tokens: List[str],
        rollout_scores: Dict[str, List[float]],
        cross_task_influence: Dict[str, Dict[str, float]],
        *,
        normalise_tokens: bool = True,
        normalise_tasks: bool = True,
        top_k_tokens: Optional[int] = None,
    ) -> AttentionGraph:
        """Build graph from explicit rollout scores + cross-task influence matrix.

        Parameters
        ----------
        tokens : tokenizer token strings
        rollout_scores : {task: [score per token]} from AttentionRollout
        cross_task_influence : influence matrix from CrossTaskAttributor
        """
        builder = cls(
            normalise_tokens=normalise_tokens,
            normalise_tasks=normalise_tasks,
            top_k_tokens=top_k_tokens,
        )

        task_names = list(cross_task_influence.keys())

        task_to_decision: Dict[str, float] = {
            task: sum(cross_task_influence[task].values())
            for task in task_names
        }

        return builder.build(
            tokens=tokens,
            token_scores=rollout_scores,
            task_to_decision=task_to_decision,
            metadata={
                "source": "rollout_and_attribution",
                "num_tasks": len(task_names),
                "num_tokens": len(tokens),
            },
        )
