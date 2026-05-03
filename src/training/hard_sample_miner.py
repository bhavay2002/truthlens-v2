"""
HardSampleMiner — online hard example mining for multi-task training.

Spec §3 (Training Pipeline Upgrade).

Hard-sample definition
----------------------
A sample is "hard" if any of:
  A. High prediction entropy (uncertain prediction):
        H(p) = -Σ p log p  > entropy_threshold
  B. Cross-task disagreement (bias ≠ ideology, emotion ↔ narrative):
        captured as disagreement_weight * cross_task_disagreement_score
  C. High per-sample loss:
        loss_i > loss_threshold

Composite hardness score (per sample, spec §3.3)
-------------------------------------------------
    hardness_i = entropy_weight * H̄(p_i)
               + disagreement_weight * D(p_i)
               + loss_weight * clamp(L_i / loss_norm, 0, 1)

Pool management (spec §3.3)
----------------------------
    * Fixed-size max-heap ``hard_pool`` of (hardness, dataset_idx) pairs.
    * After each update, top-K hardest samples are kept in the pool.
    * Easiest samples are evicted when the pool is full.

Sampling (spec §3.4)
--------------------
    * Call ``mix_indices(normal_indices)`` to get a composite batch with
      ``hard_ratio`` fraction drawn from the hard pool and
      ``(1 - hard_ratio)`` fraction from the provided normal indices.

Integration
-----------
    miner = HardSampleMiner()
    for batch in loader:
        outputs = model(batch)
        miner.update(batch["sample_idx"], outputs, per_sample_losses)
    # Next epoch: wrap normal indices
    hard_indices = miner.get_hard_indices(n=len(batch))
"""

from __future__ import annotations

import heapq
import logging
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

class HardSampleMinerConfig:
    """Configuration for ``HardSampleMiner``.

    Parameters
    ----------
    max_pool_size:
        Maximum number of (hardness, sample_idx) entries to keep in the
        hard pool. Older / easier samples are evicted when the pool fills.
    hard_ratio:
        Fraction of each batch drawn from the hard pool during Phase 2+.
        Spec §3.4 recommends 0.30.
    entropy_threshold:
        Minimum per-sample entropy (averaged across all tasks) for a sample
        to be considered "hard" by criterion A. Range [0, log(C)] where C
        is the number of classes. Typical: 0.5.
    loss_weight:
        Contribution of per-sample loss to the composite hardness score.
    entropy_weight:
        Contribution of per-sample average entropy.
    disagreement_weight:
        Contribution of cross-task disagreement.
    loss_norm:
        Scale factor for normalising losses before clamping to [0, 1].
        Losses above this threshold are treated as maximally hard.
    update_top_k:
        After each forward pass, add only the top-K hardest samples
        from the batch (avoids polluting the pool with trivially-hard
        boundary samples when the batch itself is easy on average).
    """

    def __init__(
        self,
        max_pool_size: int = 1000,
        hard_ratio: float = 0.30,
        entropy_threshold: float = 0.5,
        loss_weight: float = 0.5,
        entropy_weight: float = 0.3,
        disagreement_weight: float = 0.2,
        loss_norm: float = 2.0,
        update_top_k: int = 8,
    ) -> None:
        if not (0.0 <= hard_ratio <= 1.0):
            raise ValueError(f"hard_ratio must be in [0, 1] (got {hard_ratio})")
        if max_pool_size < 1:
            raise ValueError("max_pool_size must be >= 1")

        self.max_pool_size = max_pool_size
        self.hard_ratio = hard_ratio
        self.entropy_threshold = entropy_threshold
        self.loss_weight = loss_weight
        self.entropy_weight = entropy_weight
        self.disagreement_weight = disagreement_weight
        self.loss_norm = max(loss_norm, 1e-6)
        self.update_top_k = max(1, update_top_k)


# =========================================================
# MINER
# =========================================================

class HardSampleMiner:
    """Online hard example mining with a fixed-size max-heap pool.

    Thread-safety: NOT thread-safe. Use one instance per worker process.

    Parameters
    ----------
    config:
        Tuning parameters. Defaults applied if ``None``.
    """

    def __init__(self, config: Optional[HardSampleMinerConfig] = None) -> None:
        self.cfg = config or HardSampleMinerConfig()

        # Max-heap implemented as min-heap over (-hardness, idx).
        # Entry: (-hardness, sample_idx)  so heapq.heappop() evicts EASIEST.
        self._heap: List[tuple] = []

        # Fast membership check (idx → hardness)
        self._pool: Dict[int, float] = {}

        # Rolling stats for logging
        self._total_updates: int = 0
        self._total_evictions: int = 0

        logger.info(
            "HardSampleMiner | pool_size=%d | hard_ratio=%.2f | "
            "top_k=%d",
            self.cfg.max_pool_size,
            self.cfg.hard_ratio,
            self.cfg.update_top_k,
        )

    # -----------------------------------------------------------------------
    # HARDNESS COMPUTATION
    # -----------------------------------------------------------------------

    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Per-sample entropy from classification logits.

        Parameters
        ----------
        logits : (B, C)

        Returns
        -------
        (B,) entropy in nats, clamped to [0, log(C)].
        """
        if logits.dim() != 2:
            raise ValueError(f"Expected 2-D logits (B, C), got {logits.shape}")
        log_probs = F.log_softmax(logits.float(), dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)  # (B,)
        max_entropy = math.log(max(logits.size(1), 2))
        return entropy.clamp(0.0, max_entropy)

    def compute_multilabel_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Per-sample average binary entropy for multilabel logits.

        Parameters
        ----------
        logits : (B, L)

        Returns
        -------
        (B,) average binary entropy per sample.
        """
        p = torch.sigmoid(logits.float())
        p = p.clamp(1e-7, 1 - 1e-7)
        h = -(p * p.log() + (1 - p) * (1 - p).log())  # (B, L)
        return h.mean(dim=-1)                           # (B,)

    def compute_cross_task_disagreement(
        self,
        task_logits: Dict[str, torch.Tensor],
        task_types: Optional[Dict[str, str]] = None,
    ) -> torch.Tensor:
        """Cross-task disagreement score per sample.

        Computes the mean pairwise cosine DISTANCE between all task
        probability distributions. High distance → high disagreement.

        For multi-class tasks: softmax probabilities.
        For multilabel tasks: sigmoid probabilities.

        Returns (B,) disagreement score in [0, 1].
        """
        if not task_logits:
            B = next(iter(task_logits.values())).size(0) if task_logits else 1
            return torch.zeros(B)

        task_types = task_types or {}
        task_probs: List[torch.Tensor] = []

        for task, logits in task_logits.items():
            ttype = task_types.get(task, "multi_class")
            if ttype == "multilabel":
                probs = torch.sigmoid(logits.float())
            else:
                probs = torch.softmax(logits.float(), dim=-1)
            # L2-normalise for cosine distance
            normed = F.normalize(probs, p=2, dim=-1)        # (B, C_i)
            task_probs.append(normed)

        if len(task_probs) < 2:
            return torch.zeros(task_probs[0].size(0))

        # Pairwise cosine similarity, averaged over all pairs
        T = len(task_probs)
        pair_sims: List[torch.Tensor] = []
        for i in range(T):
            for j in range(i + 1, T):
                a = task_probs[i]
                b = task_probs[j]
                # Handle dimension mismatch by truncating to shorter
                min_d = min(a.size(1), b.size(1))
                sim = (a[:, :min_d] * b[:, :min_d]).sum(dim=-1)  # (B,)
                pair_sims.append(sim)

        mean_sim = torch.stack(pair_sims, dim=-1).mean(dim=-1)  # (B,)
        # Convert cosine similarity → distance in [0, 1]
        return ((1.0 - mean_sim) / 2.0).clamp(0.0, 1.0)

    def compute_hardness(
        self,
        model_outputs: Dict[str, Any],
        per_sample_losses: Optional[torch.Tensor] = None,
        task_types: Optional[Dict[str, str]] = None,
    ) -> torch.Tensor:
        """Composite per-sample hardness score.

        Parameters
        ----------
        model_outputs:
            Forward output dict from the model. Reads ``"task_logits"``.
        per_sample_losses:
            (B,) per-sample total loss. If ``None``, the loss term is
            skipped (entropy + disagreement only).
        task_types:
            Map task_name → "multi_class" | "multilabel".

        Returns
        -------
        (B,) hardness score (higher = harder).
        """
        task_logits: Dict[str, torch.Tensor] = model_outputs.get(
            "task_logits", {}
        )

        if not task_logits:
            logger.warning("HardSampleMiner: no task_logits in model_outputs")
            B = (
                per_sample_losses.size(0)
                if per_sample_losses is not None
                else 1
            )
            return torch.zeros(B)

        # ── Criterion A: entropy ─────────────────────────────────────────
        task_types = task_types or {}
        entropy_terms: List[torch.Tensor] = []
        for task, logits in task_logits.items():
            ttype = task_types.get(task, "multi_class")
            with torch.no_grad():
                if ttype == "multilabel":
                    h = self.compute_multilabel_entropy(logits)
                else:
                    h = self.compute_entropy(logits)
            entropy_terms.append(h)

        mean_entropy = torch.stack(entropy_terms, dim=-1).mean(dim=-1)  # (B,)

        # ── Criterion B: cross-task disagreement ─────────────────────────
        with torch.no_grad():
            disagreement = self.compute_cross_task_disagreement(
                task_logits, task_types
            )

        # Ensure they're on the same device
        device = mean_entropy.device
        disagreement = disagreement.to(device)

        # ── Criterion C: per-sample loss ──────────────────────────────────
        if per_sample_losses is not None:
            loss_term = (
                per_sample_losses.detach().float().to(device) / self.cfg.loss_norm
            ).clamp(0.0, 1.0)
        else:
            loss_term = torch.zeros_like(mean_entropy)

        hardness = (
            self.cfg.entropy_weight * mean_entropy
            + self.cfg.disagreement_weight * disagreement
            + self.cfg.loss_weight * loss_term
        )

        return hardness.cpu()   # pool lives on CPU

    # -----------------------------------------------------------------------
    # POOL UPDATE
    # -----------------------------------------------------------------------

    def update(
        self,
        sample_indices: List[int],
        model_outputs: Dict[str, Any],
        per_sample_losses: Optional[torch.Tensor] = None,
        task_types: Optional[Dict[str, str]] = None,
    ) -> None:
        """Update the hard-sample pool from a completed forward pass.

        Parameters
        ----------
        sample_indices:
            Dataset indices for each sample in the batch. Length B.
        model_outputs:
            Model forward output (reads ``"task_logits"``).
        per_sample_losses:
            (B,) per-sample losses (optional).
        task_types:
            Task type dict forwarded to hardness computation.
        """
        if not sample_indices:
            return

        B = len(sample_indices)
        hardness = self.compute_hardness(model_outputs, per_sample_losses, task_types)

        if hardness.size(0) != B:
            logger.warning(
                "HardSampleMiner.update: hardness size %d != "
                "sample_indices size %d — skipping update.",
                hardness.size(0), B,
            )
            return

        # Select top-K hardest from this batch
        k = min(self.cfg.update_top_k, B)
        top_k_vals, top_k_pos = torch.topk(hardness, k)

        for pos, score in zip(top_k_pos.tolist(), top_k_vals.tolist()):
            idx = sample_indices[pos]
            # Skip samples already in pool with higher hardness
            if idx in self._pool and self._pool[idx] >= score:
                continue
            self._add_to_pool(idx, score)

        self._total_updates += 1

        if self._total_updates % 100 == 0:
            logger.debug(
                "HardSampleMiner | pool=%d/%d | updates=%d | evictions=%d",
                len(self._pool),
                self.cfg.max_pool_size,
                self._total_updates,
                self._total_evictions,
            )

    def _add_to_pool(self, idx: int, score: float) -> None:
        """Insert (idx, score) into the max-heap pool, evicting easiest if full."""
        # Remove stale entry if present
        if idx in self._pool:
            self._pool[idx] = score
            # Heap will be fixed lazily (lazy deletion is fine for our use)
        else:
            if len(self._pool) >= self.cfg.max_pool_size:
                # Evict easiest (smallest score = smallest -score in min-heap = heappop)
                while self._heap:
                    neg_s, evict_idx = heapq.heappop(self._heap)
                    # Lazy deletion: skip if evict_idx no longer in pool or
                    # has a different score (was updated since insertion).
                    if evict_idx in self._pool and self._pool[evict_idx] == -neg_s:
                        del self._pool[evict_idx]
                        self._total_evictions += 1
                        break
            self._pool[idx] = score

        heapq.heappush(self._heap, (-score, idx))

    # -----------------------------------------------------------------------
    # SAMPLING
    # -----------------------------------------------------------------------

    def get_hard_indices(self, n: int) -> List[int]:
        """Return up to ``n`` indices from the hard-sample pool.

        Returns the hardest samples first. If the pool has fewer than
        ``n`` entries, all pool entries are returned.
        """
        if not self._pool:
            return []
        # Sort pool by hardness descending
        sorted_pool = sorted(self._pool.items(), key=lambda kv: kv[1], reverse=True)
        return [idx for idx, _ in sorted_pool[:n]]

    def mix_indices(
        self,
        normal_indices: List[int],
        rng=None,
    ) -> List[int]:
        """Compose a batch of indices with ``hard_ratio`` hard samples.

        Parameters
        ----------
        normal_indices:
            Indices drawn from the standard sampler for this batch.
        rng:
            Optional ``random.Random`` instance for reproducibility.

        Returns
        -------
        Mixed list of indices (hard + normal, shuffled).
        """
        import random as _random

        rng = rng or _random.Random()
        n = len(normal_indices)
        n_hard = min(int(n * self.cfg.hard_ratio), len(self._pool))
        n_normal = n - n_hard

        hard = self.get_hard_indices(n_hard)
        normal = normal_indices[:n_normal]

        mixed = hard + normal
        rng.shuffle(mixed)
        return mixed

    # -----------------------------------------------------------------------
    # STATS
    # -----------------------------------------------------------------------

    def pool_size(self) -> int:
        """Current number of entries in the hard pool."""
        return len(self._pool)

    def pool_stats(self) -> Dict[str, float]:
        """Summary statistics of the current hardness distribution."""
        if not self._pool:
            return {"pool_size": 0, "mean": 0.0, "max": 0.0, "min": 0.0}
        scores = list(self._pool.values())
        return {
            "pool_size": float(len(scores)),
            "mean": float(sum(scores) / len(scores)),
            "max": float(max(scores)),
            "min": float(min(scores)),
        }

    def reset(self) -> None:
        """Clear the pool (call between runs / curriculum phases)."""
        self._heap.clear()
        self._pool.clear()
        self._total_updates = 0
        self._total_evictions = 0
        logger.info("HardSampleMiner pool reset")

    def __repr__(self) -> str:
        return (
            f"HardSampleMiner(pool={len(self._pool)}/"
            f"{self.cfg.max_pool_size}, "
            f"hard_ratio={self.cfg.hard_ratio:.2f})"
        )
