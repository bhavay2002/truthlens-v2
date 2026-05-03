# src/features/semantic_features.py

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature
from src.features.base.numerics import EPS, MAX_CLIP, normalized_entropy

logger = logging.getLogger(__name__)


@dataclass
@register_feature
class SemanticFeatures(BaseFeature):

    name: str = "semantic_features"
    group: str = "semantic"
    description: str = "Advanced semantic embedding features"

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> dict:

        # Audit fix §3.5 — when no embedding is available we now return
        # the explicit empty dict instead of synthesising a 128-d zero
        # vector. The previous behaviour produced a degenerate "all
        # features look identical" row that polluted downstream variance
        # / correlation pruning and silently masked configuration
        # mistakes (no encoder configured).
        emb = self._get_embedding(context)
        if emb is None or emb.size == 0:
            return self._empty()

        emb = np.nan_to_num(emb)

        # -------------------------
        # NORMALIZATION
        # -------------------------

        norm = np.linalg.norm(emb) + EPS
        emb_norm = emb / norm

        # -------------------------
        # BASIC STATS
        # -------------------------

        mean = float(np.mean(emb_norm))
        std = float(np.std(emb_norm))

        # -------------------------
        # ENTROPY (CRITICAL)
        # -------------------------

        probs = np.abs(emb_norm)
        probs /= (np.sum(probs) + EPS)

        entropy = normalized_entropy(probs)

        # -------------------------
        # ANISOTROPY (IMPORTANT)
        # -------------------------

        anisotropy = float(np.var(emb_norm))

        # -------------------------
        # SPARSITY
        # -------------------------

        sparsity = float(np.count_nonzero(np.abs(emb_norm) < 1e-3) / len(emb_norm))

        # -------------------------
        # PEAKINESS
        # -------------------------

        peak = float(np.max(np.abs(emb_norm)))

        # -------------------------
        # OUTPUT
        # -------------------------

        return {
            "sem_norm": self._safe(norm),
            "sem_mean": self._safe(mean),
            "sem_std": self._safe(std),

            "sem_entropy": self._safe(entropy),
            "sem_anisotropy": self._safe(anisotropy),
            "sem_sparsity": self._safe(sparsity),
            "sem_peakiness": self._safe(peak),

            # Audit fix §3.5 — explicit availability indicator. The
            # 7-dim ``sem_*`` block is identical (all zeros) when no
            # encoder is wired up, which the model would otherwise
            # learn as a spurious "encoder-was-up" pattern. Emitting a
            # binary indicator lets the downstream head attenuate the
            # semantic block on encoder-failure rows.
            "sem_available": 1.0,
        }

    # -----------------------------------------------------

    def _get_embedding(self, context: FeatureContext) -> np.ndarray:
        # Audit fix §3.5 — explicit "no embedding" signal. The previous
        # ``np.zeros(128)`` fallback silently produced a fully-zero
        # feature row whenever no encoder was wired up, which then
        # passed all downstream finite/clip checks and looked like
        # legitimate output. Returning an empty array forces extract()
        # to fall back to ``_empty()`` and surface the misconfiguration
        # via diff logs upstream.
        if context.embeddings is not None:
            return np.asarray(context.embeddings)
        return np.empty(0, dtype=np.float32)

    # -----------------------------------------------------

    def _empty(self) -> dict:
        # Audit fix §3.5 — keep the schema stable but flip the
        # availability indicator off so the downstream model can tell
        # encoder-failure rows apart from "the encoder ran and emitted
        # an all-zero embedding" rows.
        return {
            "sem_norm": 0.0,
            "sem_mean": 0.0,
            "sem_std": 0.0,
            "sem_entropy": 0.0,
            "sem_anisotropy": 0.0,
            "sem_sparsity": 0.0,
            "sem_peakiness": 0.0,
            "sem_available": 0.0,
        }

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))