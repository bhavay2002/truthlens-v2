
#File Name: attention_visualizer.py

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

from src.explainability.attention_rollout import AttentionRollout

logger = logging.getLogger(__name__)
EPS = 1e-12


class AttentionVisualizer:

    def __init__(self, model: torch.nn.Module) -> None:
        if model is None:
            raise ValueError("model cannot be None")

        self.model = model
        self.rollout = AttentionRollout()

        logger.info("AttentionVisualizer initialized")

    # =====================================================
    # DEVICE
    # =====================================================

    def _resolve_model_device(self):
        try:
            return next(self.model.parameters()).device
        except Exception:
            return None

    # =====================================================
    # EXTRACTION
    # =====================================================

    def extract_attention(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, List[torch.Tensor]]:

        if input_ids.ndim != 2:
            raise ValueError("input_ids must be 2D")

        device = self._resolve_model_device()
        if device:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    return_dict=True,
                )
        except TypeError:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

        attentions = getattr(outputs, "attentions", None)

        if attentions is None:
            raise RuntimeError("Model does not return attentions")

        return {"attentions": list(attentions)}

    # =====================================================
    # AGGREGATION
    # =====================================================

    def aggregate_attention(
        self,
        attentions: List[torch.Tensor],
        sample_index: int = 0,
    ) -> np.ndarray:

        stacked = torch.stack(attentions)
        avg = stacked[:, sample_index].mean(dim=0).mean(dim=0)

        return avg.cpu().numpy()

    # =====================================================
    #  ROLLOUT (NEW)
    # =====================================================

    def compute_rollout(
        self,
        attentions: List[torch.Tensor],
        tokens: List[str],
        *,
        source_token_index: int = 0,
    ) -> Dict:

        return self.rollout.compute_rollout(
            attentions=attentions,
            tokens=tokens,
            source_token_index=source_token_index,
        )

    # =====================================================
    # NORMALIZATION
    # =====================================================

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        matrix = np.maximum(matrix, 0.0)
        total = np.sum(matrix) + EPS
        return matrix / total

    # =====================================================
    # HEATMAP
    # =====================================================

    def plot_attention(
        self,
        attention_matrix: np.ndarray,
        tokens: List[str],
        *,
        title: str = "Attention Map",
        save_path: Optional[str] = None,
        normalize: bool = True,
    ):

        if normalize:
            attention_matrix = self._normalize(attention_matrix)

        size = min(len(tokens), attention_matrix.shape[0])
        matrix = attention_matrix[:size, :size]

        import matplotlib.pyplot as plt  # GPU-5: lazy import
        fig, ax = plt.subplots(figsize=(10, 8))

        img = ax.imshow(matrix)
        fig.colorbar(img, ax=ax)

        ax.set_xticks(range(size))
        ax.set_xticklabels(tokens[:size], rotation=90)

        ax.set_yticks(range(size))
        ax.set_yticklabels(tokens[:size])

        ax.set_title(title)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path)

        plt.close(fig)

    # =====================================================
    #  TOKEN IMPORTANCE BAR (NEW)
    # =====================================================

    def plot_token_importance(
        self,
        tokens: List[str],
        scores: List[float],
        *,
        title: str = "Token Importance",
        save_path: Optional[str] = None,
    ):

        scores = np.asarray(scores)
        scores = scores / (np.sum(scores) + EPS)

        import matplotlib.pyplot as plt  # GPU-5: lazy import
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.bar(range(len(tokens)), scores)

        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)

        ax.set_title(title)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path)

        plt.close(fig)

    # =====================================================
    #  FULL PIPELINE (NEW)
    # =====================================================

    def analyze(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tokens: List[str],
    ) -> Dict:

        extracted = self.extract_attention(input_ids, attention_mask)
        attentions = extracted["attentions"]

        heatmap = self.aggregate_attention(attentions)

        rollout = self.compute_rollout(
            attentions=attentions,
            tokens=tokens,
        )

        return {
            "attention_matrix": heatmap,
            "rollout": rollout,
        }