from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# ATTENTION EXTRACTOR
# =========================================================

class AttentionExtractor:
    """
    Extracts attention maps from transformer-based models.

    Supports:
        - HuggingFace models with output_attentions=True
        - Custom models exposing attention tensors in outputs
        - Hook-based extraction from attention modules

    Outputs:
        Dict with:
            - attentions: (L, B, H, T, T)
            - token_importance: (B, T)
            - head_importance: (L, H)
            - layer_importance: (L,)
    """

    def __init__(
        self,
        model: nn.Module,
        use_hooks: bool = False,
        target_modules: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.use_hooks = use_hooks
        self.target_modules = target_modules or ["attention", "attn"]

        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._hook_storage: List[torch.Tensor] = []

        if use_hooks:
            self._register_hooks()

    # =====================================================
    # HOOKS
    # =====================================================

    def _register_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if any(t in name.lower() for t in self.target_modules):
                handle = module.register_forward_hook(self._hook_fn)
                self._hook_handles.append(handle)
                logger.info(f"[ATTN] hook registered on {name}")

    def _hook_fn(self, module, inputs, output):
        # Expect output to contain attention weights or be a tuple
        if isinstance(output, tuple):
            for o in output:
                if isinstance(o, torch.Tensor) and o.dim() == 4:
                    self._hook_storage.append(o.detach())
        elif isinstance(output, torch.Tensor) and output.dim() == 4:
            self._hook_storage.append(output.detach())

    def clear_hooks(self) -> None:
        self._hook_storage.clear()

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    # =====================================================
    # FORWARD EXTRACTION
    # =====================================================

    def extract(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:

        self.model.eval()

        self.clear_hooks()

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                task=task,
                output_attentions=True,
            )

        attentions = None

        # HuggingFace-style
        if isinstance(outputs, dict) and "attentions" in outputs:
            attentions = outputs["attentions"]

        elif hasattr(outputs, "attentions"):
            attentions = outputs.attentions

        # Hook fallback
        elif self.use_hooks and self._hook_storage:
            attentions = tuple(self._hook_storage)

        if attentions is None:
            raise ValueError("No attention tensors found")

        # Convert to tensor (L, B, H, T, T)
        attn_tensor = self._stack_attentions(attentions)

        token_importance = self._compute_token_importance(attn_tensor)
        head_importance = self._compute_head_importance(attn_tensor)
        layer_importance = self._compute_layer_importance(attn_tensor)

        return {
            "attentions": attn_tensor,
            "token_importance": token_importance,
            "head_importance": head_importance,
            "layer_importance": layer_importance,
        }

    # =====================================================
    # PROCESSING
    # =====================================================

    def _stack_attentions(
        self,
        attentions: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """
        Convert list of attention tensors to unified format:
        (L, B, H, T, T)
        """
        stacked = []

        for attn in attentions:
            if attn.dim() == 4:
                # (B, H, T, T)
                stacked.append(attn)
            else:
                raise ValueError("Unexpected attention shape")

        return torch.stack(stacked, dim=0)

    def _compute_token_importance(
        self,
        attn: torch.Tensor,
    ) -> torch.Tensor:
        """
        Token importance via attention rollout.
        Returns: (B, T)
        """
        # mean over heads
        attn_mean = attn.mean(dim=2)  # (L, B, T, T)

        # rollout (simple multiplication)
        rollout = attn_mean[0]

        for i in range(1, attn_mean.shape[0]):
            rollout = torch.bmm(attn_mean[i], rollout)

        # importance = sum over source tokens
        importance = rollout.sum(dim=1)

        return self._normalize(importance)

    def _compute_head_importance(
        self,
        attn: torch.Tensor,
    ) -> torch.Tensor:
        """
        Head importance based on average attention strength.
        Returns: (L, H)
        """
        return attn.mean(dim=(1, 3, 4))

    def _compute_layer_importance(
        self,
        attn: torch.Tensor,
    ) -> torch.Tensor:
        """
        Layer importance based on total attention magnitude.
        Returns: (L,)
        """
        return attn.mean(dim=(1, 2, 3, 4))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.max(dim=1, keepdim=True).values + EPS)