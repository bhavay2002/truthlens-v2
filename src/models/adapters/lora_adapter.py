from __future__ import annotations

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) applied to a linear layer.

    Implements:
        W' = W + (alpha / r) * (B @ A)

    where:
        A: (r, in_features)
        B: (out_features, r)

    Inputs:
        x: (B, *, in_features)

    Outputs:
        y: (B, *, out_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
        merge_weights: bool = False,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(r, 1)
        self.merge_weights = merge_weights

        # Base linear layer (frozen by default externally)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        # LoRA parameters
        if r > 0:
            self.A = nn.Parameter(torch.zeros(r, in_features))
            self.B = nn.Parameter(torch.zeros(out_features, r))
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._reset_parameters()

        self.merged = False

    def _reset_parameters(self) -> None:
        # Initialize base weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # LoRA init (important)
        if self.r > 0:
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """

        # Base projection
        result = torch.nn.functional.linear(x, self.weight, self.bias)

        if self.r > 0 and not self.merged:
            lora_out = self.dropout(x) @ self.A.T  # (B, *, r)
            lora_out = lora_out @ self.B.T         # (B, *, out_features)
            result = result + lora_out * self.scaling

        return result

    # =====================================================
    # MERGE / UNMERGE
    # =====================================================

    def merge(self) -> None:
        """
        Merge LoRA weights into base weights for inference.
        """
        if self.r > 0 and not self.merged:
            delta = (self.B @ self.A) * self.scaling
            self.weight.data += delta
            self.merged = True

    def unmerge(self) -> None:
        """
        Restore original weights (for training).
        """
        if self.r > 0 and self.merged:
            delta = (self.B @ self.A) * self.scaling
            self.weight.data -= delta
            self.merged = False


# =========================================================
# UTILITY: APPLY LORA TO MODEL
# =========================================================

def apply_lora_to_linear(
    module: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_keywords: tuple[str, ...] = ("query", "key", "value", "dense"),
) -> nn.Module:
    """
    Recursively replace Linear layers with LoRALinear.

    Args:
        module: model or submodule
        target_keywords: names to match for replacement

    Returns:
        modified module
    """

    for name, child in module.named_children():

        if isinstance(child, nn.Linear) and any(k in name.lower() for k in target_keywords):

            lora_layer = LoRALinear(
                in_features=child.in_features,
                out_features=child.out_features,
                r=r,
                alpha=alpha,
                dropout=dropout,
                bias=child.bias is not None,
            )

            # copy original weights
            lora_layer.weight.data = child.weight.data.clone()

            if child.bias is not None:
                lora_layer.bias.data = child.bias.data.clone()

            setattr(module, name, lora_layer)

        else:
            apply_lora_to_linear(child, r, alpha, dropout, target_keywords)

    return module