from __future__ import annotations

import torch
import torch.nn as nn


class AdapterLayer(nn.Module):
    """
    Generic Adapter Layer for parameter-efficient fine-tuning.

    Architecture:
        x -> LayerNorm -> Down Projection -> Activation -> Dropout
          -> Up Projection -> Residual Add

    Inputs:
        hidden_states: (B, T, H)

    Outputs:
        adapted_states: (B, T, H)
    """

    def __init__(
        self,
        hidden_dim: int,
        adapter_dim: int = 64,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layernorm: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.adapter_dim = adapter_dim
        self.residual = residual

        # Optional pre-normalization
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.norm = nn.LayerNorm(hidden_dim)

        # Down projection
        self.down_proj = nn.Linear(hidden_dim, adapter_dim)

        # Activation
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.dropout = nn.Dropout(dropout)

        # Up projection
        self.up_proj = nn.Linear(adapter_dim, hidden_dim)

        # Initialize near identity
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)

        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: (B, T, H)

        Returns:
            adapted_states: (B, T, H)
        """

        residual = hidden_states

        x = hidden_states

        if self.use_layernorm:
            x = self.norm(x)

        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)

        if self.residual:
            x = x + residual

        return x