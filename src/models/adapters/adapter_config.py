from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal


# =========================================================
# ADAPTER CONFIGURATION
# =========================================================

@dataclass
class AdapterConfig:
    """
    Configuration for generic adapter layers.

    Controls architecture, initialization, and placement of adapters
    inside transformer models.
    """

    enabled: bool = False

    # -------- Architecture --------
    adapter_type: Literal["standard", "lora"] = "standard"
    adapter_dim: int = 64
    activation: Literal["gelu", "relu", "tanh"] = "gelu"
    dropout: float = 0.1

    # -------- Normalization --------
    use_layernorm: bool = True
    pre_layernorm: bool = True

    # -------- Residual --------
    residual: bool = True

    # -------- Placement --------
    target_modules: Optional[list[str]] = field(
        default_factory=lambda: ["attention", "ffn"]
    )

    # -------- Initialization --------
    init_scale: float = 1e-3

    # -------- Training --------
    freeze_base_model: bool = True
    train_bias: bool = False

    # -------- Performance --------
    use_compiled_adapter: bool = False


# =========================================================
# LORA CONFIGURATION
# =========================================================

@dataclass
class LoRAConfig:
    """
    Configuration for Low-Rank Adaptation (LoRA).

    Used for parameter-efficient fine-tuning.
    """

    enabled: bool = False

    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.1

    target_modules: Optional[list[str]] = field(
        default_factory=lambda: ["query", "key", "value"]
    )

    merge_weights: bool = False


# =========================================================
# COMBINED CONFIG
# =========================================================

@dataclass
class AdapterSystemConfig:
    """
    Unified adapter configuration for model-level usage.
    """

    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    enabled: bool = False

    def is_active(self) -> bool:
        return self.enabled and (self.adapter.enabled or self.lora.enabled)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "adapter": vars(self.adapter),
            "lora": vars(self.lora),
        }