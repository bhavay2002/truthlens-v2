"""Compatibility wrapper around :class:`MultiTaskTruthLensModel`.

`TruthLensMultiTaskModel(model_name)` is the small string-based
constructor used by older callers (notably ``src.pipelines.baseline_training``).
It builds a :class:`MultiTaskTruthLensConfig` from the supplied
``model_name`` (and any extra keyword overrides) and delegates to the
canonical config-driven init path on the parent class.
"""

from __future__ import annotations

from typing import Any

from .multitask_truthlens_model import (
    MultiTaskTruthLensConfig,
    MultiTaskTruthLensModel,
)


class TruthLensMultiTaskModel(MultiTaskTruthLensModel):
    """Initialize the multitask model from a base model name string."""

    def __init__(self, model_name: str = "roberta-base", **kwargs: Any) -> None:

        valid_fields = {
            f.name for f in MultiTaskTruthLensConfig.__dataclass_fields__.values()
        }

        extra = {k: v for k, v in kwargs.items() if k not in valid_fields}
        config_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

        config = MultiTaskTruthLensConfig(model_name=model_name, **config_kwargs)

        if extra:
            # Stash any non-config kwargs on the config's metadata bag so
            # they survive (and remain inspectable) without breaking the
            # downstream constructor.
            config.extra_metadata.update(extra)

        # MUST be `config=...` — the parent's __init__ enforces the
        # mutual-exclusivity of (encoder, task_heads) vs `config=`.
        super().__init__(config=config)
