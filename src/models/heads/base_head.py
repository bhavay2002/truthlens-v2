"""Base contract for all task heads (audit 3.4).

Every head MUST return a ``dict`` containing at minimum a ``logits`` tensor.
The multi-task wrappers, the loss engine and ``MultiTaskOutput`` all
assume this shape; tensor-only return values previously trained fine but
crashed at calibration time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn


class BaseHead(nn.Module, ABC):
    """Abstract base class for classification / regression / multilabel heads.

    Subclasses must implement :meth:`forward` to return a dictionary
    containing at least a ``logits`` tensor. Additional fields
    (``probabilities``, ``predictions``, ``confidence``, ``entropy``,
    ``features`` …) are head-specific and may be omitted at training
    time for performance.
    """

    @abstractmethod
    def forward(self, features: torch.Tensor) -> Dict[str, Any]:
        """Compute task outputs for ``features``.

        Returns
        -------
        Dict[str, Any]
            Must contain key ``"logits"`` mapped to a ``Tensor``.
        """


__all__ = ["BaseHead"]
