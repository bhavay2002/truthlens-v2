from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


class BaseModel(nn.Module, ABC):

    def __init__(self) -> None:
        super().__init__()
        self._device: torch.device = torch.device("cpu")

    # =====================================================
    # FORWARD
    # =====================================================

    @abstractmethod
    def forward(self, *inputs: torch.Tensor, **kwargs: Any) -> Any:
        raise NotImplementedError

    # =====================================================
    # DEVICE
    # =====================================================

    def set_device(self, device: torch.device | str) -> None:

        if isinstance(device, str):
            device = torch.device(device)

        if not isinstance(device, torch.device):
            raise TypeError("device must be torch.device or str")

        logger.info("Moving model to device: %s", device)

        self._device = device
        self.to(device)

    @property
    def device(self) -> torch.device:
        # A3.3: cached fast path. ``set_device`` keeps ``self._device``
        # authoritative; we used to walk ``next(self.parameters())`` on
        # every access, which constructs a fresh generator and shows
        # up in profiles for short-batch encoder forwards. Only fall
        # back to a parameter walk when ``_device`` was never set
        # (defensive — the constructor seeds it to ``cpu``).
        if self._device is not None:
            return self._device

        for p in self.parameters():
            return p.device

        for b in self.buffers():
            return b.device

        return torch.device("cpu")

    # -----------------------------------------------------
    # LAZY-MODULE DEVICE SYNC  (A5.3)
    # -----------------------------------------------------

    def attach_module(self, name: str, module: nn.Module) -> None:
        """Register ``module`` and migrate it to the model's device.

        ``set_device`` only moves modules that exist at call time. Any
        sub-module created later (a temperature scalar added by
        :class:`TemperatureScaler`, a freshly attached LoRA adapter,
        any post-hoc auxiliary head) silently sits on CPU, producing
        a ``RuntimeError`` deep in the next forward pass. Use this
        helper instead of ``self.add_module`` / direct attribute
        assignment to keep placement consistent.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        if not isinstance(module, nn.Module):
            raise TypeError("module must be nn.Module")

        if self._device is not None:
            module.to(self._device)
        self.add_module(name, module)

    # =====================================================
    # PARAMS
    # =====================================================

    def num_parameters(self, trainable_only: bool = True) -> int:

        if trainable_only:
            params = (p for p in self.parameters() if p.requires_grad)
        else:
            params = self.parameters()

        return sum(p.numel() for p in params)

    def parameter_breakdown(self) -> Dict[str, int]:

        result: Dict[str, int] = {}

        if hasattr(self, "encoder"):
            result["encoder"] = sum(p.numel() for p in self.encoder.parameters())

        if hasattr(self, "task_heads"):
            for name, head in self.task_heads.items():
                result[f"head_{name}"] = sum(p.numel() for p in head.parameters())

        elif hasattr(self, "heads"):
            for name, head in self.heads.items():
                result[f"head_{name}"] = sum(p.numel() for p in head.parameters())

        return result

    # =====================================================
    # OPTIMIZER PARAMETER GROUPS  (G4)
    # =====================================================
    #
    # G4: post-hoc calibration parameters (notably ``self.temperature``
    # on the per-task classifiers) MUST NOT receive the same gradient
    # signal as the rest of the model. They are fitted on held-out
    # logits AFTER training; if the main optimizer touches them the
    # network learns to drive ``T`` to a degenerate value to lower the
    # loss, which destroys both the calibration AND the classification
    # head it scales.
    #
    # Trainers should build their optimizer with two parameter groups:
    # one for ``model.get_optimization_parameters()`` (the parameters
    # that should be updated by the main loss) and one for
    # ``model.get_calibration_parameters()`` (excluded from the main
    # optimizer; fitted separately by ``TemperatureScaler`` /
    # ``fit_temperature``). The names below are the canonical
    # markers — anything matching is treated as a calibration param.

    #: Names that identify a parameter as belonging to the calibration
    #: subsystem (G4). Subclasses may override or extend this set.
    CALIBRATION_PARAMETER_NAMES: tuple[str, ...] = ("temperature",)

    def _is_calibration_parameter_name(self, name: str) -> bool:
        """Return True iff ``name`` is a calibration parameter (G4).

        We match on the *trailing* component of the dotted parameter
        name so that nested modules with their own temperature scalar
        are picked up too (e.g. ``"task_heads.bias.temperature"``).
        """
        leaf = name.rsplit(".", 1)[-1]
        return leaf in self.CALIBRATION_PARAMETER_NAMES

    def get_calibration_parameters(self) -> list[nn.Parameter]:
        """Parameters fitted post-hoc on held-out logits (G4).

        These MUST be excluded from the main training optimizer.
        """
        return [
            p
            for n, p in self.named_parameters()
            if self._is_calibration_parameter_name(n)
        ]

    def get_optimization_parameters(self) -> list[nn.Parameter]:
        """Parameters that the main training optimizer should update (G4).

        Complement of ``get_calibration_parameters`` — every parameter
        on the model that is NOT a calibration scalar.
        """
        return [
            p
            for n, p in self.named_parameters()
            if not self._is_calibration_parameter_name(n)
        ]

    # =====================================================
    # CHECKPOINT
    # =====================================================

    def save_checkpoint(
        self,
        path: Path,
        optimizer_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:

        if not isinstance(path, Path):
            raise TypeError("path must be Path")

        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer_state,
            "metadata": metadata,
        }

        try:
            torch.save(checkpoint, path)
            logger.info("Checkpoint saved: %s", path)
        except Exception as e:
            logger.exception("Checkpoint save failed")
            raise

    def load_checkpoint(
        self,
        path: Path | str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: Optional[str | torch.device] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        # NOTE (A5): default switched from ``strict=False`` to ``True``.
        # Silent acceptance of mismatched checkpoints lets a stale or
        # mislabelled .pt file load against a different head/encoder
        # configuration, producing a "working" model whose predictions
        # silently come from re-initialised weights. Callers that
        # genuinely need partial loads (e.g. fine-tuning a new head on
        # top of a pretrained encoder) must now opt in explicitly.

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        # C1.3: ``weights_only=True`` is the only safe deserialiser for
        # untrusted .pt files. Checkpoints written by this module are
        # plain dicts of tensors plus pickle-safe scalars (``epoch``,
        # ``step``, ``metadata``), so the safer path covers every key
        # we actually emit. Loading with ``weights_only=False`` would
        # execute arbitrary code from the pickle stream — unacceptable
        # for any code path that may consume a checkpoint downloaded
        # from object storage or sent across a network.
        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )

        load_result = self.load_state_dict(
            checkpoint["model_state_dict"],
            strict=strict,
        )

        if load_result.missing_keys:
            logger.warning("Missing keys: %s", load_result.missing_keys)

        if load_result.unexpected_keys:
            logger.warning("Unexpected keys: %s", load_result.unexpected_keys)

        if optimizer and checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info("Checkpoint loaded: %s", path)

        return checkpoint.get("metadata", {})

    # =====================================================
    # FREEZE
    # =====================================================

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        logger.info("Model frozen")

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True
        logger.info("Model unfrozen")

    def freeze_encoder(self) -> None:

        if not hasattr(self, "encoder"):
            raise AttributeError("No encoder")

        for p in self.encoder.parameters():
            p.requires_grad = False

        logger.info("Encoder frozen")

    def unfreeze_encoder(self) -> None:

        if not hasattr(self, "encoder"):
            raise AttributeError("No encoder")

        for p in self.encoder.parameters():
            p.requires_grad = True

        logger.info("Encoder unfrozen")

    def freeze_head(self, task: str) -> None:

        heads = getattr(self, "task_heads", None)

        if heads and task in heads:
            for p in heads[task].parameters():
                p.requires_grad = False
        else:
            attr = f"{task}_head"
            head = getattr(self, attr, None)

            if head is None:
                raise ValueError(f"No head: {task}")

            for p in head.parameters():
                p.requires_grad = False

        logger.info("Head frozen: %s", task)

    # =====================================================
    # META
    # =====================================================

    @property
    def model_type(self) -> str:
        if hasattr(self, "task_heads") or hasattr(self, "heads"):
            return "multitask"
        return "single_task"

    def get_tasks(self) -> list:

        if hasattr(self, "task_heads"):
            return list(self.task_heads.keys())

        if hasattr(self, "heads"):
            return list(self.heads.keys())

        return []

    # =====================================================
    # SUMMARY
    # =====================================================

    def summary(self) -> Dict[str, Any]:

        return {
            "model_class": self.__class__.__name__,
            "device": str(self.device),
            "trainable_parameters": self.num_parameters(True),
            "total_parameters": self.num_parameters(False),
            "model_type": self.model_type,
            "tasks": self.get_tasks(),
        }