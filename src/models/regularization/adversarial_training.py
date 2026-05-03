from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# C1.5: EMBEDDING TARGET RESOLUTION
# =========================================================
#
# Previously every adversarial method below identified the parameter to
# perturb with ``self.emb_name in name``. Substring matching is wrong:
# the default ``emb_name="embedding"`` matches *any* parameter whose
# fully-qualified name happens to contain the substring (e.g.
# ``encoder.embeddings.LayerNorm.weight``,
# ``encoder.embeddings.position_embeddings.weight``,
# ``encoder.embeddings.token_type_embeddings.weight``,
# ``encoder.embeddings.word_embeddings.weight`` — and several
# downstream task heads). The intended target is the *input* word
# embedding matrix only. Perturbing LayerNorm weights or positional
# embeddings is not FGM/PGD/FreeAT and produces meaningless gradient
# signal for adversarial training.
#
# Correct behaviour: resolve the target via
# ``model.get_input_embeddings().weight`` and key the backup by
# ``id(param)`` (parameter identity is the stable key — module names
# can collide when the same parameter is registered under multiple
# attribute paths, and ``name`` cannot be re-used as a dict key for an
# anonymous tensor).
#
# We retain ``emb_name`` as a *fallback* and as a back-compat knob so
# callers that previously relied on substring matching against custom
# parameter names still have an escape hatch — but we prefer the
# canonical ``get_input_embeddings`` path whenever it is available.


def _resolve_embedding_param(
    model: nn.Module,
    fallback_name: str,
) -> Optional[torch.nn.Parameter]:
    """Resolve the input-embedding parameter for adversarial perturbation.

    Order of preference:
      1) ``model.get_input_embeddings().weight`` — the canonical
         transformers / nn.Embedding contract.
      2) Direct attribute walk for nested ``encoder`` /
         ``transformer`` containers that themselves expose
         ``get_input_embeddings``.
      3) Last-resort substring fallback against ``fallback_name`` so
         legacy custom architectures keep working.
    """

    # (1) Canonical path.
    getter = getattr(model, "get_input_embeddings", None)
    if callable(getter):
        try:
            emb = getter()
            if isinstance(emb, nn.Module) and hasattr(emb, "weight"):
                return emb.weight
        except Exception:
            pass

    # (2) Common wrappers.
    for attr in ("encoder", "transformer", "model", "bert", "roberta"):
        sub = getattr(model, attr, None)
        if sub is None:
            continue
        sub_getter = getattr(sub, "get_input_embeddings", None)
        if callable(sub_getter):
            try:
                emb = sub_getter()
                if isinstance(emb, nn.Module) and hasattr(emb, "weight"):
                    return emb.weight
            except Exception:
                continue

    # (3) Fallback: substring match (legacy behaviour). We restrict the
    # match to parameters whose *trailing* name is exactly ``weight``
    # AND whose path contains the requested keyword, which excludes
    # LayerNorm / bias parameters that previously got swept up.
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if fallback_name not in name:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf != "weight":
            continue
        if "LayerNorm" in name or "layer_norm" in name or "norm" in name.split("."):
            continue
        return param

    return None


# =========================================================
# FGM (Fast Gradient Method)
# =========================================================

class FGM:
    """
    Fast Gradient Method for adversarial training.

    Perturbs the input embedding matrix in the direction of its gradient.
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 1e-5,
        emb_name: str = "embedding",
    ) -> None:
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        # Keyed by id(param) — see module-level note (C1.5).
        self.backup: Dict[int, torch.Tensor] = {}

    def attack(self) -> None:
        """
        Add adversarial perturbation to the input embedding matrix.
        """
        param = _resolve_embedding_param(self.model, self.emb_name)
        if param is None or not param.requires_grad or param.grad is None:
            return

        grad = param.grad
        norm = torch.norm(grad)
        if norm == 0 or not torch.isfinite(norm):
            return

        # ``with torch.no_grad()`` rather than ``param.data.add_(...)``:
        # ``.data`` bypasses the autograd version counter, which means
        # any saved tensors from the *previous* backward pass that
        # reference this leaf are now silently inconsistent. The
        # ``no_grad`` context is the documented, autograd-aware way to
        # mutate a parameter in place.
        with torch.no_grad():
            self.backup[id(param)] = param.detach().clone()
            r_at = self.epsilon * grad / (norm + EPS)
            param.add_(r_at)

    def restore(self) -> None:
        """
        Restore the original embedding values.
        """
        param = _resolve_embedding_param(self.model, self.emb_name)
        if param is None:
            self.backup.clear()
            return

        backup = self.backup.pop(id(param), None)
        if backup is None:
            return

        with torch.no_grad():
            param.copy_(backup)

        self.backup.clear()


# =========================================================
# PGD (Projected Gradient Descent)
# =========================================================

class PGD:
    """
    Multi-step adversarial training (projected gradient descent).
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 1e-5,
        alpha: float = 1e-6,
        steps: int = 3,
        emb_name: str = "embedding",
    ) -> None:
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.emb_name = emb_name

        # Keyed by id(param) — see module-level note (C1.5).
        self.emb_backup: Dict[int, torch.Tensor] = {}
        self.grad_backup: Dict[str, torch.Tensor] = {}

    def attack(self, is_first_attack: bool = False) -> None:
        param = _resolve_embedding_param(self.model, self.emb_name)
        if param is None or not param.requires_grad or param.grad is None:
            return

        grad = param.grad
        norm = torch.norm(grad)
        if norm == 0 or not torch.isfinite(norm):
            return

        with torch.no_grad():
            if is_first_attack:
                self.emb_backup[id(param)] = param.detach().clone()

            r_at = self.alpha * grad / (norm + EPS)
            param.add_(r_at)
            param.copy_(self._project(id(param), param.detach()))

    def restore(self) -> None:
        param = _resolve_embedding_param(self.model, self.emb_name)
        if param is None:
            self.emb_backup.clear()
            return

        backup = self.emb_backup.pop(id(param), None)
        if backup is not None:
            with torch.no_grad():
                param.copy_(backup)

        self.emb_backup.clear()

    def backup_grad(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.detach().clone()

    def restore_grad(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.grad_backup:
                param.grad = self.grad_backup[name]
        self.grad_backup.clear()

    def _project(self, param_id: int, param_data: torch.Tensor) -> torch.Tensor:
        """
        Project perturbation onto the epsilon ball around the original.
        """
        original = self.emb_backup.get(param_id)
        if original is None:
            return param_data

        r = param_data - original
        r_norm = torch.norm(r)
        if r_norm > self.epsilon:
            r = self.epsilon * r / (r_norm + EPS)
        return original + r


# =========================================================
# FREE ADVERSARIAL TRAINING (FAST)
# =========================================================

class FreeAT:
    """
    Free Adversarial Training (reuses gradient).
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 1e-5,
        emb_name: str = "embedding",
    ) -> None:
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        # Keyed by id(param) — see module-level note (C1.5).
        self.delta: Dict[int, torch.Tensor] = {}

    def attack(self) -> None:
        param = _resolve_embedding_param(self.model, self.emb_name)
        if param is None or not param.requires_grad or param.grad is None:
            return

        grad = param.grad
        norm = torch.norm(grad)
        if norm == 0 or not torch.isfinite(norm):
            return

        with torch.no_grad():
            if id(param) not in self.delta:
                self.delta[id(param)] = torch.zeros_like(param.detach())

            self.delta[id(param)] += self.epsilon * grad / (norm + EPS)
            param.add_(self.delta[id(param)])

    def restore(self) -> None:
        param = _resolve_embedding_param(self.model, self.emb_name)
        if param is None:
            self.delta.clear()
            return

        d = self.delta.pop(id(param), None)
        if d is not None:
            with torch.no_grad():
                param.sub_(d)

        self.delta.clear()


# =========================================================
# ADVERSARIAL TRAINING WRAPPER
# =========================================================

class AdversarialTrainer:
    """
    Unified adversarial training wrapper.

    Supports:
        - fgm
        - pgd
        - free
    """

    def __init__(
        self,
        model: nn.Module,
        method: str = "fgm",
        epsilon: float = 1e-5,
        alpha: float = 1e-6,
        steps: int = 3,
        emb_name: str = "embedding",
    ) -> None:

        self.method = method

        if method == "fgm":
            self.strategy = FGM(model, epsilon, emb_name)

        elif method == "pgd":
            self.strategy = PGD(model, epsilon, alpha, steps, emb_name)

        elif method == "free":
            self.strategy = FreeAT(model, epsilon, emb_name)

        else:
            raise ValueError(f"Unsupported method: {method}")

    def attack(self, **kwargs) -> None:
        self.strategy.attack(**kwargs)

    def restore(self) -> None:
        self.strategy.restore()

    def backup_grad(self) -> None:
        if hasattr(self.strategy, "backup_grad"):
            self.strategy.backup_grad()

    def restore_grad(self) -> None:
        if hasattr(self.strategy, "restore_grad"):
            self.strategy.restore_grad()
