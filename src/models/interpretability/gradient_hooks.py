from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =========================================================
# GRADIENT HOOKS
# =========================================================

class GradientHookManager:
    """
    Manages forward + backward hooks for capturing:
        - activations
        - gradients

    Useful for:
        - saliency maps
        - Grad-CAM
        - integrated gradients
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

        self.activations: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}

    # =====================================================
    # REGISTER HOOKS
    # =====================================================

    def register_hooks(
        self,
        target_modules: List[str],
    ) -> None:
        """
        Register hooks on modules by name.
        """

        for name, module in self.model.named_modules():

            if any(t in name for t in target_modules):

                self._register_forward_hook(name, module)
                self._register_backward_hook(name, module)

                logger.info(f"[HOOK] registered on {name}")

    def _register_forward_hook(self, name: str, module: nn.Module):

        def forward_hook(module, inputs, output):
            self.activations[name] = output.detach()

        handle = module.register_forward_hook(forward_hook)
        self.handles.append(handle)

    def _register_backward_hook(self, name: str, module: nn.Module):

        def backward_hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0].detach()

        handle = module.register_full_backward_hook(backward_hook)
        self.handles.append(handle)

    # =====================================================
    # CLEAR / REMOVE
    # =====================================================

    def clear(self) -> None:
        self.activations.clear()
        self.gradients.clear()

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

        logger.info("[HOOK] all hooks removed")

    # =====================================================
    # ACCESS
    # =====================================================

    def get_activations(self) -> Dict[str, torch.Tensor]:
        return self.activations

    def get_gradients(self) -> Dict[str, torch.Tensor]:
        return self.gradients


# =========================================================
# SALIENCY MAP
# =========================================================

def compute_saliency(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_index: Optional[int] = None,
    task: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute gradient-based saliency map.

    Returns:
        saliency: (B, T)
    """

    model.eval()

    input_ids = input_ids.clone().detach().requires_grad_(True)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        task=task,
    )

    logits = outputs["logits"]

    if target_index is None:
        target_index = torch.argmax(logits, dim=-1)

    selected = logits.gather(1, target_index.unsqueeze(1)).squeeze()

    selected.backward(torch.ones_like(selected))

    saliency = input_ids.grad.abs()

    return saliency


# =========================================================
# GRAD-CAM (TOKEN LEVEL)
# =========================================================

def compute_gradcam(
    hook_manager: GradientHookManager,
    target_layer: str,
) -> torch.Tensor:
    """
    Compute Grad-CAM using captured hooks.

    Returns:
        cam: (B, T)
    """

    activations = hook_manager.activations.get(target_layer)
    gradients = hook_manager.gradients.get(target_layer)

    if activations is None or gradients is None:
        raise ValueError("Hooks not found for layer")

    # global average pooling over hidden dim
    weights = torch.mean(gradients, dim=-1, keepdim=True)

    cam = torch.sum(weights * activations, dim=-1)

    cam = torch.relu(cam)

    # normalize
    cam = cam / (cam.max(dim=1, keepdim=True).values + 1e-12)

    return cam


# =========================================================
# INTEGRATED GRADIENTS
# =========================================================

def integrated_gradients(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
    steps: int = 50,
    task: Optional[str] = None,
) -> torch.Tensor:
    """
    Integrated Gradients.

    Returns:
        attributions: (B, T)
    """

    model.eval()

    if baseline is None:
        baseline = torch.zeros_like(input_ids)

    scaled_inputs = [
        baseline + (float(i) / steps) * (input_ids - baseline)
        for i in range(steps + 1)
    ]

    grads = []

    for inp in scaled_inputs:

        inp = inp.clone().detach().requires_grad_(True)

        outputs = model(
            input_ids=inp,
            attention_mask=attention_mask,
            task=task,
        )

        logits = outputs["logits"]
        target = torch.argmax(logits, dim=-1)

        selected = logits.gather(1, target.unsqueeze(1)).squeeze()

        selected.backward(torch.ones_like(selected))

        grads.append(inp.grad.detach())

    avg_grads = torch.mean(torch.stack(grads), dim=0)

    attributions = (input_ids - baseline) * avg_grads

    return attributions.abs()