from __future__ import annotations

from typing import Optional, Tuple

import torch


# =========================================================
# C1.6: REPRODUCIBLE MIXUP RNG
# =========================================================
#
# Previously ``_sample_lambda`` used ``np.random.beta`` and
# ``_shuffle_indices`` used a generator-less ``torch.randperm``. Two
# problems:
#
#   1. NumPy's global RNG is *not* synchronised with PyTorch's RNG.
#      Setting ``torch.manual_seed(SEED)`` (the standard reproducibility
#      knob) does not seed numpy, so the lambda drawn for each batch is
#      effectively unsynchronised — runs that should be byte-identical
#      diverge in their loss curves.
#
#   2. ``torch.randperm`` without a ``generator=`` argument falls back
#      to the *default* CUDA / CPU RNG, which is shared with the rest
#      of the model (dropout, augmentations). Two consecutive calls can
#      perturb the global state of unrelated stochastic ops.
#
# Both samplers below now route through ``torch`` and accept an explicit
# ``torch.Generator`` so distributed callers can derive a per-rank
# deterministic stream and trainers can pin reproducibility on a single
# RNG handle.

def _sample_lambda(
    alpha: float,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> float:
    if alpha <= 0:
        return 1.0

    # ``torch.distributions.Beta`` does not accept a ``generator`` kwarg
    # at sampling time, so we hand-roll the sampler from two Gamma
    # draws via ``empty().exponential_(...)``-style primitives that DO
    # accept a generator. Beta(α, α) = G1 / (G1 + G2) where G_i ~
    # Gamma(α, 1); ``torch.empty(()).gamma_(α, generator=...)`` is the
    # generator-aware Gamma sampler exposed by PyTorch.
    a = torch.tensor(float(alpha), device=device or torch.device("cpu"))

    if hasattr(torch.Tensor, "gamma_") and generator is not None:
        try:
            g1 = torch.empty((), device=a.device).gamma_(
                a.item(), generator=generator
            )
            g2 = torch.empty((), device=a.device).gamma_(
                a.item(), generator=generator
            )
            return float(g1 / (g1 + g2))
        except (RuntimeError, AttributeError, TypeError):
            # Fall through to the global-RNG path below.
            pass

    # Generator-less path: still uses the *torch* RNG (so
    # ``torch.manual_seed`` controls reproducibility), unlike the
    # previous numpy implementation.
    sample = torch.distributions.Beta(a, a).sample()
    return float(sample.item())


def _shuffle_indices(
    batch_size: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if generator is not None:
        # ``randperm`` requires the generator's device to match the
        # output device — fall back to CPU + ``.to(device)`` when the
        # generator lives on CPU, which is the common case.
        try:
            return torch.randperm(
                batch_size, device=device, generator=generator
            )
        except RuntimeError:
            return torch.randperm(
                batch_size, device=generator.device, generator=generator
            ).to(device)
    return torch.randperm(batch_size, device=device)


# =========================================================
# STANDARD MIXUP (INPUT-LEVEL)
# =========================================================

def mixup(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.2,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Perform MixUp on inputs and targets.

    Args:
        inputs: (B, ...)
        targets: (B, C) or (B,)
        alpha: Beta distribution parameter
        generator: optional torch.Generator for reproducible sampling

    Returns:
        mixed_inputs
        targets_a
        targets_b
        lam
    """

    device = inputs.device
    batch_size = inputs.size(0)

    lam = _sample_lambda(alpha, device=device, generator=generator)

    index = _shuffle_indices(batch_size, device, generator=generator)

    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]

    targets_a = targets
    targets_b = targets[index]

    return mixed_inputs, targets_a, targets_b, lam


# =========================================================
# EMBEDDING-LEVEL MIXUP
# =========================================================

def embedding_mixup(
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.2,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    MixUp applied to hidden representations.

    Args:
        embeddings: (B, H) or (B, T, H)
        targets: labels
        generator: optional torch.Generator for reproducible sampling

    Returns:
        mixed_embeddings, targets_a, targets_b, lam
    """

    device = embeddings.device
    batch_size = embeddings.size(0)

    lam = _sample_lambda(alpha, device=device, generator=generator)

    index = _shuffle_indices(batch_size, device, generator=generator)

    mixed_embeddings = lam * embeddings + (1 - lam) * embeddings[index]

    return mixed_embeddings, targets, targets[index], lam


# =========================================================
# LOSS WRAPPER
# =========================================================

def mixup_loss(
    criterion,
    preds: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Compute MixUp loss.

    Args:
        criterion: loss function
        preds: model predictions
        targets_a: original targets
        targets_b: shuffled targets
        lam: mix coefficient
    """

    loss_a = criterion(preds, targets_a)
    loss_b = criterion(preds, targets_b)

    return lam * loss_a + (1 - lam) * loss_b


# =========================================================
# MULTILABEL SUPPORT
# =========================================================

def mixup_multilabel(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.2,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MixUp for multilabel classification.

    Targets are mixed directly.

    Returns:
        mixed_inputs, mixed_targets
    """

    device = inputs.device
    batch_size = inputs.size(0)

    lam = _sample_lambda(alpha, device=device, generator=generator)

    index = _shuffle_indices(batch_size, device, generator=generator)

    mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
    mixed_targets = lam * targets + (1 - lam) * targets[index]

    return mixed_inputs, mixed_targets


# =========================================================
# TOKEN-LEVEL MIXUP (NLP)
# =========================================================

def token_mixup(
    embeddings: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    alpha: float = 0.2,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, float]:
    """
    MixUp at token level (sequence-wise interpolation).

    Args:
        embeddings: (B, T, H)
        attention_mask: optional mask
        generator: optional torch.Generator for reproducible sampling

    Returns:
        mixed_embeddings, lam
    """

    device = embeddings.device
    batch_size = embeddings.size(0)

    lam = _sample_lambda(alpha, device=device, generator=generator)
    index = _shuffle_indices(batch_size, device, generator=generator)

    mixed = lam * embeddings + (1 - lam) * embeddings[index]

    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1)
        mixed = mixed * mask

    return mixed, lam
