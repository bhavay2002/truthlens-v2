from __future__ import annotations

import logging
import weakref
from typing import Any, Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =========================================================
# P2.5: BREAKDOWN CACHE
# =========================================================
#
# ``layer_parameter_breakdown`` and ``top_k_layers_by_parameters`` walk
# every module in the model on every call and sum ``p.numel()`` for
# each one. Logging / observability paths can call them many times per
# epoch (per-task summary, per-step histograms, eval reports …) which
# is wasted work — for a built model, the per-layer parameter counts
# are constant across the model's lifetime.
#
# We memoise the result keyed by ``id(model)`` via a
# ``WeakValueDictionary`` of marker objects: the cached entry is held
# only as long as the model itself is reachable, so swapping models in
# and out of memory does not leak. Mutating the model (adding /
# removing parameters) is the caller's signal to call
# ``clear_parameter_count_cache`` — the cache will otherwise serve
# stale counts. We expose that function so tests and dynamic-architecture
# callers can invalidate explicitly.

_BREAKDOWN_CACHE: "dict[int, Dict[str, Dict[str, int]]]" = {}
_TOPK_CACHE: "dict[tuple[int, int], Dict[str, int]]" = {}
_KEEPALIVE: "weakref.WeakValueDictionary[int, nn.Module]" = weakref.WeakValueDictionary()


def _register_for_invalidation(model: nn.Module) -> int:
    key = id(model)
    if key not in _KEEPALIVE:
        # Register a finaliser so that when the model is garbage
        # collected we drop its cached entries automatically.
        _KEEPALIVE[key] = model
        weakref.finalize(model, _evict, key)
    return key


def _evict(key: int) -> None:
    _BREAKDOWN_CACHE.pop(key, None)
    for ck in [k for k in _TOPK_CACHE if k[0] == key]:
        _TOPK_CACHE.pop(ck, None)


def clear_parameter_count_cache(model: nn.Module | None = None) -> None:
    """Invalidate cached breakdowns.

    Call after any structural mutation of the model (registering a new
    head, freezing a sub-module, etc.). Pass ``model=None`` to wipe the
    cache for every model.
    """
    if model is None:
        _BREAKDOWN_CACHE.clear()
        _TOPK_CACHE.clear()
        return

    _evict(id(model))


# =========================================================
# BASIC COUNTS
# =========================================================

def count_parameters(model: nn.Module) -> int:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be nn.Module")

    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_frozen_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


# =========================================================
# MEMORY ESTIMATION
# =========================================================

def estimate_model_size_mb(model: nn.Module) -> float:
    """
    Approximate model size in MB (assuming float32 unless specified).
    """

    total_params = count_parameters(model)
    bytes_per_param = 4  # float32

    return (total_params * bytes_per_param) / (1024 ** 2)


# =========================================================
# SUMMARY
# =========================================================

def parameter_summary(model: nn.Module) -> Dict[str, Any]:

    total = count_parameters(model)
    trainable = count_trainable_parameters(model)
    frozen = count_frozen_parameters(model)

    summary = {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "trainable_ratio": trainable / total if total > 0 else 0.0,
        "model_size_mb": estimate_model_size_mb(model),
    }

    logger.info(
        "Params | total=%d | trainable=%d | frozen=%d | size=%.2fMB",
        total,
        trainable,
        frozen,
        summary["model_size_mb"],
    )

    return summary


# =========================================================
# LAYER BREAKDOWN
# =========================================================

def layer_parameter_breakdown(model: nn.Module) -> Dict[str, Dict[str, int]]:
    # P2.5: serve from cache when the same model object is asked for
    # the breakdown more than once. See module-level note for the
    # invalidation contract.
    key = _register_for_invalidation(model)
    cached = _BREAKDOWN_CACHE.get(key)
    if cached is not None:
        return cached

    breakdown: Dict[str, Dict[str, int]] = {}

    for name, module in model.named_modules():

        params = sum(p.numel() for p in module.parameters(recurse=False))
        trainable = sum(
            p.numel() for p in module.parameters(recurse=False) if p.requires_grad
        )

        if params > 0:
            breakdown[name] = {
                "total": params,
                "trainable": trainable,
                "frozen": params - trainable,
            }

    _BREAKDOWN_CACHE[key] = breakdown
    return breakdown


# =========================================================
# TOP-K HEAVIEST LAYERS
# =========================================================

def top_k_layers_by_parameters(
    model: nn.Module,
    k: int = 10,
) -> Dict[str, int]:

    # P2.5: same caching strategy as ``layer_parameter_breakdown`` but
    # keyed on (id(model), k) since callers commonly request multiple
    # ``k`` values for the same model.
    model_key = _register_for_invalidation(model)
    cache_key = (model_key, k)
    cached = _TOPK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    layer_counts = {
        name: sum(p.numel() for p in module.parameters(recurse=False))
        for name, module in model.named_modules()
    }

    sorted_layers = sorted(
        layer_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    result = dict(sorted_layers[:k])
    _TOPK_CACHE[cache_key] = result
    return result