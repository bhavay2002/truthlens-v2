#src\utils\seed_utils.py

from __future__ import annotations

import logging
import os
import random
from typing import Optional

import numpy as np
# torch is imported lazily inside functions to avoid a ~5 s startup penalty
# every time this module is imported.

logger = logging.getLogger(__name__)


# =========================================================
# MAIN ENTRY
# =========================================================

def set_seed(
    seed: int = 42,
    *,
    deterministic: Optional[bool] = None,
    enable_tf32: bool = True,
    matmul_precision: str = "high",
    rank: int = 0,
) -> int:
    """
    Set global seed with DDP support.

    Returns
    -------
    int
        Final seed used (seed + rank)
    """

    import torch

    if not isinstance(seed, int):
        raise TypeError("seed must be int")

    final_seed = seed + rank

    # -----------------------------
    # Core Seeding
    # -----------------------------
    random.seed(final_seed)
    np.random.seed(final_seed)
    torch.manual_seed(final_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(final_seed)
        torch.cuda.manual_seed_all(final_seed)

    # -----------------------------
    # Determinism
    # -----------------------------
    if deterministic is None:
        deterministic = os.environ.get("TRUTHLENS_DETERMINISTIC", "0") == "1"

    if deterministic:
        _set_deterministic()
    else:
        _set_fast_mode(enable_tf32, matmul_precision)

    logger.info(
        "Seed initialized | base=%d | rank=%d | final=%d | deterministic=%s",
        seed,
        rank,
        final_seed,
        deterministic,
    )

    return final_seed


# =========================================================
# DETERMINISTIC MODE
# =========================================================

def _set_deterministic():
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        logger.warning("Deterministic algorithms not fully supported")

    logger.debug("Deterministic mode enabled")


# =========================================================
# FAST MODE
# =========================================================

def _set_fast_mode(enable_tf32: bool, matmul_precision: str):
    import torch
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
        torch.backends.cudnn.allow_tf32 = enable_tf32

    try:
        torch.set_float32_matmul_precision(matmul_precision)
    except Exception:
        logger.warning("Matmul precision setting not supported")

    logger.debug(
        "Fast mode | TF32=%s | precision=%s",
        enable_tf32,
        matmul_precision,
    )


# =========================================================
# FULL REPRO MODE (RESEARCH)
# =========================================================

def enable_full_reproducibility(seed: int = 42, rank: int = 0):
    """
    Strict reproducibility mode for research.
    """
    return set_seed(
        seed,
        deterministic=True,
        enable_tf32=False,
        matmul_precision="highest",
        rank=rank,
    )


# =========================================================
# DATALOADER SUPPORT
# =========================================================

def seed_worker(worker_id: int):
    import torch
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def create_generator(seed: int) -> "torch.Generator":
    import torch
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# =========================================================
# DEBUG / TRACKING
# =========================================================

def get_seed_state() -> dict:
    import torch
    return {
        "python_seed": os.environ.get("PYTHONHASHSEED"),
        "torch_seed": torch.initial_seed(),
        "cuda": torch.cuda.is_available(),
        "deterministic": torch.backends.cudnn.deterministic,
    }