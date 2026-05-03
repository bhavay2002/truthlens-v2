from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


# =========================================================
# CORE PATH RESOLUTION
# =========================================================

def resolve_checkpoint_path(
    path: str | Path,
    *,
    must_exist: bool = True,
) -> Path:
    """
    Resolve checkpoint path from:
    - file path
    - directory (auto-picks latest)
    - None → error

    Parameters
    ----------
    path : str | Path
    must_exist : bool

    Returns
    -------
    Path
    """

    if path is None:
        raise ValueError("Checkpoint path cannot be None")

    path = Path(path)

    # -----------------------------------------------------
    # CASE 1: FILE
    # -----------------------------------------------------
    if path.is_file():
        return path

    # -----------------------------------------------------
    # CASE 2: DIRECTORY → find latest
    # -----------------------------------------------------
    if path.is_dir():
        latest = get_latest_checkpoint(path)
        if latest is None:
            raise FileNotFoundError(f"No checkpoints found in directory: {path}")
        return latest

    # -----------------------------------------------------
    # CASE 3: DOES NOT EXIST
    # -----------------------------------------------------
    if must_exist:
        raise FileNotFoundError(path)

    return path


# =========================================================
# DISCOVERY
# =========================================================

def list_checkpoints(
    directory: str | Path,
    *,
    extensions: tuple = (".pt", ".pth", ".ckpt"),
) -> List[Path]:
    """
    List all checkpoint files in directory.
    """

    directory = Path(directory)

    if not directory.exists():
        return []

    files = []

    for ext in extensions:
        files.extend(directory.glob(f"*{ext}"))

    # sort by modification time (newest first)
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return files


def get_latest_checkpoint(
    directory: str | Path,
) -> Optional[Path]:
    """
    Return most recent checkpoint file.
    """

    files = list_checkpoints(directory)

    if not files:
        return None

    latest = files[0]

    logger.debug("Latest checkpoint resolved: %s", latest)

    return latest


# =========================================================
# BEST CHECKPOINT (BY METADATA)
# =========================================================

def find_best_checkpoint(
    directory: str | Path,
    *,
    metric_name: str = "val_loss",
    mode: str = "min",
) -> Optional[Path]:
    """
    Find best checkpoint based on metadata.json files.

    Assumes:
    - each checkpoint has a metadata file:
        checkpoint.pt → checkpoint.meta.json

    Parameters
    ----------
    metric_name : str
    mode : "min" | "max"
    """

    from .metadata import load_metadata  # lazy import

    directory = Path(directory)

    checkpoints = list_checkpoints(directory)

    if not checkpoints:
        return None

    best_path = None
    best_score = None

    for ckpt in checkpoints:

        meta_path = ckpt.with_suffix(".meta.json")

        if not meta_path.exists():
            continue

        try:
            meta = load_metadata(meta_path)
            metrics = meta.get("metrics", {})

            if metric_name not in metrics:
                continue

            score = float(metrics[metric_name])

            if best_score is None:
                best_score = score
                best_path = ckpt
                continue

            if mode == "min" and score < best_score:
                best_score = score
                best_path = ckpt

            elif mode == "max" and score > best_score:
                best_score = score
                best_path = ckpt

        except Exception:
            logger.warning("Skipping invalid metadata: %s", ckpt)

    if best_path:
        logger.info(
            "Best checkpoint found | metric=%s | score=%.6f | path=%s",
            metric_name,
            best_score,
            best_path,
        )

    return best_path


# =========================================================
# FLEXIBLE RESOLVE (LATEST / BEST / EXPLICIT)
# =========================================================

def resolve(
    path: str | Path,
    *,
    strategy: str = "latest",  # "latest" | "best" | "explicit"
    metric_name: str = "val_loss",
    mode: str = "min",
) -> Path:
    """
    High-level resolver.

    Examples:
    ---------
    resolve("checkpoints/", strategy="latest")
    resolve("checkpoints/", strategy="best")
    resolve("model.pt", strategy="explicit")
    """

    path = Path(path)

    if strategy == "explicit":
        return resolve_checkpoint_path(path)

    if not path.is_dir():
        raise ValueError("For 'latest' or 'best', path must be a directory")

    if strategy == "latest":
        latest = get_latest_checkpoint(path)
        if latest is None:
            raise FileNotFoundError(f"No checkpoints in {path}")
        return latest

    if strategy == "best":
        best = find_best_checkpoint(
            path,
            metric_name=metric_name,
            mode=mode,
        )
        if best is None:
            raise FileNotFoundError(f"No valid 'best' checkpoint in {path}")
        return best

    raise ValueError(f"Unknown resolve strategy: {strategy}")