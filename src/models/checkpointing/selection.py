from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


# =========================================================
# METRIC COMPARISON
# =========================================================

def is_better(
    current: float,
    best: Optional[float],
    *,
    mode: str = "min",
) -> bool:
    """
    Compare metric values.

    Parameters
    ----------
    current : float
    best : float | None
    mode : "min" | "max"

    Returns
    -------
    bool
    """

    if best is None:
        return True

    if mode == "min":
        return current < best
    elif mode == "max":
        return current > best
    else:
        raise ValueError(f"Invalid mode: {mode}")


# =========================================================
# BEST METRIC TRACKING
# =========================================================

def load_best_metric(path: Path) -> Optional[float]:
    """
    Load best metric from file.
    """

    path = Path(path)

    if not path.exists():
        return None

    try:
        return float(path.read_text().strip())
    except Exception:
        logger.warning("Failed to read best metric file: %s", path)
        return None


def save_best_metric(path: Path, value: float) -> None:
    """
    Save best metric to file.
    """

    path = Path(path)

    try:
        path.write_text(f"{value:.10f}")
    except Exception:
        logger.exception("Failed to save best metric: %s", path)
        raise


# =========================================================
# UPDATE BEST CHECKPOINT
# =========================================================

def update_best_checkpoint(
    checkpoint_path: Path,
    *,
    metric: float,
    metric_name: str = "val_loss",
    mode: str = "min",
    best_file: str = "best_metric.txt",
    symlink_name: str = "best.pt",
) -> bool:
    """
    Update best checkpoint if improved.

    Features
    --------
    - Metric comparison
    - Persistent tracking
    - Symlink to best model

    Returns
    -------
    bool
        True if updated
    """

    checkpoint_path = Path(checkpoint_path)
    directory = checkpoint_path.parent

    best_metric_path = directory / best_file

    best_metric = load_best_metric(best_metric_path)

    if not is_better(metric, best_metric, mode=mode):
        return False

    # -----------------------------------------------------
    # SAVE NEW BEST METRIC
    # -----------------------------------------------------

    save_best_metric(best_metric_path, metric)

    # -----------------------------------------------------
    # UPDATE SYMLINK
    # -----------------------------------------------------

    best_link = directory / symlink_name

    try:
        if best_link.exists() or best_link.is_symlink():
            best_link.unlink()

        best_link.symlink_to(checkpoint_path.name)

    except Exception:
        logger.warning(
            "Symlink update failed (non-critical): %s -> %s",
            best_link,
            checkpoint_path,
        )

    logger.info(
        "New BEST checkpoint | %s=%.6f | path=%s",
        metric_name,
        metric,
        checkpoint_path,
    )

    return True


# =========================================================
# METADATA-BASED SELECTION (ADVANCED)
# =========================================================

def select_best_from_metadata(
    checkpoints: Dict[Path, Dict[str, Any]],
    *,
    metric_name: str = "val_loss",
    mode: str = "min",
) -> Optional[Path]:
    """
    Select best checkpoint from in-memory metadata.

    Parameters
    ----------
    checkpoints : dict[path → metadata]
    metric_name : str
    mode : "min" | "max"

    Returns
    -------
    Path | None
    """

    best_path = None
    best_score = None

    for path, meta in checkpoints.items():

        metrics = meta.get("metrics", {})

        if metric_name not in metrics:
            continue

        score = float(metrics[metric_name])

        if best_score is None or is_better(score, best_score, mode=mode):
            best_score = score
            best_path = path

    if best_path:
        logger.debug(
            "Best checkpoint selected (in-memory) | %s=%.6f | path=%s",
            metric_name,
            best_score,
            best_path,
        )

    return best_path