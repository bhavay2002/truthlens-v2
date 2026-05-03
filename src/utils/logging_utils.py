#src\utils\logging_utils.py

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler
from threading import Lock

from src.utils.distributed_utils import is_primary


# =========================================================
# GLOBAL CONTEXT (THREAD-SAFE)
# =========================================================

_GLOBAL_CONTEXT: Dict[str, Any] = {}
_CONTEXT_LOCK = Lock()


def set_global_context(**kwargs) -> None:
    """
    Set global logging context.

    Example:
        set_global_context(experiment_id="exp_123", epoch=1)
    """
    with _CONTEXT_LOCK:
        _GLOBAL_CONTEXT.update(kwargs)


def clear_global_context() -> None:
    """Clear all global context."""
    with _CONTEXT_LOCK:
        _GLOBAL_CONTEXT.clear()


def get_global_context() -> Dict[str, Any]:
    with _CONTEXT_LOCK:
        return dict(_GLOBAL_CONTEXT)


# =========================================================
# FORMATTERS
# =========================================================

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "time": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Inject global context
        log_record.update(get_global_context())

        # Inject per-call extra
        if hasattr(record, "extra"):
            log_record.update(record.extra)

        return json.dumps(log_record, ensure_ascii=False)


def _create_text_formatter():
    return logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )


# =========================================================
# CONFIGURATION
# =========================================================

def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
    *,
    json_logging: bool = False,
) -> None:

    root = logging.getLogger()
    root.setLevel(level)

    formatter = JsonFormatter() if json_logging else _create_text_formatter()

    _setup_stream(root, formatter)

    if log_file:
        _setup_file(root, formatter, log_file)


def _setup_stream(logger, formatter):
    if any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def _setup_file(logger, formatter, log_file):
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        path,
        maxBytes=50 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
        delay=True,
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# =========================================================
# STRUCTURED LOGGING API
# =========================================================

def log_event(
    logger: logging.Logger,
    message: str,
    *,
    level: int = logging.INFO,
    **extra: Any,
) -> None:

    if not is_primary():
        return

    logger.log(level, message, extra={"extra": extra})


# =========================================================
# TRAINING LOGGING
# =========================================================

def log_training_step(
    logger: logging.Logger,
    step: int,
    task: str,
    loss: float,
    lr: float,
):

    log_event(
        logger,
        "training_step",
        step=step,
        task=task,
        loss=round(loss, 6),
        lr=lr,
    )


def log_epoch_summary(
    logger: logging.Logger,
    epoch: int,
    metrics: Dict[str, Any],
):

    log_event(
        logger,
        "epoch_summary",
        epoch=epoch,
        metrics=metrics,
    )


# =========================================================
# INFERENCE LOGGING
# =========================================================

def log_prediction(
    logger: logging.Logger,
    task: str,
    confidence: float,
):

    log_event(
        logger,
        "prediction",
        task=task,
        confidence=confidence,
    )