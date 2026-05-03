#src\utils\time_utils.py#

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Callable, Any, Tuple, TypeVar, Optional, Dict

import torch

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =========================================================
# GLOBAL PROFILING SWITCH
# =========================================================

_PROFILING_ENABLED: bool = False


def enable_profiling(enabled: bool = True) -> None:
    """
    Enable / disable profiling globally.
    """
    global _PROFILING_ENABLED
    _PROFILING_ENABLED = enabled


def is_profiling_enabled() -> bool:
    return _PROFILING_ENABLED


# =========================================================
# TIMESTAMP
# =========================================================

def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


def current_datetime() -> datetime:
    return datetime.now(timezone.utc)


# =========================================================
# GPU SYNC (CRITICAL)
# =========================================================

def _sync_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# =========================================================
# FUNCTION TIMER
# =========================================================

def measure_runtime(
    func: Callable[..., T],
    *args: Any,
    sync_cuda: bool = True,
    log: bool = False,
    **kwargs: Any,
) -> Tuple[T, float]:

    if not _PROFILING_ENABLED:
        return func(*args, **kwargs), 0.0

    if sync_cuda:
        _sync_if_needed()

    start = time.perf_counter()

    try:
        result = func(*args, **kwargs)
    except Exception:
        logger.exception("Error in %s", getattr(func, "__name__", "unknown"))
        raise

    if sync_cuda:
        _sync_if_needed()

    runtime = time.perf_counter() - start

    if log:
        logger.debug(
            "Runtime | function=%s | time=%.6f",
            getattr(func, "__name__", "unknown"),
            runtime,
        )

    return result, runtime


# =========================================================
# DECORATOR
# =========================================================

def timeit(sync_cuda: bool = True):
    def decorator(func: Callable[..., T]):
        def wrapper(*args, **kwargs):
            if not _PROFILING_ENABLED:
                return func(*args, **kwargs)

            _, runtime = measure_runtime(
                func, *args, sync_cuda=sync_cuda, log=True, **kwargs
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


# =========================================================
# CONTEXT TIMER
# =========================================================

class Timer:
    def __init__(
        self,
        name: Optional[str] = None,
        *,
        sync_cuda: bool = True,
        logger_enabled: bool = True,
    ):
        self.name = name or "timer"
        self.sync_cuda = sync_cuda
        self.logger_enabled = logger_enabled
        self._start: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self):
        if not _PROFILING_ENABLED:
            return self

        if self.sync_cuda:
            _sync_if_needed()

        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not _PROFILING_ENABLED or self._start is None:
            return

        if self.sync_cuda:
            _sync_if_needed()

        self.elapsed = time.perf_counter() - self._start

        if self.logger_enabled:
            logger.debug(
                "Timer | name=%s | elapsed=%.6f",
                self.name,
                self.elapsed,
            )


# =========================================================
# MULTI-TIMER (AGGREGATION)
# =========================================================

class MultiTimer:
    def __init__(self):
        self.times: Dict[str, list[float]] = {}

    def add(self, name: str, value: float):
        if not _PROFILING_ENABLED:
            return

        self.times.setdefault(name, []).append(value)

    def summary(self) -> Dict[str, float]:
        return {
            k: sum(v) / len(v)
            for k, v in self.times.items()
            if v
        }

    def total(self) -> Dict[str, float]:
        return {
            k: sum(v)
            for k, v in self.times.items()
        }

    def reset(self):
        self.times.clear()