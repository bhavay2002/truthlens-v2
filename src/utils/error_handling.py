from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Any, Optional, Dict

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class ErrorHandlingConfig:
    """
    Controls global error handling behavior.
    """
    raise_exceptions: bool = False   # True = fail fast
    log_traceback: bool = True
    return_on_failure: Any = None


# =========================================================
# CORE ERROR WRAPPER
# =========================================================

def safe_execute(
    fn: Callable,
    *args,
    config: Optional[ErrorHandlingConfig] = None,
    context: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Execute a function safely with standardized error handling.
    """

    config = config or ErrorHandlingConfig()

    try:
        return fn(*args, **kwargs)

    except Exception as e:

        msg = f"[ERROR] {context or fn.__name__}: {str(e)}"

        if config.log_traceback:
            logger.error(msg)
            logger.error(traceback.format_exc())
        else:
            logger.error(msg)

        if config.raise_exceptions:
            raise

        return config.return_on_failure


# =========================================================
# DECORATOR
# =========================================================

def safe(
    *,
    context: Optional[str] = None,
    config: Optional[ErrorHandlingConfig] = None,
):
    """
    Decorator version of safe_execute.
    """

    def decorator(fn: Callable):

        @wraps(fn)
        def wrapper(*args, **kwargs):
            return safe_execute(
                fn,
                *args,
                context=context or fn.__name__,
                config=config,
                **kwargs,
            )

        return wrapper

    return decorator


# =========================================================
# PIPELINE STAGE GUARD (CRITICAL)
# =========================================================

def guarded_stage(
    name: str,
    *,
    critical: bool = False,
):
    """
    Decorator for pipeline stages.

    critical=True → crash pipeline
    critical=False → continue execution
    """

    def decorator(fn: Callable):

        @wraps(fn)
        def wrapper(*args, **kwargs):

            try:
                return fn(*args, **kwargs)

            except Exception as e:

                logger.exception(f"[STAGE FAILED] {name}")

                if critical:
                    raise RuntimeError(f"Critical stage failed: {name}") from e

                return None

        return wrapper

    return decorator


# =========================================================
# ERROR REPORT (STRUCTURED)
# =========================================================

def build_error_report(
    *,
    error: Exception,
    stage: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create structured error object for logging/tracking.
    """

    return {
        "stage": stage,
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
        "metadata": metadata or {},
    }


# =========================================================
# RETRY MECHANISM (OPTIONAL)
# =========================================================

def retry(
    retries: int = 3,
    exceptions: tuple = (Exception,),
):
    """
    Retry decorator for transient failures.
    """

    def decorator(fn: Callable):

        @wraps(fn)
        def wrapper(*args, **kwargs):

            last_error = None

            for attempt in range(retries):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    logger.warning(
                        "Retry %d/%d failed for %s",
                        attempt + 1,
                        retries,
                        fn.__name__,
                    )

            logger.error("All retries failed for %s", fn.__name__)
            raise last_error

        return wrapper

    return decorator