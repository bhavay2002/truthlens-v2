from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Dict, Optional, Tuple, Iterable, Iterator, Any

import spacy
from spacy.language import Language
from spacy.util import is_package

from src.analysis.analysis_config import ANALYSIS_CONFIG

logger = logging.getLogger(__name__)


# =========================================================
# CACHE (THREAD-SAFE, MULTI-MODEL)
# =========================================================

_CacheKey = Tuple[str, Tuple[str, ...]]
_CACHE: Dict[_CacheKey, Language] = {}
_LOCK = threading.RLock()


# =========================================================
# CONFIG (CENTRALIZED)
# =========================================================

DEFAULT_MODEL = ANALYSIS_CONFIG.spacy.model
ENABLE_GPU = ANALYSIS_CONFIG.spacy.use_gpu
DEFAULT_BATCH_SIZE = ANALYSIS_CONFIG.spacy.batch_size
DEFAULT_N_PROCESS = ANALYSIS_CONFIG.spacy.n_process

TASK_DISABLE_MAP = ANALYSIS_CONFIG.spacy.task_disable_map


# =========================================================
# INTERNAL HELPERS
# =========================================================

@lru_cache(maxsize=None)
def _resolve_model(model: str) -> str:
    """
    Resolve model with safe fallback.

    VOCAB-1 follow-up: memoised so the warning is emitted exactly once
    per missing model name and ``is_package`` (which hits the package
    metadata DB) is not paid on every ``get_nlp`` call. ``get_nlp`` now
    invokes this on every entry, including cache hits, in order to
    decide whether to normalise the cache key for the blank fallback.
    """
    if is_package(model):
        return model

    logger.warning("[spaCy] Model not found: %s → using blank 'en'", model)
    return "en"


# Section 6: GPU initialization must happen exactly once, BEFORE any
# spaCy model is loaded. The previous version called `_maybe_enable_gpu`
# from inside `get_nlp` after the cache check, which meant:
#   1. The first cache hit raced GPU init on cold start.
#   2. Every subsequent cache miss paid the cost of `prefer_gpu()` again.
# Now we run it exactly once at module import via a sentinel.
_GPU_INIT_DONE = False
_GPU_INIT_LOCK = threading.Lock()


def _maybe_enable_gpu() -> None:
    global _GPU_INIT_DONE
    if _GPU_INIT_DONE:
        return

    with _GPU_INIT_LOCK:
        if _GPU_INIT_DONE:
            return

        _GPU_INIT_DONE = True

        if not ENABLE_GPU:
            return

        try:
            if spacy.prefer_gpu():
                logger.info("[spaCy] GPU enabled")
            else:
                logger.warning("[spaCy] GPU requested but not available")
        except Exception as e:
            logger.warning("[spaCy] GPU init failed: %s", e)


def _configure_torch_threads_for_multiprocess() -> None:
    """Section 6: avoid CPU oversubscription under spaCy multiprocessing.

    When ``nlp.pipe(..., n_process=N>1)`` is used, spaCy forks N worker
    processes. If torch is installed and left at its default thread
    count (= number of CPU cores), each worker spawns a thread pool of
    that size, producing N x cores threads competing for the same CPUs.
    Pin torch to a single intra-op thread per worker so the total
    parallelism stays bounded by ``n_process`` (matching numpy's
    behavior under our default BLAS configuration).

    Best-effort: torch may not be installed, and `set_num_threads` may
    fail if torch is already in use. Either case is non-fatal.
    """
    try:
        import torch  # type: ignore
        torch.set_num_threads(1)
    except Exception:
        pass


# Run GPU bootstrap exactly once at import time so the very first
# `get_nlp` call sees a fully initialized GPU state.
_maybe_enable_gpu()


def _validate_pipeline(nlp: Language, disable: Tuple[str, ...]) -> None:
    active = set(nlp.pipe_names)

    for pipe in disable:
        if pipe in active:
            logger.warning(
                "[spaCy] Pipe '%s' expected disabled but still active",
                pipe,
            )


# =========================================================
# CORE LOADER
# =========================================================

def get_nlp(
    model: str = DEFAULT_MODEL,
    disable: Optional[Tuple[str, ...]] = None,
) -> Language:

    disable_tuple = tuple(disable or ())

    # VOCAB-1: when the requested model is missing and we fall back to a
    # blank ``en`` pipeline, the ``disable`` tuple is meaningless (the
    # blank pipeline has no pipes to disable). Keying the cache by the
    # original ``(model, disable)`` pair would mint a fresh
    # ``spacy.blank("en")`` for every distinct disable variant, and each
    # blank pipeline owns its own ``Vocab``. Downstream components such
    # as ``EmotionTargetAnalyzer`` build a ``PhraseMatcher`` from one
    # Vocab in ``__init__`` and then receive ``Doc`` objects backed by a
    # different Vocab at request time, producing the
    # ``doc.vocab does not match PhraseMatcher vocab`` warning and
    # silently degrading every entity-aware analyzer. We resolve the
    # model first and, when the fallback fires, normalise the cache key
    # so all callers share one blank pipeline (one Vocab).
    resolved_model = _resolve_model(model)

    if resolved_model == "en":
        # Blank pipeline has no pipes — disable is irrelevant. Share one
        # instance across the whole process.
        key: _CacheKey = ("en", ())
        effective_disable: Tuple[str, ...] = ()
    else:
        key = (resolved_model, disable_tuple)
        effective_disable = disable_tuple

    if key in _CACHE:
        return _CACHE[key]

    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]

        logger.info(
            "[spaCy] Loading | model=%s | disable=%s",
            resolved_model,
            effective_disable,
        )

        try:
            if resolved_model == "en":
                nlp = spacy.blank("en")
                # Blank model has no parser/sentencizer so doc.sents raises
                # E030. Add a cheap rule-based sentencizer so preprocessing
                # and any graph component that calls doc.sents always works,
                # regardless of initialization order.
                if not nlp.has_pipe("sentencizer"):
                    nlp.add_pipe("sentencizer")
            else:
                nlp = spacy.load(resolved_model, disable=list(effective_disable))
        except Exception as e:
            logger.exception("[spaCy] Load failed")
            raise RuntimeError(f"Failed to load spaCy model: {model}") from e

        nlp.max_length = 2_000_000

        _validate_pipeline(nlp, effective_disable)

        _CACHE[key] = nlp
        return nlp


# =========================================================
# TASK-AWARE LOADER
# =========================================================

def get_task_nlp(task: str) -> Language:

    if task not in TASK_DISABLE_MAP:
        raise ValueError(f"Unknown task: {task}")

    disable = TASK_DISABLE_MAP[task]
    return get_nlp(DEFAULT_MODEL, disable=disable)


# =========================================================
#  SHARED DOC CACHE (CRITICAL OPTIMIZATION)
# =========================================================

def get_doc(context: Any, task: str):
    """
    Retrieve spaCy doc using shared cache.

    Ensures:
    - single NLP pass per task per context
    - reused across features
    """

    if not hasattr(context, "shared") or context.shared is None:
        context.shared = {}

    cache = context.shared.setdefault("spacy_docs", {})

    if task in cache:
        return cache[task]

    nlp = get_task_nlp(task)
    doc = nlp(context.text)

    cache[task] = doc
    return doc


# =========================================================
# SHARED PIPELINE (PIPELINE-LEVEL ENTRY POINT)
# =========================================================
# `get_shared_nlp(mode)` returns the NLP pipeline that the
# top-level `AnalysisPipeline` uses to materialize the doc
# once. Modes:
#   "safe" — full pipeline (tagger, parser, NER, lemmatizer)
#   "fast" — minimal tokenizer-only pipeline
# Anything else falls back to "safe".

_SHARED_MODE_DISABLE: Dict[str, Tuple[str, ...]] = {
    "safe": (),
    "fast": ("ner", "tagger", "parser", "attribute_ruler", "lemmatizer"),
}


def get_shared_nlp(mode: str = "safe") -> Language:
    disable = _SHARED_MODE_DISABLE.get(mode, ())
    return get_nlp(DEFAULT_MODEL, disable=disable)


# =========================================================
# STREAM PROCESSING (HIGH PERFORMANCE)
# =========================================================

def process_docs_stream(
    texts: Iterable[str],
    *,
    task: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_process: int = DEFAULT_N_PROCESS,
) -> Iterator:

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    if n_process < 1:
        raise ValueError("n_process must be >= 1")

    nlp = get_task_nlp(task)

    # Section 6: pin torch to a single thread per worker process when
    # spaCy is about to fork. No-op for n_process=1 (single-process
    # path keeps torch's default threading).
    if n_process > 1:
        _configure_torch_threads_for_multiprocess()

    logger.debug(
        "[spaCy] Stream | task=%s | batch=%d | proc=%d",
        task,
        batch_size,
        n_process,
    )

    try:
        yield from nlp.pipe(
            texts,
            batch_size=batch_size,
            n_process=n_process,
        )
    except Exception as e:
        logger.exception("[spaCy] Stream processing failed")
        raise RuntimeError("spaCy pipeline execution failed") from e


# =========================================================
# MATERIALIZED PROCESSING
# =========================================================

def process_docs(
    texts: Iterable[str],
    *,
    task: str,
) -> list:

    return list(process_docs_stream(texts, task=task))


# =========================================================
# WARMUP (LOW LATENCY)
# =========================================================

def warmup_all_tasks() -> None:

    logger.info("[spaCy] Warmup start")

    for task in TASK_DISABLE_MAP:
        try:
            nlp = get_task_nlp(task)
            _ = nlp("Warmup text.")
        except Exception:
            logger.exception("[spaCy] Warmup failed for task=%s", task)

    logger.info("[spaCy] Warmup complete")


# =========================================================
# CACHE CONTROL
# =========================================================

def clear_cache() -> None:

    with _LOCK:
        _CACHE.clear()
        logger.info("[spaCy] Cache cleared")


# =========================================================
# INTROSPECTION
# =========================================================

def get_loaded_models() -> Dict[str, Dict]:

    info = {}

    for (model, disable), nlp in _CACHE.items():
        key = f"{model}|disable={disable}"
        info[key] = {
            "pipes": list(nlp.pipe_names),
            "max_length": nlp.max_length,
        }

    return info