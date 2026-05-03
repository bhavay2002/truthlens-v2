"""Centralized loader for the JSON lexicon files in ``src/config/lexicons/``.

Audit fix §1.1 + §9.4 — every bias / narrative / propaganda / framing
extractor used to ship inline ``LEXICON = {...}`` placeholders that
silently emitted permanently-zero feature columns in production. The
loader replaces those placeholders with a single source of truth so the
lexicons can be expanded, reviewed, and version-controlled without
touching extractor code.

Public API:

    load_lexicon(name)          -> dict[str, list[str]]
    load_lexicon_set(name, key) -> set[str]
    load_lexicon_dict(name, key, default_weight=1.0) -> dict[str, float]

The loader is lazy (one disk read per file per process) and thread-safe
under CPython's GIL because dict assignment is atomic.

A single file can contain multiple named lexicons keyed by category, so
the disk shape mirrors how each extractor groups its lexicons:

    src/config/lexicons/bias.json
        {
          "loaded":      ["regime", "thug", ...],
          "subjective":  ["amazing", "awful", ...],
          ...
        }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)

LEXICON_DIR = Path(__file__).resolve().parents[2] / "config" / "lexicons"

_FILE_CACHE: Dict[str, Dict[str, Any]] = {}

# Names whose JSON file was missing on disk — used to suppress the
# follow-up flood of per-key "Lexicon key missing" warnings, which are
# always implied by (and never add information beyond) the single
# "Lexicon file missing" warning we already logged.
_MISSING_FILES: Set[str] = set()


def _load_file(name: str) -> Dict[str, Any]:
    """Load and cache the JSON file for ``name`` (no extension)."""
    cached = _FILE_CACHE.get(name)
    if cached is not None:
        return cached

    path = LEXICON_DIR / f"{name}.json"
    if not path.exists():
        logger.warning("Lexicon file missing: %s", path)
        _MISSING_FILES.add(name)
        _FILE_CACHE[name] = {}
        return _FILE_CACHE[name]

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Lexicon file unreadable %s: %s", path, exc)
        _FILE_CACHE[name] = {}
        return _FILE_CACHE[name]

    if not isinstance(data, dict):
        logger.warning("Lexicon file %s top-level must be an object", path)
        _FILE_CACHE[name] = {}
        return _FILE_CACHE[name]

    _FILE_CACHE[name] = data
    return data


def load_lexicon(name: str) -> Dict[str, List[str]]:
    """Return the full {category: [terms]} mapping for ``name``.

    Keys starting with ``_`` (e.g. ``_doc``) are stripped — they are
    metadata, not lexicon categories.
    """
    data = _load_file(name)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def load_lexicon_set(name: str, key: str) -> Set[str]:
    """Return ``name.key`` as a lowercased ``set[str]``.

    Returns an empty set (with a warning) if the file or key is missing,
    so a misspelled lookup never raises in production but still surfaces
    in the application logs.
    """
    data = _load_file(name)
    raw = data.get(key)
    if raw is None:
        if name not in _MISSING_FILES:
            logger.warning("Lexicon key missing: %s.%s", name, key)
        return set()
    if isinstance(raw, dict):
        items = raw.keys()
    else:
        items = raw
    return {str(w).lower() for w in items if isinstance(w, str) and w}


def load_lexicon_dict(
    name: str,
    key: str,
    default_weight: float = 1.0,
) -> Dict[str, float]:
    """Return ``name.key`` as a lowercased ``dict[str, float]``.

    Accepts both list-shaped lexicons (each term gets ``default_weight``)
    and dict-shaped lexicons (``{term: weight}``). The output is sorted by
    insertion order; ``WeightedLexiconMatcher`` is order-insensitive.
    """
    data = _load_file(name)
    raw = data.get(key)
    if raw is None:
        if name not in _MISSING_FILES:
            logger.warning("Lexicon key missing: %s.%s", name, key)
        return {}
    if isinstance(raw, dict):
        return {
            str(w).lower(): float(v)
            for w, v in raw.items()
            if isinstance(w, str) and w
        }
    return {
        str(w).lower(): default_weight
        for w in raw
        if isinstance(w, str) and w
    }


def reset_cache() -> None:
    """Drop the in-process file cache (test hook)."""
    _FILE_CACHE.clear()
