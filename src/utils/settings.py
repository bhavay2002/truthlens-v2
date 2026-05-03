"""
File: settings.py
Location: src/utils/

Runtime settings loader.

Exposes a single :func:`load_settings` entry point that returns an
attribute-accessible view over ``config/config.yaml``. Several callers
(``api/app.py``, ``src.inference.predict_api``, training scripts) depend on
nested attribute access — e.g. ``SETTINGS.model.path`` or
``SETTINGS.paths.tfidf_vectorizer_path`` — so this loader builds a recursive
namespace from the YAML and applies a small set of project-specific
conveniences (path resolution, attribute defaults).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable

import yaml

logger = logging.getLogger(__name__)

# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


# =========================================================
# ATTRIBUTE NAMESPACE
# =========================================================

class AttrDict(SimpleNamespace):
    """SimpleNamespace that also supports ``dict``-style access and ``.get``."""

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def __contains__(self, key: object) -> bool:  # type: ignore[override]
        return isinstance(key, str) and hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, AttrDict):
                out[k] = v.to_dict()
            elif isinstance(v, list):
                out[k] = [x.to_dict() if isinstance(x, AttrDict) else x for x in v]
            else:
                out[k] = v
        return out


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return AttrDict(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


# =========================================================
# YAML LOADER
# =========================================================

def _resolve_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# =========================================================
# DEFAULTS
# =========================================================

_DEFAULT_API: Dict[str, Any] = {
    "title": "TruthLens AI API",
    "description": "Multi-task NLP system",
    "version": "2.0.0",
    "text_preview_chars": 100,
}

_DEFAULT_INFERENCE: Dict[str, Any] = {
    "batch_size": 32,
    "device": "auto",
    "use_half_precision": False,
    "allow_raw_text_fallback": True,
}

_DEFAULT_PATHS: Dict[str, Any] = {
    "tfidf_vectorizer_path": "saved_models/tfidf_vectorizer.joblib",
    "evaluation_results_path": "reports/evaluation_results.json",
    "confusion_matrix_path": "reports/confusion_matrix.png",
    "cleaning_report_path": "reports/cleaning_report.json",
    "models_dir": "saved_models",
    "logs_dir": "logs",
    "reports_dir": "reports",
}


def _ensure_defaults(target: AttrDict, defaults: Dict[str, Any]) -> None:
    for key, val in defaults.items():
        if not hasattr(target, key):
            setattr(target, key, _to_namespace(val))


def _resolve_paths(ns: AttrDict, keys: Iterable[str]) -> None:
    """Resolve any keys that look like paths into absolute :class:`Path` objects."""
    for key in keys:
        if hasattr(ns, key):
            value = getattr(ns, key)
            if isinstance(value, (str, Path)):
                setattr(ns, key, _resolve_path(value))


# =========================================================
# PUBLIC API
# =========================================================

@lru_cache(maxsize=1)
def load_settings(config_path: str | Path | None = None) -> AttrDict:
    """Load the runtime settings as an attribute-accessible namespace."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    raw = _load_yaml(path)

    settings = _to_namespace(raw)
    if not isinstance(settings, AttrDict):  # pragma: no cover - defensive
        raise TypeError("config root must be a mapping")

    # ---- model ----
    model = getattr(settings, "model", AttrDict())
    if not hasattr(model, "path"):
        # Sensible default so callers like ``api/app.py`` always have a value.
        model.path = "saved_models"
    model.path = _resolve_path(model.path)
    encoder = getattr(model, "encoder", AttrDict())
    if isinstance(encoder, str):
        encoder = AttrDict(name=encoder)
    if not hasattr(model, "name"):
        model.name = getattr(encoder, "name", "roberta-base")
    if not hasattr(model, "max_length"):
        model.max_length = int(getattr(encoder, "max_length", 512))
    settings.model = model

    # ---- paths ----
    paths = getattr(settings, "paths", AttrDict())
    _ensure_defaults(paths, _DEFAULT_PATHS)
    _resolve_paths(
        paths,
        (
            "tfidf_vectorizer_path",
            "evaluation_results_path",
            "confusion_matrix_path",
            "cleaning_report_path",
            "models_dir",
            "logs_dir",
            "reports_dir",
        ),
    )
    settings.paths = paths

    # ---- training ----
    training = getattr(settings, "training", AttrDict())
    if not hasattr(training, "text_column"):
        training.text_column = "text"
    settings.training = training

    # ---- api / inference defaults ----
    api_ns = getattr(settings, "api", AttrDict())
    _ensure_defaults(api_ns, _DEFAULT_API)
    settings.api = api_ns

    inference_ns = getattr(settings, "inference", AttrDict())
    _ensure_defaults(inference_ns, _DEFAULT_INFERENCE)
    settings.inference = inference_ns

    logger.info("Loaded settings from %s", path)
    return settings
