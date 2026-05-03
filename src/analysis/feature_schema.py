from __future__ import annotations

import logging
from typing import Dict, Tuple, List, Iterable, Any

import numpy as np

# -----------------------------------------------------
# Re-export feature key tuples from the canonical source.
# Existing call sites import names like RHETORICAL_DEVICE_KEYS
# from this module, so we keep the import surface stable.
# -----------------------------------------------------
from src.analysis.feature_keys import (
    RHETORICAL_DEVICE_KEYS,
    ARGUMENT_MINING_KEYS,
    CONTEXT_OMISSION_KEYS,
    DISCOURSE_COHERENCE_KEYS,
    EMOTION_TARGET_KEYS,
    FRAMING_KEYS,
    INFORMATION_DENSITY_KEYS,
    INFORMATION_OMISSION_KEYS,
    IDEOLOGICAL_LANGUAGE_KEYS,
    NARRATIVE_CONFLICT_KEYS,
    NARRATIVE_PROPAGATION_KEYS,
    NARRATIVE_ROLE_KEYS,
    NARRATIVE_TEMPORAL_KEYS,
    SOURCE_ATTRIBUTION_KEYS,
    PROPAGANDA_PATTERN_KEYS,
    ALL_FEATURE_KEYS,
)

logger = logging.getLogger(__name__)


# =========================================================
# REGISTRY (LOCKABLE + SAFE)
# =========================================================

SCHEMA_REGISTRY: Dict[str, Tuple[str, ...]] = {}
_SCHEMA_LOCKED: bool = False


# =========================================================
# REGISTRATION
# =========================================================

def register_schema(name: str, keys: List[str]) -> None:
    global _SCHEMA_LOCKED

    if _SCHEMA_LOCKED:
        raise RuntimeError("Schema registry is locked")

    if not name or not isinstance(name, str):
        raise ValueError("Invalid schema name")

    if not keys:
        raise ValueError(f"Schema '{name}' cannot be empty")

    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate keys in schema '{name}'")

    # enforce deterministic ordering
    ordered = tuple(keys)

    if name in SCHEMA_REGISTRY:
        logger.warning("Overwriting schema: %s", name)

    SCHEMA_REGISTRY[name] = ordered


def register_many(schemas: Dict[str, List[str]]) -> None:
    for name, keys in schemas.items():
        register_schema(name, keys)


def lock_schema_registry() -> None:
    global _SCHEMA_LOCKED
    _SCHEMA_LOCKED = True
    logger.info("Schema registry locked")


# =========================================================
# ACCESS
# =========================================================

def get_schema(name: str) -> Tuple[str, ...]:
    if name not in SCHEMA_REGISTRY:
        raise ValueError(f"Schema not found: {name}")
    return SCHEMA_REGISTRY[name]


def list_schemas() -> List[str]:
    return list(SCHEMA_REGISTRY.keys())


# =========================================================
# VECTOR CREATION (OPTIMIZED)
# =========================================================

def make_vector(
    features: Dict[str, float],
    keys: Tuple[str, ...],
    *,
    strict: bool = False,
    safe: bool = True,
    clip: Tuple[float, float] | None = None,
    return_metadata: bool = False,
) -> Any:

    if features is None:
        raise ValueError("features cannot be None")

    # Strict mode: check for missing AND unknown keys up-front
    if strict:
        key_set = set(keys)
        missing = [k for k in keys if k not in features]
        unknown = [k for k in features if k not in key_set]
        if missing:
            raise ValueError(f"Missing required feature keys: {missing}")
        if unknown:
            raise ValueError(f"Unknown feature keys: {unknown}")

    # -----------------------------------------------------
    # FAST PATH (NUMPY ARRAY BUILD)
    # -----------------------------------------------------

    values = np.empty(len(keys), dtype=np.float32)
    missing_keys: List[str] = []

    for i, k in enumerate(keys):

        if k not in features:
            missing_keys.append(k)
            v = 0.0
        else:
            v = features[k]

        # -------------------------
        # SAFE MODE
        # -------------------------
        if safe:
            if not isinstance(v, (int, float)):
                logger.debug("Non-numeric value for %s → 0.0", k)
                v = 0.0
            elif not np.isfinite(v):
                logger.debug("Invalid value for %s → 0.0", k)
                v = 0.0

        v = float(v)

        # -------------------------
        # CLIPPING
        # -------------------------
        if clip is not None:
            v = float(np.clip(v, clip[0], clip[1]))

        values[i] = v

    if return_metadata:
        return {
            "vector": values,
            "missing_keys": missing_keys,
            "dim": int(values.shape[0]),
        }

    return values


# =========================================================
# SCHEMA-BASED VECTOR
# =========================================================

def make_vector_from_schema(
    features: Dict[str, float],
    schema_name: str,
    *,
    strict: bool = False,
    **kwargs,
):
    keys = get_schema(schema_name)
    return make_vector(features, keys, strict=strict, **kwargs)


# =========================================================
# MULTI-SCHEMA MERGE (NEW 🔥)
# =========================================================

def build_combined_schema(
    names: Iterable[str],
    *,
    deduplicate: bool = True,
) -> Tuple[str, ...]:

    combined: List[str] = []

    for name in names:
        keys = get_schema(name)
        combined.extend(keys)

    if deduplicate:
        # preserve order while removing duplicates
        seen = set()
        ordered = []
        for k in combined:
            if k not in seen:
                seen.add(k)
                ordered.append(k)
        return tuple(ordered)

    return tuple(combined)


# =========================================================
# VALIDATION
# =========================================================

def validate_features(
    features: Dict[str, float],
    schema_keys: Tuple[str, ...],
    *,
    strict: bool = False,
) -> bool:

    if not isinstance(features, dict):
        raise TypeError("features must be dict")

    ok = True

    for k in schema_keys:

        if k not in features:
            if strict:
                logger.error("Missing key: %s", k)
                return False
            ok = False
            continue

        v = features[k]

        if not isinstance(v, (int, float)):
            logger.warning("Non-numeric value for %s", k)
            ok = False

        if isinstance(v, float) and not np.isfinite(v):
            logger.warning("Invalid value for %s", k)
            ok = False

    return ok


# =========================================================
# SCHEMA INTEGRITY CHECK (NEW 🔥)
# =========================================================

def validate_schema_integrity() -> None:
    """
    Ensures no key collisions across schemas.
    Critical for multi-task models.
    """

    all_keys = {}
    duplicates = {}

    for name, keys in SCHEMA_REGISTRY.items():
        for k in keys:
            if k in all_keys:
                duplicates.setdefault(k, []).append(name)
            else:
                all_keys[k] = name

    if duplicates:
        logger.warning("Schema key collisions detected: %s", duplicates)
    else:
        logger.info("Schema integrity check passed")


# =========================================================
# METADATA
# =========================================================

SCHEMA_VERSION = "4.0.0"


def get_schema_metadata() -> Dict[str, str]:
    return {
        "version": SCHEMA_VERSION,
        "num_schemas": str(len(SCHEMA_REGISTRY)),
        "locked": str(_SCHEMA_LOCKED),
    }


# =========================================================
# DEFAULT REGISTRATIONS (run once at import)
# =========================================================
# Register all known analyzer schemas so callers like
# `bias_profile_vector` can look them up by name. Use
# `register_many` to keep the registry stable and ordered.

_DEFAULT_SCHEMAS = {
    "rhetorical_device": list(RHETORICAL_DEVICE_KEYS),
    "argument_mining": list(ARGUMENT_MINING_KEYS),
    "context_omission": list(CONTEXT_OMISSION_KEYS),
    "discourse_coherence": list(DISCOURSE_COHERENCE_KEYS),
    "emotion_target": list(EMOTION_TARGET_KEYS),
    "framing": list(FRAMING_KEYS),
    "information_density": list(INFORMATION_DENSITY_KEYS),
    "information_omission": list(INFORMATION_OMISSION_KEYS),
    "ideology": list(IDEOLOGICAL_LANGUAGE_KEYS),
    "narrative_conflict": list(NARRATIVE_CONFLICT_KEYS),
    "narrative_propagation": list(NARRATIVE_PROPAGATION_KEYS),
    "narrative_role": list(NARRATIVE_ROLE_KEYS),
    "narrative_temporal": list(NARRATIVE_TEMPORAL_KEYS),
    "source_attribution": list(SOURCE_ATTRIBUTION_KEYS),
    "propaganda_pattern": list(PROPAGANDA_PATTERN_KEYS),
}

if not SCHEMA_REGISTRY:
    register_many(_DEFAULT_SCHEMAS)