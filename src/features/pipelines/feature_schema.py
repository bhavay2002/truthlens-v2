"""
Compatibility shim.

This module previously declared a *second* feature schema whose names
silently drifted from the canonical schema in
`src/features/feature_schema.py` (e.g. `bias_loaded_language_ratio`
vs `bias_loaded`).  That drift caused FeatureSchemaValidator to either
zero-fill or drop entire feature groups depending on which schema was
imported first.

To keep one source of truth, this module now re-exports from the
canonical schema using the historical name aliases.  All new code must
import from `src.features.feature_schema`.
"""

from __future__ import annotations

from src.features.feature_schema import (
    BIAS_FEATURES        as BIAS_FEATURE_NAMES,
    FRAMING_FEATURES     as FRAMING_FEATURE_NAMES,
    IDEOLOGICAL_FEATURES as IDEOLOGICAL_FEATURE_NAMES,
    ARGUMENT_FEATURES    as ARGUMENT_STRUCTURE_FEATURE_NAMES,
    DISCOURSE_FEATURES   as DISCOURSE_FEATURE_NAMES,
    GRAPH_FEATURES,
    NARRATIVE_FEATURES   as NARRATIVE_FEATURE_NAMES,
    CONFLICT_FEATURES    as CONFLICT_FEATURE_NAMES,
    PROPAGANDA_FEATURES  as PROPAGANDA_FEATURE_NAMES,
    LEXICAL_FEATURES     as LEXICAL_FEATURE_NAMES,
    SEMANTIC_FEATURES    as SEMANTIC_FEATURE_NAMES,
    SYNTACTIC_FEATURES   as SYNTACTIC_FEATURE_NAMES,
    TOKEN_FEATURES       as TOKEN_FEATURE_NAMES,
    ALL_FEATURES,
)

# Historical sub-splits (kept for ABI compatibility).
ENTITY_GRAPH_FEATURE_NAMES = [n for n in GRAPH_FEATURES if n.startswith("entity_")]
INTERACTION_GRAPH_FEATURE_NAMES = [n for n in GRAPH_FEATURES if n.startswith("interaction_")]

__all__ = [
    "BIAS_FEATURE_NAMES",
    "FRAMING_FEATURE_NAMES",
    "IDEOLOGICAL_FEATURE_NAMES",
    "ARGUMENT_STRUCTURE_FEATURE_NAMES",
    "DISCOURSE_FEATURE_NAMES",
    "ENTITY_GRAPH_FEATURE_NAMES",
    "INTERACTION_GRAPH_FEATURE_NAMES",
    "CONFLICT_FEATURE_NAMES",
    "NARRATIVE_FEATURE_NAMES",
    "PROPAGANDA_FEATURE_NAMES",
    "LEXICAL_FEATURE_NAMES",
    "SEMANTIC_FEATURE_NAMES",
    "SYNTACTIC_FEATURE_NAMES",
    "TOKEN_FEATURE_NAMES",
    "ALL_FEATURES",
]
