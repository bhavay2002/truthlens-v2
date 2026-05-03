"""
File Name: spacy_config.py
Module: Analysis - spaCy Loader Configuration

Description:
    Centralized configuration for the shared spaCy loader (:mod:`spacy_loader`).
    Defines the default model, GPU preference, batch / process counts, and the
    per-task pipeline-disable map. Lives in its own module to keep the spaCy
    settings decoupled from the broader :class:`AnalysisConfig` and to avoid
    import cycles between ``analysis_config`` and ``spacy_loader``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


# =========================================================
# DEFAULTS
# =========================================================

DEFAULT_SPACY_MODEL = "en_core_web_sm"
DEFAULT_BATCH_SIZE = 32
DEFAULT_N_PROCESS = 1


# =========================================================
# TASK → PIPELINE DISABLE MAP
# =========================================================
# Maps a logical "task" name (used by analyzers) to the spaCy pipeline
# components that should be DISABLED for that task. Keeping pipelines lean
# avoids unnecessary work when an analyzer only needs lemmas or entities.
#
# Tasks used in the codebase:
#   - "syntax": default; analyzers need POS / DEP / lemma. Disable NER
#     because no syntax-task analyzer reads `ent_type_`/`ent_iob_`.
#   - "ner": entity-driven analyzers; need ents + lemma. The dependency
#     parser is the heaviest component and is unused, so disable it.
#     Keep the tagger because spaCy's NER backs off to POS features.
#   - "fast": tokenizer-only fallback used by the shared "fast" mode.
#
# Section 6: previously every entry was an empty tuple, which made
# `get_task_nlp(task)` load the full pipeline regardless of task and
# negated the whole point of the map (15-30% wasted CPU per analyzer
# that only needs lemmas or only needs entities).

DEFAULT_TASK_DISABLE_MAP: Dict[str, Tuple[str, ...]] = {
    "syntax": ("ner",),
    "ner": ("parser",),
    "fast": ("ner", "tagger", "parser", "attribute_ruler", "lemmatizer"),
}


# =========================================================
# CONFIG
# =========================================================

@dataclass(slots=True)
class SpacyConfig:
    """spaCy loader configuration."""

    model: str = DEFAULT_SPACY_MODEL
    use_gpu: bool = False
    batch_size: int = DEFAULT_BATCH_SIZE
    n_process: int = DEFAULT_N_PROCESS

    task_disable_map: Dict[str, Tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_TASK_DISABLE_MAP)
    )


__all__ = [
    "SpacyConfig",
    "DEFAULT_SPACY_MODEL",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_N_PROCESS",
    "DEFAULT_TASK_DISABLE_MAP",
]
