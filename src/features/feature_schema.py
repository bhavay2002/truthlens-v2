from __future__ import annotations

"""
Central Feature Schema for TruthLens

SINGLE SOURCE OF TRUTH for hand-engineered feature names emitted by
`src/features/*/...`.

Important conventions
---------------------
* These names are FEATURES (model inputs).  They are intentionally
  distinct from LABEL columns defined in
  `src/data_processing/data_contracts.py` (e.g. `hero/villain/victim`).
  Feature names use a domain prefix (`narrative_role_*`) so that
  partition_feature_sections() in pipelines/feature_pipeline.py can
  route them to the correct task head.
* Each list below MUST stay in lock-step with the keys returned by the
  corresponding extractor's `extract()`.  `assert_schema_consistency()`
  performs a runtime check on a sentinel input at startup.
"""

from typing import Dict, List


# =========================================================
# BIAS  (BiasFeaturesV2)
# =========================================================

BIAS_FEATURES = [
    "bias_loaded",
    "bias_subjective",
    "bias_uncertainty",
    "bias_polarization",
    "bias_evaluative",
    "bias_intensity",
    "bias_diversity",            # was: bias_variance (drift fixed)
    "bias_caps_ratio",
    "bias_exclamation_density",
]


# =========================================================
# FRAMING  (FramingFeatures)
# =========================================================

FRAMING_FEATURES = [
    "frame_economic_ratio",
    "frame_moral_ratio",
    "frame_security_ratio",
    "frame_human_interest_ratio",
    "frame_conflict_ratio",
    "frame_phrase_count",
    "frame_quote_density",
    "frame_diversity",
    "frame_dominance",
    "frame_entropy",
]


# =========================================================
# IDEOLOGY  (IdeologicalFeatures)
# =========================================================

IDEOLOGICAL_FEATURES = [
    "ideology_left_ratio",
    "ideology_right_ratio",
    "ideology_balance",
    "ideology_entropy",
    "ideology_polarization_ratio",
    "ideology_group_reference_ratio",
    "ideology_phrase_count",
    "ideology_signal_strength",
]


# =========================================================
# DISCOURSE
# =========================================================

DISCOURSE_FEATURES = [
    "discourse_causal_ratio",
    "discourse_contrast_ratio",
    "discourse_additive_ratio",
    "discourse_sequential_ratio",
    "discourse_evidential_ratio",
    "discourse_marker_density",
    "discourse_diversity",
]


# =========================================================
# ARGUMENT STRUCTURE
# =========================================================

ARGUMENT_FEATURES = [
    "argument_claim_ratio",
    "argument_premise_ratio",
    "argument_evidence_ratio",
    "argument_counterargument_ratio",
    "argument_rhetorical_question_ratio",
    "argument_structure_density",
    "argument_structure_diversity",
]


# =========================================================
# EMOTION
# =========================================================

# EMOTION-11: import from the canonical schema (src/features/emotion/
# emotion_schema.py) so this module can never silently disagree with
# the live label list. The previous duplicated 20-name inline list was
# the exact bug the user hit — feature_schema kept emitting 20 columns
# while the rest of the pipeline had moved to 11.
from src.features.emotion.emotion_schema import (
    EMOTION_LABELS,
    NUM_EMOTION_LABELS,  # noqa: F401  (re-exported for legacy callers)
)

EMOTION_FEATURES = (
    [f"emotion_{e}" for e in EMOTION_LABELS]
    + ["emotion_intensity"]
)
# NOTE: per-label one-hot ``emotion_dominant_<label>`` columns were
# removed (audit task 3, multi-label fix). The per-label scalars above
# (`emotion_<label>` = normalized hit share) carry the same signal in a
# distributional form; argmax recovers the dominant label at inference
# time without throwing away the rest of the distribution.


# =========================================================
# NARRATIVE  (NarrativeRoleFeatures)
#
# NOTE: feature names use the `narrative_role_*` prefix and `_ratio`
# suffix so they can be routed by partition_feature_sections().
# They are DISTINCT from the label columns `hero/villain/victim`
# declared in data_contracts.CONTRACTS["narrative"].
# =========================================================

NARRATIVE_FEATURES = [
    "narrative_role_hero_ratio",
    "narrative_role_villain_ratio",
    "narrative_role_victim_ratio",
    "narrative_role_polarization_ratio",
    "narrative_role_intensity",
    "narrative_role_entropy",
    "narrative_role_balance",
    "narrative_role_diversity",
    "narrative_entity_density",
]


# =========================================================
# PROPAGANDA
# =========================================================

PROPAGANDA_FEATURES = [
    "propaganda_name_calling_ratio",
    "propaganda_fear_ratio",
    "propaganda_exaggeration_ratio",
    "propaganda_glitter_ratio",
    "propaganda_us_vs_them_ratio",
    "propaganda_authority_ratio",
    "propaganda_intensifier_ratio",
    "propaganda_exclamation_density",
    "propaganda_caps_ratio",
    "propaganda_intensity",
    "propaganda_diversity",
]


# =========================================================
# CONFLICT
# =========================================================

CONFLICT_FEATURES = [
    "conflict_confrontation_ratio",
    "conflict_dispute_ratio",
    "conflict_accusation_ratio",
    "conflict_aggression_ratio",
    "conflict_polarization_ratio",
    "conflict_escalation_ratio",
    "conflict_intensity",
    "conflict_diversity",
    "conflict_rhetoric_score",
]


# =========================================================
# GRAPH
# =========================================================

GRAPH_FEATURES = [
    "entity_count",
    "entity_edge_count",
    "entity_avg_degree",
    "entity_density",
    "entity_centralization",

    "interaction_node_count",
    "interaction_edge_count",
    "interaction_avg_degree",
    "interaction_density",
    "interaction_clustering",
    "interaction_component_count",
]

GRAPH_PIPELINE_FEATURES = [
    "graph_pipeline_entity_density",
    "graph_pipeline_entity_centralization",
    "graph_pipeline_narrative_flow",
    "graph_pipeline_narrative_coherence",
]


# =========================================================
# TEXT
# =========================================================

LEXICAL_FEATURES = [
    "vocabulary_size",
    "hapax_legomena_ratio",
    "hapax_dislegomena_ratio",
    "lexical_density",
    "average_word_length",
]

SEMANTIC_FEATURES = [
    "embedding_norm",
    "embedding_mean",
    "embedding_std",
    "embedding_max",
    "embedding_min",
]

SYNTACTIC_FEATURES = [
    "sentence_count",
    "avg_sentence_length",
    "noun_ratio",
    "verb_ratio",
    "adjective_ratio",
    "adverb_ratio",
    "punctuation_ratio",
]

TOKEN_FEATURES = [
    "token_count",
    "unique_token_count",
    "type_token_ratio",
    "avg_token_length",
    "max_token_length",
    "repetition_ratio",
]


# =========================================================
# ALL FEATURES (MASTER LIST)
# =========================================================

ALL_FEATURES: List[str] = sorted(
    BIAS_FEATURES
    + FRAMING_FEATURES
    + IDEOLOGICAL_FEATURES
    + DISCOURSE_FEATURES
    + ARGUMENT_FEATURES
    + EMOTION_FEATURES
    + NARRATIVE_FEATURES
    + PROPAGANDA_FEATURES
    + CONFLICT_FEATURES
    + GRAPH_FEATURES
    + GRAPH_PIPELINE_FEATURES
    + LEXICAL_FEATURES
    + SEMANTIC_FEATURES
    + SYNTACTIC_FEATURES
    + TOKEN_FEATURES
)


# =========================================================
# SECTION MAP (mirrors partition_feature_sections() prefixes)
# =========================================================

FEATURE_SECTIONS: Dict[str, List[str]] = {
    "bias":       BIAS_FEATURES,
    "framing":    FRAMING_FEATURES,
    "ideology":   IDEOLOGICAL_FEATURES,
    "discourse":  DISCOURSE_FEATURES + ARGUMENT_FEATURES,
    "emotion":    EMOTION_FEATURES,
    "narrative":  NARRATIVE_FEATURES + CONFLICT_FEATURES,
    "propaganda": PROPAGANDA_FEATURES,
    "graph":      GRAPH_FEATURES + GRAPH_PIPELINE_FEATURES,
    "text":       LEXICAL_FEATURES + SEMANTIC_FEATURES
                  + SYNTACTIC_FEATURES + TOKEN_FEATURES,
}


# =========================================================
# HELPERS
# =========================================================

def get_all_features() -> List[str]:
    """Return full feature schema."""
    return ALL_FEATURES


def get_feature_sections() -> Dict[str, List[str]]:
    """Return section-wise feature mapping."""
    return FEATURE_SECTIONS


def validate_feature_names(features: List[str]) -> None:
    """Sanity check for schema integrity."""
    duplicates = {f for f in features if features.count(f) > 1}
    if duplicates:
        raise ValueError(f"Duplicate features in schema: {duplicates}")
