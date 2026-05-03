"""
File Name: feature_keys.py
Module: Analysis - Feature Key Registry

Description:
    Single source of truth for the ordered tuple of feature keys produced by
    every analyzer in :mod:`src.analysis`. The keys here MUST exactly match the
    keys returned by each analyzer's ``analyze()`` and ``_empty()``/_empty_features()
    methods. Vector builders, schema validators, and the propaganda detector all
    rely on these tuples to produce stable, deterministic feature vectors.

    These tuples are re-exported from :mod:`src.analysis.feature_schema` for
    backward compatibility with existing imports.
"""

from __future__ import annotations

from typing import Tuple


# =========================================================
# RHETORICAL DEVICES
# =========================================================

RHETORICAL_DEVICE_KEYS: Tuple[str, ...] = (
    "rhetoric_exaggeration_score",
    "rhetoric_loaded_language_score",
    "rhetoric_emotional_appeal_score",
    "rhetoric_fear_appeal_score",
    "rhetoric_intensifier_ratio",
    "rhetoric_scapegoating_score",
    "rhetoric_false_dilemma_score",
    "rhetoric_punctuation_score",
    "rhetoric_intensity",
    "rhetoric_diversity",
)


# =========================================================
# ARGUMENT MINING
# =========================================================

ARGUMENT_MINING_KEYS: Tuple[str, ...] = (
    "argument_claim_ratio",
    "argument_premise_ratio",
    "argument_support_ratio",
    "argument_contrast_ratio",
    "argument_rebuttal_ratio",
    "argument_verb_density",
    "argument_clause_density",
    "argument_complexity",
)


# =========================================================
# CONTEXT OMISSION
# =========================================================

CONTEXT_OMISSION_KEYS: Tuple[str, ...] = (
    "context_vague_reference_ratio",
    "context_attribution_ratio",
    "context_evidence_ratio",
    "context_uncertainty_ratio",
    "context_quote_ratio",
    "context_entity_ratio",
    "context_entity_type_diversity",
    "context_grounding_score",
)


# =========================================================
# DISCOURSE COHERENCE
# =========================================================

DISCOURSE_COHERENCE_KEYS: Tuple[str, ...] = (
    "sentence_coherence",
    "topic_drift",
    "narrative_continuity",
    # F14: canonical alias of `narrative_continuity`. The metric measures
    # entity repetition within the doc; both keys are emitted with the
    # same value for backward compatibility.
    "entity_repetition_ratio",
    "discourse_transition_ratio",
)


# =========================================================
# EMOTION TARGET
# =========================================================

EMOTION_TARGET_KEYS: Tuple[str, ...] = (
    "emotion_target_diversity",
    "emotion_target_focus",
    "emotion_expression_ratio",
    "emotion_type_diversity",
    "dominant_emotion_strength",
)


# =========================================================
# FRAMING
# =========================================================

FRAMING_KEYS: Tuple[str, ...] = (
    "frame_conflict_score",
    "frame_economic_score",
    "frame_moral_score",
    "frame_human_interest_score",
    "frame_security_score",
    "frame_dominance_score",
    "frame_diversity_score",
)


# =========================================================
# INFORMATION DENSITY
# =========================================================

INFORMATION_DENSITY_KEYS: Tuple[str, ...] = (
    "factual_density",
    "opinion_density",
    "claim_density",
    "rhetorical_density",
    "emotion_density",
    "modal_density",
    "rhetorical_punctuation_density",
    "information_emotion_ratio",
    "information_diversity",
)


# =========================================================
# INFORMATION OMISSION
# =========================================================

INFORMATION_OMISSION_KEYS: Tuple[str, ...] = (
    "missing_counterargument_score",
    "one_sided_framing_score",
    "incomplete_evidence_score",
    "claim_evidence_imbalance",
)


# =========================================================
# IDEOLOGICAL LANGUAGE
# =========================================================

IDEOLOGICAL_LANGUAGE_KEYS: Tuple[str, ...] = (
    "liberty_language_ratio",
    "equality_language_ratio",
    "tradition_language_ratio",
    "anti_elite_language_ratio",
    "liberty_vs_equality_balance",
    "ideology_phrase_density",
    "ideology_diversity",
)


# =========================================================
# NARRATIVE CONFLICT
# =========================================================

NARRATIVE_CONFLICT_KEYS: Tuple[str, ...] = (
    "conflict_verb_ratio",
    "opposition_marker_ratio",
    "polarization_ratio",
    "hero_villain_victim_ratio",
    "conflict_intensity",
    "conflict_exclamation_ratio",
    "conflict_question_ratio",
)


# =========================================================
# NARRATIVE PROPAGATION
# =========================================================

NARRATIVE_PROPAGATION_KEYS: Tuple[str, ...] = (
    "violent_conflict_ratio",
    "political_conflict_ratio",
    "discursive_conflict_ratio",
    "institutional_conflict_ratio",
    "coercion_conflict_ratio",
    "opposition_marker_ratio",
    "polarization_ratio",
    "conflict_phrase_ratio",
    "hero_villain_conflict_score",
    "villain_victim_harm_score",
    "hero_victim_protection_score",
    "conflict_propagation_intensity",
    "conflict_diversity",
    "conflict_exclamation_ratio",
    "conflict_question_ratio",
)


# =========================================================
# NARRATIVE ROLE
# =========================================================

NARRATIVE_ROLE_KEYS: Tuple[str, ...] = (
    "hero_ratio",
    "villain_ratio",
    "victim_ratio",
    "hero_vs_villain_balance",
    "hero_entities",
    "villain_entities",
    "victim_entities",
)


# =========================================================
# NARRATIVE TEMPORAL
# =========================================================

NARRATIVE_TEMPORAL_KEYS: Tuple[str, ...] = (
    "past_framing_ratio",
    "crisis_escalation_ratio",
    "urgency_language_ratio",
    "past_tense_ratio",
    "present_tense_ratio",
    "future_tense_ratio",
    "temporal_contrast_score",
    "temporal_intensity",
    "temporal_diversity",
)


# =========================================================
# SOURCE ATTRIBUTION
# =========================================================

SOURCE_ATTRIBUTION_KEYS: Tuple[str, ...] = (
    "expert_attribution_ratio",
    "anonymous_source_ratio",
    "credibility_indicator_ratio",
    "attribution_verb_ratio",
    "quotation_ratio",
    "named_source_ratio",
    "source_credibility_balance",
    "attribution_intensity",
    "attribution_diversity",
)


# =========================================================
# PROPAGANDA PATTERN
# =========================================================

PROPAGANDA_PATTERN_KEYS: Tuple[str, ...] = (
    "fear_propaganda_score",
    "scapegoating_score",
    "polarization_score",
    "emotional_amplification_score",
    "narrative_imbalance_score",
    "propaganda_intensity",
    "propaganda_diversity",
)


# =========================================================
# REGISTRY (NAME → KEYS)
# =========================================================

ALL_FEATURE_KEYS = {
    "rhetorical_device": RHETORICAL_DEVICE_KEYS,
    "argument_mining": ARGUMENT_MINING_KEYS,
    "context_omission": CONTEXT_OMISSION_KEYS,
    "discourse_coherence": DISCOURSE_COHERENCE_KEYS,
    "emotion_target": EMOTION_TARGET_KEYS,
    "framing": FRAMING_KEYS,
    "information_density": INFORMATION_DENSITY_KEYS,
    "information_omission": INFORMATION_OMISSION_KEYS,
    "ideology": IDEOLOGICAL_LANGUAGE_KEYS,
    "narrative_conflict": NARRATIVE_CONFLICT_KEYS,
    "narrative_propagation": NARRATIVE_PROPAGATION_KEYS,
    "narrative_role": NARRATIVE_ROLE_KEYS,
    "narrative_temporal": NARRATIVE_TEMPORAL_KEYS,
    "source_attribution": SOURCE_ATTRIBUTION_KEYS,
    "propaganda_pattern": PROPAGANDA_PATTERN_KEYS,
}


__all__ = [
    "RHETORICAL_DEVICE_KEYS",
    "ARGUMENT_MINING_KEYS",
    "CONTEXT_OMISSION_KEYS",
    "DISCOURSE_COHERENCE_KEYS",
    "EMOTION_TARGET_KEYS",
    "FRAMING_KEYS",
    "INFORMATION_DENSITY_KEYS",
    "INFORMATION_OMISSION_KEYS",
    "IDEOLOGICAL_LANGUAGE_KEYS",
    "NARRATIVE_CONFLICT_KEYS",
    "NARRATIVE_PROPAGATION_KEYS",
    "NARRATIVE_ROLE_KEYS",
    "NARRATIVE_TEMPORAL_KEYS",
    "SOURCE_ATTRIBUTION_KEYS",
    "PROPAGANDA_PATTERN_KEYS",
    "ALL_FEATURE_KEYS",
]
