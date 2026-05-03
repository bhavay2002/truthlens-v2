from __future__ import annotations

from typing import Dict, Any, Tuple, List
import numpy as np
from pydantic import BaseModel, Field, field_validator

from src.analysis.feature_schema import (
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
    NARRATIVE_TEMPORAL_KEYS,
    SOURCE_ATTRIBUTION_KEYS,
    PROPAGANDA_PATTERN_KEYS,
    make_vector,
)


# =========================================================
# BASE MODEL
# =========================================================

class FeatureModel(BaseModel):
    """
    Base class:
    - numeric safety
    - vector conversion
    - completeness tracking
    """

    @field_validator("*", mode="before")
    @classmethod
    def _validate_numeric(cls, v):
        if v is None:
            return 0.0
        if not isinstance(v, (int, float)):
            return 0.0
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, 1.0))

    # -------------------------

    def to_dict(self) -> Dict[str, float]:
        return self.model_dump()

    # -------------------------

    def to_vector(
        self,
        keys: Tuple[str, ...],
        *,
        strict: bool = False,
    ) -> np.ndarray:
        return make_vector(
            self.model_dump(),
            keys,
            strict=strict,
        )

    # -------------------------

    def completeness(self, keys: Tuple[str, ...]) -> float:
        present = sum(1 for k in keys if k in self.model_dump())
        return present / max(len(keys), 1)


# =========================================================
# FEATURE MODELS
# =========================================================

class RhetoricalFeatures(FeatureModel):
    rhetoric_exaggeration_score: float = 0.0
    rhetoric_loaded_language_score: float = 0.0
    rhetoric_emotional_appeal_score: float = 0.0
    rhetoric_fear_appeal_score: float = 0.0
    rhetoric_intensifier_ratio: float = 0.0
    rhetoric_scapegoating_score: float = 0.0
    rhetoric_false_dilemma_score: float = 0.0
    rhetoric_punctuation_score: float = 0.0

    def vector(self):
        return self.to_vector(RHETORICAL_DEVICE_KEYS)


class ArgumentFeatures(FeatureModel):
    argument_claim_ratio: float = 0.0
    argument_premise_ratio: float = 0.0
    argument_support_ratio: float = 0.0
    argument_contrast_ratio: float = 0.0
    argument_rebuttal_ratio: float = 0.0
    argument_verb_density: float = 0.0
    argument_clause_density: float = 0.0
    argument_complexity: float = 0.0

    def vector(self):
        return self.to_vector(ARGUMENT_MINING_KEYS)


class ContextFeatures(FeatureModel):
    context_vague_reference_ratio: float = 0.0
    context_attribution_ratio: float = 0.0
    context_evidence_ratio: float = 0.0
    context_uncertainty_ratio: float = 0.0
    context_quote_ratio: float = 0.0
    context_entity_ratio: float = 0.0
    context_entity_type_diversity: float = 0.0
    context_grounding_score: float = 0.0

    def vector(self):
        return self.to_vector(CONTEXT_OMISSION_KEYS)


class DiscourseFeatures(FeatureModel):
    sentence_coherence: float = 0.0
    topic_drift: float = 0.0
    narrative_continuity: float = 0.0
    # F14: canonical alias of `narrative_continuity`. See feature_keys.
    entity_repetition_ratio: float = 0.0
    discourse_transition_ratio: float = 0.0

    def vector(self):
        return self.to_vector(DISCOURSE_COHERENCE_KEYS)


class EmotionFeatures(FeatureModel):
    emotion_target_diversity: float = 0.0
    emotion_target_focus: float = 0.0
    emotion_expression_ratio: float = 0.0
    emotion_type_diversity: float = 0.0
    dominant_emotion_strength: float = 0.0

    def vector(self):
        return self.to_vector(EMOTION_TARGET_KEYS)


class FramingFeatures(FeatureModel):
    frame_conflict_score: float = 0.0
    frame_economic_score: float = 0.0
    frame_moral_score: float = 0.0
    frame_human_interest_score: float = 0.0
    frame_security_score: float = 0.0
    frame_dominance_score: float = 0.0
    frame_diversity_score: float = 0.0

    def vector(self):
        return self.to_vector(FRAMING_KEYS)


class InformationFeatures(FeatureModel):
    factual_density: float = 0.0
    opinion_density: float = 0.0
    claim_density: float = 0.0
    rhetorical_density: float = 0.0
    emotion_density: float = 0.0
    modal_density: float = 0.0
    rhetorical_punctuation_density: float = 0.0
    information_emotion_ratio: float = 0.0
    # F3: emitted by InformationDensityAnalyzer + present in
    # INFORMATION_DENSITY_KEYS but previously dropped here, which
    # silently truncated the feature row.
    information_diversity: float = 0.0

    def vector(self):
        return self.to_vector(INFORMATION_DENSITY_KEYS)


class IdeologyFeatures(FeatureModel):
    liberty_language_ratio: float = 0.0
    equality_language_ratio: float = 0.0
    tradition_language_ratio: float = 0.0
    anti_elite_language_ratio: float = 0.0
    liberty_vs_equality_balance: float = 0.0
    ideology_phrase_density: float = 0.0
    # F3
    ideology_diversity: float = 0.0

    def vector(self):
        return self.to_vector(IDEOLOGICAL_LANGUAGE_KEYS)


class SourceAttributionFeatures(FeatureModel):
    expert_attribution_ratio: float = 0.0
    anonymous_source_ratio: float = 0.0
    credibility_indicator_ratio: float = 0.0
    attribution_verb_ratio: float = 0.0
    quotation_ratio: float = 0.0
    named_source_ratio: float = 0.0
    source_credibility_balance: float = 0.0
    # F3
    attribution_intensity: float = 0.0
    attribution_diversity: float = 0.0

    def vector(self):
        return self.to_vector(SOURCE_ATTRIBUTION_KEYS)


# =========================================================
# PROPAGANDA
# =========================================================

class PropagandaFeatures(FeatureModel):
    fear_propaganda_score: float = 0.0
    scapegoating_score: float = 0.0
    polarization_score: float = 0.0
    emotional_amplification_score: float = 0.0
    narrative_imbalance_score: float = 0.0
    propaganda_intensity: float = 0.0
    propaganda_diversity: float = 0.0

    def vector(self):
        return self.to_vector(PROPAGANDA_PATTERN_KEYS)


# =========================================================
# PROFILE
# =========================================================

class BiasProfile(BaseModel):
    # Core sections handled by `BiasProfileBuilder.build_profile`.
    bias: Dict[str, float]
    emotion: Dict[str, float]
    narrative: Dict[str, float]
    discourse: Dict[str, float]
    ideology: Dict[str, float]
    metrics: Dict[str, float]
    bias_score: float

    # F3: optional sections that newer analyzers add to the profile.
    # Default to empty dicts so older callers continue to validate.
    argument: Dict[str, float] = Field(default_factory=dict)
    source: Dict[str, float] = Field(default_factory=dict)
    context: Dict[str, float] = Field(default_factory=dict)
    propaganda: Dict[str, float] = Field(default_factory=dict)
    information_omission: Dict[str, float] = Field(default_factory=dict)
    narrative_role: Dict[str, float] = Field(default_factory=dict)
    narrative_conflict: Dict[str, float] = Field(default_factory=dict)
    narrative_propagation: Dict[str, float] = Field(default_factory=dict)
    narrative_temporal: Dict[str, float] = Field(default_factory=dict)


# =========================================================
# PIPELINE OUTPUT
# =========================================================

class PipelineOutput(BaseModel):
    rhetorical: Dict[str, float] = Field(default_factory=dict)
    argument: Dict[str, float] = Field(default_factory=dict)
    context: Dict[str, float] = Field(default_factory=dict)
    discourse: Dict[str, float] = Field(default_factory=dict)
    emotion: Dict[str, float] = Field(default_factory=dict)
    framing: Dict[str, float] = Field(default_factory=dict)
    information: Dict[str, float] = Field(default_factory=dict)
    ideology: Dict[str, float] = Field(default_factory=dict)
    source: Dict[str, float] = Field(default_factory=dict)

    # F3: surface the sections produced by analyzers that were
    # registered later (information_omission, narrative_role, the three
    # narrative_* analyzers, and the propaganda detector). Without
    # these, those analyzer outputs were merged but then dropped at
    # serialization time.
    information_omission: Dict[str, float] = Field(default_factory=dict)
    narrative_role: Dict[str, float] = Field(default_factory=dict)
    narrative_conflict: Dict[str, float] = Field(default_factory=dict)
    narrative_propagation: Dict[str, float] = Field(default_factory=dict)
    narrative_temporal: Dict[str, float] = Field(default_factory=dict)
    propaganda: Dict[str, float] = Field(default_factory=dict)


# =========================================================
# FINAL OUTPUT
# =========================================================

class FullAnalysisOutput(BaseModel):
    features: PipelineOutput
    profile: BiasProfile
    propaganda: PropagandaFeatures
    meta: Dict[str, Any] = Field(default_factory=dict)

    # -----------------------------------------------------
    # GLOBAL VECTOR (ORDERED + STABLE)
    # -----------------------------------------------------

    def to_vector(self) -> np.ndarray:
        parts: List[float] = []

        # 🔥 enforce deterministic ordering. Newer sections (F3) are
        # appended at the end so the historical prefix of the vector
        # remains stable for any downstream model trained on the old
        # layout.
        ordered_sections = [
            "rhetorical",
            "argument",
            "context",
            "discourse",
            "emotion",
            "framing",
            "information",
            "ideology",
            "source",
            "information_omission",
            "narrative_role",
            "narrative_conflict",
            "narrative_propagation",
            "narrative_temporal",
            "propaganda",
        ]

        feature_dict = self.features.model_dump()

        for section_name in ordered_sections:
            section = feature_dict.get(section_name, {})
            if isinstance(section, dict):
                for k in sorted(section.keys()):
                    v = section[k]
                    if isinstance(v, (int, float)):
                        parts.append(float(v))
                    else:
                        parts.append(0.0)

        return np.asarray(parts, dtype=np.float32)

    # -----------------------------------------------------
    # DEBUG / INSPECTION
    # -----------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        return {
            "num_features": len(self.to_vector()),
            "confidence": self.meta.get("confidence", 0.0),
            "has_error": self.meta.get("error", False),
        }