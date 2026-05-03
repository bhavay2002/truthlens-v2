# Features Audit Report
**Scope:** `src/features/` — ~75 files  
**Date:** 2026-05-03  
**Prompts applied:** performance + signal quality  
**Status:** All actionable findings fixed ✓

---

## Summary

| ID | File | Severity | Category | Keys affected | Status |
|----|------|----------|----------|---------------|--------|
| DRY-01 | `base/numerics.py` | Low | DRY / code quality | `EPS` (duplicate) | Fixed |
| SCHEMA-01 | `bias/framing_features.py` | Critical | Schema drift | 6 keys | Fixed |
| SCHEMA-02 | `bias/ideological_features.py` | Critical | Schema drift | 6 keys | Fixed |
| SCHEMA-03 | `narrative/conflict_features.py` | Critical | Schema drift | 7 keys | Fixed |
| SCHEMA-04 | `propaganda/propaganda_features.py` | Critical | Schema drift | 9 keys | Fixed |
| INFO-01 | `emotion/emotion_features.py` | Info | Dead code / schema drift | — | No fix needed |
| INFO-02 | `emotion/emotion_intensity_features.py` | Info | Schema drift (active) | — | Documented |

**Critical fixes: 4 extractors, 28 key corrections.**

---

## Finding Detail

### DRY-01 — Duplicate `EPS` constant declaration (LOW)

**File:** `src/features/base/numerics.py` lines 31 and 39  
**Category:** DRY / code quality

**Description:**  
`EPS = 1e-8` was declared once above the standard-library imports at line 31 (alongside `MAX_CLIP = 1.0`) and again at line 39 after the `numpy` import. The second declaration shadowed the first with an identical value, so there was no behavioural difference, but it created confusion about the authoritative definition and violated the DRY principle that motivated centralising these constants in the first place.

**Fix:** Removed the second `EPS = 1e-8` declaration at line 39. `MAX_CLIP = 1.0` and `EPS = 1e-8` are now declared exactly once, before the imports.

---

### SCHEMA-01 — `FramingFeatures` emits wrong key names (CRITICAL)

**File:** `src/features/bias/framing_features.py`  
**Schema reference:** `feature_schema.FRAMING_FEATURES`  
**Category:** Schema drift — silent zero fill in downstream model

**Description:**  
`FramingFeatures.extract` returned keys that did not match `feature_schema.FRAMING_FEATURES`. The schema contract defines `_ratio`-suffixed names for the five category distribution slots and `frame_phrase_count` for the phrase-density slot. The extractor emitted bare names without the suffix, causing every schema slot to fill silently with `0.0` (the default for missing keys in `partition_feature_sections`). Additionally `frame_intensity` was emitted but is absent from the schema, wasting a float per document.

**Key corrections:**

| Old key (emitted) | New key (schema) |
|-------------------|------------------|
| `frame_economic` | `frame_economic_ratio` |
| `frame_moral` | `frame_moral_ratio` |
| `frame_security` | `frame_security_ratio` |
| `frame_human` | `frame_human_interest_ratio` |
| `frame_conflict` | `frame_conflict_ratio` |
| `frame_phrase_score` | `frame_phrase_count` |
| `frame_intensity` | *(removed — not in schema)* |

`frame_quote_density`, `frame_diversity`, `frame_entropy`, `frame_dominance` were already correctly named and unchanged.

**Cleanup:** The `intensity = float(np.mean(...))` computation was the sole source of the removed key; it has been deleted to eliminate the dead calculation.

---

### SCHEMA-02 — `IdeologicalFeatures` emits wrong key names (CRITICAL)

**File:** `src/features/bias/ideological_features.py`  
**Schema reference:** `feature_schema.IDEOLOGICAL_FEATURES`  
**Category:** Schema drift — silent zero fill in downstream model

**Description:**  
Same pattern as SCHEMA-01. The four category-distribution keys were missing the `_ratio` suffix, `ideology_phrase_score` did not match the schema's `ideology_phrase_count`, and `ideology_intensity` was emitted but not listed in the schema.

**Key corrections:**

| Old key (emitted) | New key (schema) |
|-------------------|------------------|
| `ideology_left` | `ideology_left_ratio` |
| `ideology_right` | `ideology_right_ratio` |
| `ideology_polarization` | `ideology_polarization_ratio` |
| `ideology_group_reference` | `ideology_group_reference_ratio` |
| `ideology_phrase_score` | `ideology_phrase_count` |
| `ideology_intensity` | *(removed — not in schema)* |

`ideology_balance`, `ideology_entropy`, `ideology_signal_strength` were already correct.

**Cleanup:** The `intensity = float(np.mean(...))` dead computation removed.

---

### SCHEMA-03 — `ConflictFeatures` emits wrong key names (CRITICAL)

**File:** `src/features/narrative/conflict_features.py`  
**Schema reference:** `feature_schema.CONFLICT_FEATURES`  
**Category:** Schema drift — silent zero fill in downstream model

**Description:**  
The six category-distribution keys lacked the `_ratio` suffix required by the schema. Additionally `conflict_entropy` was emitted but is absent from `CONFLICT_FEATURES`.

**Key corrections:**

| Old key (emitted) | New key (schema) |
|-------------------|------------------|
| `conflict_confrontation` | `conflict_confrontation_ratio` |
| `conflict_dispute` | `conflict_dispute_ratio` |
| `conflict_accusation` | `conflict_accusation_ratio` |
| `conflict_aggression` | `conflict_aggression_ratio` |
| `conflict_polarization` | `conflict_polarization_ratio` |
| `conflict_escalation` | `conflict_escalation_ratio` |
| `conflict_entropy` | *(removed — not in schema)* |

`conflict_intensity`, `conflict_diversity`, `conflict_rhetoric_score` were already correct.

**Cleanup:** The `entropy = normalized_entropy(probs)` block and associated `probs` array removed. The `normalized_entropy` import was also removed as it is now unused in this file.

---

### SCHEMA-04 — `PropagandaFeatures` emits wrong key names (CRITICAL)

**File:** `src/features/propaganda/propaganda_features.py`  
**Schema reference:** `feature_schema.PROPAGANDA_FEATURES`  
**Category:** Schema drift — silent zero fill in downstream model

**Description:**  
Seven category-distribution keys lacked the `_ratio` suffix. Additionally:

- `propaganda_entropy` was emitted but is absent from `PROPAGANDA_FEATURES`.
- `propaganda_rhetoric` was emitted as `exclamation_density + question_density`, but the schema slot is `propaganda_exclamation_density` (exclamation density only). The combined rhetoric composite is not in the schema.

**Key corrections:**

| Old key (emitted) | New key (schema) |
|-------------------|------------------|
| `propaganda_name_calling` | `propaganda_name_calling_ratio` |
| `propaganda_fear` | `propaganda_fear_ratio` |
| `propaganda_exaggeration` | `propaganda_exaggeration_ratio` |
| `propaganda_glitter` | `propaganda_glitter_ratio` |
| `propaganda_us_vs_them` | `propaganda_us_vs_them_ratio` |
| `propaganda_authority` | `propaganda_authority_ratio` |
| `propaganda_intensifier` | `propaganda_intensifier_ratio` |
| `propaganda_rhetoric` | `propaganda_exclamation_density` (value: `signals["exclamation_density"]`) |
| `propaganda_entropy` | *(removed — not in schema)* |

`propaganda_intensity`, `propaganda_diversity`, `propaganda_caps_ratio` were already correct.

**Cleanup:** The `probs` array, `entropy = normalized_entropy(probs)` computation, and `rhetoric = ...` variable removed as they were exclusively feeding the deleted/renamed output keys. The `normalized_entropy` import removed.

---

## INFO-01 — `emotion_features.py`: dead code with schema drift (INFO)

**File:** `src/features/emotion/emotion_features.py`  
**Category:** Dead code / informational

**Description:**  
`EmotionFeatures` is decorated with `@register_feature` but is **not imported** by `feature_bootstrap.py` (removed in a prior audit). Because it is never imported, it never registers and has no production impact. Its output would have schema drift (`emotion_coverage`, `emotion_entropy`, `emotion_polarity` are not in `EMOTION_FEATURES`) but this is moot.

**Action:** No fix applied. File is effectively dead. If re-activated, schema alignment will be required.

---

## INFO-02 — `emotion_intensity_features.py`: schema mismatch (INFO)

**File:** `src/features/emotion/emotion_intensity_features.py`  
**Category:** Active extractor with schema drift

**Description:**  
The active emotion extractor emits transformer-derived keys (`emotion_intensity_max`, `emotion_intensity_mean`, `emotion_intensity_std`, `emotion_intensity_range`, `emotion_intensity_l2`, `emotion_intensity_entropy`, `emotion_coverage`, `emotion_transformer_available`). None of these match the legacy `feature_schema.EMOTION_FEATURES` list (`emotion_<label>` per label + `emotion_intensity`). The schema list predates the transformer-based extractor.

**Action:** No fix applied in this audit. Correcting this requires updating `EMOTION_FEATURES` in `feature_schema.py` and verifying downstream model compatibility — a deliberate breaking change that should be tracked separately.

---

## Files Reviewed (no actionable findings)

### `base/`
`base_feature.py`, `feature_registry.py`, `lexicon_loader.py`, `lexicon_matcher.py`,
`text_signals.py`, `tokenization.py`

### `bias/`
`bias_features.py`, `manipulation_features.py`, `subjectivity_features.py`

### `emotion/`
`emotion_intensity_features.py` *(INFO-02 documented above)*,
`emotion_features.py` *(INFO-01 documented above)*

### `linguistic/`
`complexity_features.py`, `coherence_features.py`, `discourse_features.py`,
`hedging_features.py`, `lexical_diversity_features.py`, `negation_features.py`,
`passive_voice_features.py`, `pos_features.py`, `punctuation_features.py`,
`readability_features.py`, `sentence_structure_features.py`, `syntactic_features.py`,
`tense_features.py`

### `narrative/`
`causal_features.py`, `narrative_features.py`, `sensationalism_features.py`,
`story_arc_features.py`, `temporal_features.py`

### `network/`
`citation_features.py`, `source_features.py`, `url_features.py`

### `propaganda/`
`propaganda_lexicon_features.py` *(dead code, not in FEATURE_MODULES — no findings)*

### `semantic/`
`claim_features.py`, `entity_features.py`, `factual_density_features.py`,
`knowledge_features.py`, `semantic_similarity_features.py`, `topic_features.py`

### `structural/`
`document_structure_features.py`, `formatting_features.py`, `headline_features.py`,
`length_features.py`, `metadata_features.py`

### Root
`feature_bootstrap.py`, `feature_pipeline.py`, `feature_schema.py`,
`feature_schema_validator.py`
