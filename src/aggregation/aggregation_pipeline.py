from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List

import numpy as np

from src.aggregation.feature_mapper import FeatureMapper
from src.aggregation.calibration import get_calibrator
from src.aggregation.weight_manager import WeightManager
from src.aggregation.risk_assessment import (
    assess_truthlens_risks,
    from_pydantic_config as risk_from_pydantic,
)
from src.aggregation.score_explainer import ScoreExplainer
from src.aggregation.truthlens_score_calculator import TruthLensScoreCalculator
from src.aggregation.aggregation_config import AggregationConfig
from src.aggregation.aggregation_metrics import AggregationMetrics
from src.aggregation.score_schema import (
    TruthLensAggregationOutputModel,
    TruthLensScoreModel,
    TruthLensRiskModel,
    ExplanationModel,
    TaskScore,
    RiskValue,
)
from src.aggregation.aggregation_validator import AggregationValidator

# Aggregation Engine v2 — learned + hybrid scoring (spec §4–7)
from src.aggregation.feature_builder import AggregatorFeatureBuilder
from src.aggregation.hybrid_scorer import HybridScorer


logger = logging.getLogger(__name__)
EPS = 1e-12


# =========================================================
# Profile-section keys produced by `BiasProfileBuilder` that are NOT
# semantic feature dicts and must therefore be skipped when adapting
# a builder profile into a Branch B input for the FeatureMapper.
# =========================================================

_BUILDER_NON_FEATURE_KEYS = frozenset({
    "metadata",
    "bias_score",
})


class AggregationPipeline:

    def __init__(
        self,
        *,
        config: Optional[AggregationConfig] = None,
    ) -> None:

        self.config = config or AggregationConfig()

        # Calibrator is a single shared instance: it is fitted offline
        # (e.g. by `scripts/calibrate.py`) and used in passthrough mode
        # until then. CRIT-AG-6 moves the application of this
        # calibrator out of the per-feature aggregation step and into
        # the logit -> probability conversion in `FeatureMapper`.
        self.calibrator = get_calibrator(self.config.calibration.method)

        self.mapper = FeatureMapper(
            strict=self.config.strict_mode,
            normalize=False,        # NORM-AG-1: drop redundant max-norm
            calibrator=self.calibrator,
        )

        # CRIT-AG-9: surface YAML-driven weights into the manager so
        # that edits to `config.weights.weights` actually take effect.
        # CRIT-AG-7 + CRIT-AG-10 + WGT-AG-4 are also enforced inside
        # WeightManager itself.
        self.weight_manager = WeightManager(
            weights=self.config.weights.weights or None,
            version=self.config.weights.version,
            frozen=not self.config.weights.allow_dynamic_adjustment,
            smoothing=self.config.weights.smoothing,
            uncertainty_penalty=self.config.risk.uncertainty_penalty,
        )

        # WGT-AG-2 + WGT-AG-3: fusion constants are config-driven and
        # weights are always supplied externally (no hidden defaults
        # inside the calculator).
        self.calculator = TruthLensScoreCalculator(
            graph_influence_cap=self.config.fusion.graph_influence_cap,
            explanation_blend=self.config.fusion.explanation_blend,
        )

        self.explainer = ScoreExplainer(
            method=self.config.attribution.method
        )

        # CFG-AG-4: derive the runtime RiskConfig from the Pydantic
        # `risk` block (low/medium/uncertainty_penalty) so that edits to
        # config.yaml actually flow through, and the two `RiskConfig`
        # shapes don't drift apart.
        self.risk_config = risk_from_pydantic(
            self.config.risk,
            invert_keys=["truthlens_credibility_score"],
        )

        self.validator = AggregationValidator()

        # UNUSED-AG: AggregationMetrics is now actually instantiated
        # and updated per article so callers can inspect rolling
        # statistics through `pipeline.metrics.summarize()`.
        self.metrics = AggregationMetrics()

        # CRIT-AG-12: the entropy formula must distinguish multilabel
        # (Bernoulli) from multiclass (Categorical) tasks. Pull the
        # mapping from the explicit config field first, falling back
        # to the global app config.
        self._task_types = (
            dict(self.config.task_types)
            if self.config.task_types
            else self._load_task_types()
        )

        # ── Aggregation Engine v2 (spec §4–7) ──────────────────────────
        # AggregatorFeatureBuilder is always constructed — it is
        # lightweight (no weights) and its feature vector is useful even
        # when the neural path is disabled (e.g. for offline training
        # data collection via `analysis_modules["feature_vector"]`).
        self.feature_builder = AggregatorFeatureBuilder()

        # NeuralAggregator + HybridScorer are only instantiated when
        # `config.neural.enabled` is True AND a checkpoint exists.
        # When disabled, `_neural_module` is None and the pipeline
        # falls back to pure rule-based scoring (existing behaviour).
        self._neural_module: Optional[Any] = None
        self.hybrid_scorer: Optional[HybridScorer] = None

        if self.config.neural.enabled:
            self._init_neural()

        logger.info(
            "[AggregationPipeline] Initialized | neural=%s",
            self._neural_module is not None,
        )

    def _init_neural(self) -> None:
        """Instantiate the NeuralAggregator and HybridScorer from config.

        Called during ``__init__`` when ``config.neural.enabled`` is True.
        Errors are non-fatal: the pipeline logs a warning and continues
        in rule-only mode so a missing / corrupt checkpoint never blocks
        inference startup.
        """
        ncfg = self.config.neural
        try:
            import torch
            from src.aggregation.neural_aggregator import NeuralAggregator

            if ncfg.checkpoint_path:
                module = NeuralAggregator.load(
                    ncfg.checkpoint_path,
                    ncfg,
                    device="cpu",
                )
            else:
                module = NeuralAggregator.build(
                    ncfg,
                    input_dim=self.feature_builder.feature_dim,
                )
                module.eval()
                logger.warning(
                    "[AggregationPipeline] NeuralAggregator has no "
                    "checkpoint — running with random weights. "
                    "Train and set config.neural.checkpoint_path."
                )

            self._neural_module = module

            self.hybrid_scorer = HybridScorer(
                alpha=ncfg.alpha,
                dynamic=ncfg.dynamic_alpha,
                min_alpha=ncfg.alpha_min,
                max_alpha=ncfg.alpha_max,
            )
            logger.info(
                "[AggregationPipeline] NeuralAggregator ready | "
                "arch=%s dim=%d checkpoint=%s",
                ncfg.architecture,
                self.feature_builder.feature_dim,
                ncfg.checkpoint_path or "<untrained>",
            )
        except Exception as exc:
            logger.warning(
                "[AggregationPipeline] NeuralAggregator init failed — "
                "falling back to rule-based scoring. Error: %s",
                exc,
            )
            self._neural_module = None
            self.hybrid_scorer = None

    @staticmethod
    def _load_task_types() -> Dict[str, str]:
        try:
            from src.utils.config_loader import load_app_config
            app_cfg = load_app_config()
            return {
                name: tcfg.task_type
                for name, tcfg in getattr(app_cfg, "tasks", {}).items()
            }
        except Exception as exc:
            logger.debug(
                "[AggregationPipeline] could not load task_types from app config: %s",
                exc,
            )
            return {}

    # =====================================================
    # MAIN
    # =====================================================

    def run(
        self,
        model_outputs: Optional[Dict[str, Any]] = None,
        *,
        text: Optional[str] = None,
        profile: Optional[Dict[str, Any]] = None,
        analysis_modules: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the aggregation pipeline.

        CRIT-AG-1 / CRIT-AG-2: accept either ``model_outputs`` (a dict
        of task -> {logits/probabilities, ...}) or a pre-built
        ``profile`` from :class:`BiasProfileBuilder`. Three of the
        four call sites in the codebase pass a builder profile rather
        than raw model outputs, so support for that shape is
        first-class instead of a silent failure.

        Parameters
        ----------
        model_outputs:
            Multi-task prediction dict. Keys are task names; each
            value is a dict with ``probabilities`` and/or ``logits``.
            One of ``model_outputs`` / ``profile`` must be provided.
        text:
            Optional source text — kept for API compatibility and
            attached to the result for downstream consumers.
        profile:
            Optional pre-built profile (typically the output of
            ``BiasProfileBuilder.build_profile``). When provided it
            is converted into a Branch B input for the FeatureMapper.
        analysis_modules:
            Optional dict of analysis-module outputs to attach to
            ``result["analysis_modules"]["external_analysis"]``.
        """

        # CRIT-AG-1: accept either kind of input. The builder-profile
        # branch is what `analyze_article.py` and `truthlens_pipeline.py`
        # have always relied on; previously they crashed with an
        # unexpected-keyword-argument or returned all-zero scores.
        if model_outputs is None and profile is None:
            raise ValueError(
                "AggregationPipeline.run requires either "
                "`model_outputs` or `profile`"
            )

        if profile is not None and not model_outputs:
            source = self._adapt_profile(profile)
        else:
            source = model_outputs

        if not isinstance(source, dict):
            raise ValueError("AggregationPipeline.run input must be dict")

        # EDGE-AG: surface an empty / non-feature-shaped source instead
        # of silently returning an all-zero result. Callers occasionally
        # pass `model_outputs={}` from the API layer when upstream
        # inference failed — this would otherwise look like a confident
        # "low risk, low manipulation" prediction.
        if not source:
            logger.warning(
                "[AggregationPipeline] empty source — every section "
                "will be zero. Check upstream inference."
            )

        # =========================
        # 1. FEATURE MAPPING
        # =========================
        section_profile = self.mapper.map_from_model_outputs(source)

        # REC-AG-1: compute confidence + entropy ONCE and reuse the
        # cached `TaskSignal` everywhere. Previously the same softmax /
        # nan_to_num pass ran three times per article.
        task_signals = self.mapper.extract_task_signals(
            source, task_types=self._task_types
        )
        confidence = {t: s.confidence for t, s in task_signals.items()}
        entropy = {t: s.entropy for t, s in task_signals.items()}

        # =========================
        # 3. NORMALIZATION  — removed (CRIT-AG-5 / NORM-AG-1)
        #    The previous per-section minmax fit collapsed any
        #    1-feature section to 0 and binarised 2-feature sections
        #    to {0, 1} regardless of input magnitude. The mapper's
        #    own clip-to-[0,1] step is the single source of truth now;
        #    use ScoreNormalizer.load_state_dict(...) at startup if a
        #    population-level scaler is required.
        # =========================
        # 4. CALIBRATION  — removed (CRIT-AG-6)
        #    Calibration now happens inside FeatureMapper at the
        #    logit boundary where it is mathematically meaningful.
        # =========================
        profile_for_scoring = section_profile

        # =========================
        # 5. EXPLANATION (optional)
        # =========================
        explanation_scores: Dict[str, float] = {}
        explanations_raw: Dict[str, Any] = {}

        if self.config.enable_explanations:
            try:
                # GPU-AG-1: when an attribution model + tokenizer were
                # supplied AND the caller passed Branch-A model outputs
                # AND raw `text` is available, run real Integrated
                # Gradients (`explain_from_prediction`). Without all
                # three pieces we fall back to the cheap profile-based
                # heuristic — the previous code always took the
                # heuristic path even when a usable model was wired up.
                use_ig = (
                    self.config.attribution.method == "integrated_gradients"
                    and getattr(self.explainer, "model", None) is not None
                    and getattr(self.explainer, "tokenizer", None) is not None
                    and text is not None
                    and model_outputs is not None
                )

                if use_ig:
                    ig_raw = self.explainer.explain_from_prediction(
                        text=text,
                        predictor_output=model_outputs,
                        top_k=self.config.attribution.top_k,
                    ) or {}
                    # Aggregate per-task IG section scores into a single
                    # section -> score dict, and turn the per-task
                    # `top_tokens` into the (section, token, score)
                    # shape that `_build_explanation_model` expects.
                    section_totals: Dict[str, float] = {}
                    top_features: List[tuple] = []
                    for task, payload in ig_raw.items():
                        if not isinstance(payload, dict):
                            continue
                        for sec, val in (payload.get("section_scores") or {}).items():
                            try:
                                section_totals[sec] = (
                                    section_totals.get(sec, 0.0) + float(val)
                                )
                            except (TypeError, ValueError):
                                pass
                        for tok, score in payload.get("top_tokens", []):
                            top_features.append((task, str(tok), float(score)))
                    explanations_raw = {
                        "section_scores": section_totals,
                        "top_features": top_features,
                        "per_task": ig_raw,
                    }
                    explanation_scores = section_totals
                else:
                    explanations_raw = self.explainer.explain_profile(
                        profile_for_scoring,
                        top_k=self.config.attribution.top_k,
                    ) or {}
                    explanation_scores = explanations_raw.get("section_scores", {}) or {}
            except Exception as exc:
                logger.warning("[AggregationPipeline] explanation failed: %s", exc)
                explanations_raw = {}
                explanation_scores = {}

        # =========================
        # 6. ADAPTIVE WEIGHTS
        # =========================
        adaptive_weights = self.weight_manager.get_adaptive_weights(
            confidence=confidence if self.config.weights.use_confidence else None,
            entropy=entropy if self.config.weights.use_entropy else None,
            explanation_scores=(
                explanation_scores if self.config.weights.use_explainability else None
            ),
        )

        # =========================
        # 7. SCORING — adaptive weights forwarded into calculator
        # =========================
        scores_raw = self.calculator.compute_scores(
            profile_for_scoring,
            weights=adaptive_weights,
            explanation_scores=explanation_scores,
        )

        rule_final_score = float(scores_raw.get("final_score", 0.0))

        # =========================
        # 7a. NEURAL AGGREGATOR  (Aggregation Engine v2, spec §4–7)
        #     Builds the structured feature vector, runs the neural
        #     forward pass, then blends with the rule score.
        #     Guarded: disabled → neural_meta is None, pipeline is
        #     unchanged from v1.  Runtime error + fallback_on_error →
        #     neural_meta is None, rule score used as-is.
        # =========================
        neural_meta: Optional[Dict[str, Any]] = None
        hybrid_result: Optional[Dict[str, Any]] = None

        # Always build the feature vector so callers can use it for
        # offline training data collection even when neural is off.
        feature_vec = self.feature_builder.build(
            model_outputs=source,
            task_signals=task_signals,
            section_profile=section_profile,
            analyzer_features=None,
        )

        if self._neural_module is not None and self.hybrid_scorer is not None:
            try:
                import torch
                with torch.no_grad():
                    x_t = torch.from_numpy(feature_vec).unsqueeze(0).float()
                    agg_out = self._neural_module(x_t)

                neural_score = float(agg_out.credibility_score[0].item())
                risk_probs   = (
                    torch.softmax(agg_out.risk_logits[0], dim=-1)
                    .cpu()
                    .tolist()
                )
                exp_weights = agg_out.explanation_weights[0].cpu().tolist()

                # Confidence mean over all tasks for dynamic alpha
                conf_vals = list(confidence.values()) if confidence else []
                mean_conf = (
                    float(np.mean(conf_vals)) if conf_vals else None
                )

                hybrid_result = self.hybrid_scorer.score(
                    neural_score=neural_score,
                    rule_score=rule_final_score,
                    mean_confidence=mean_conf,
                    task_confidences=confidence,
                )

                feature_names = self.feature_builder.feature_names()
                neural_meta = {
                    "neural_credibility_score": neural_score,
                    "rule_final_score":  rule_final_score,
                    "hybrid_final_score": hybrid_result["final"],
                    "hybrid_alpha":      hybrid_result["alpha"],
                    "hybrid_mode":       hybrid_result["mode"],
                    "risk_probs":  {
                        "low":    risk_probs[0],
                        "medium": risk_probs[1],
                        "high":   risk_probs[2],
                    },
                    "top_feature_weights": dict(
                        sorted(
                            zip(feature_names, exp_weights),
                            key=lambda kv: kv[1],
                            reverse=True,
                        )[:10]
                    ),
                }
                logger.debug(
                    "[AggregationPipeline] neural=%.3f rule=%.3f "
                    "α=%.3f → hybrid=%.3f",
                    neural_score,
                    rule_final_score,
                    hybrid_result["alpha"],
                    hybrid_result["final"],
                )

            except Exception as exc:
                if self.config.neural.fallback_on_error:
                    logger.warning(
                        "[AggregationPipeline] Neural forward failed — "
                        "falling back to rule score. Error: %s", exc,
                    )
                    neural_meta = None
                    hybrid_result = None
                else:
                    raise

        # Resolve the final score: hybrid when neural ran, else rule-only.
        if hybrid_result is not None:
            scores_raw["final_score"] = hybrid_result["final"]

        # =========================
        # 8. RISK
        # =========================
        risks_dict: Dict[str, Any] = {}

        if self.config.enable_risk:
            risk_input = {
                "truthlens_manipulation_risk": self._safe_unit(scores_raw.get("manipulation_risk", 0.0)),
                "truthlens_credibility_score": self._safe_unit(scores_raw.get("credibility_score", 0.0)),
                "truthlens_final_score":       self._safe_unit(scores_raw.get("final_score", 0.0)),
            }
            risks_dict = assess_truthlens_risks(
                risk_input,
                probabilities=None,
                config=self.risk_config,
            )

        # =========================
        # 9. BUILD TYPED MODELS
        # =========================
        section_scores = scores_raw.get("section_scores", {})

        # Populate neural extension fields when the neural path ran.
        _neural_cred  = neural_meta["neural_credibility_score"] if neural_meta else None
        _hybrid_alpha = neural_meta["hybrid_alpha"]             if neural_meta else None

        scores_model = TruthLensScoreModel(
            tasks={
                section: TaskScore(score=self._safe_unit(val))
                for section, val in section_scores.items()
            },
            manipulation_risk=self._safe_unit(scores_raw.get("manipulation_risk", 0.0)),
            credibility_score=self._safe_unit(scores_raw.get("credibility_score", 0.0)),
            final_score=self._safe_unit(scores_raw.get("final_score", 0.0)),
            neural_credibility_score=(
                self._safe_unit(_neural_cred) if _neural_cred is not None else None
            ),
            hybrid_alpha=(
                float(np.clip(_hybrid_alpha, 0.0, 1.0))
                if _hybrid_alpha is not None else None
            ),
        )

        risks_model = self._build_risk_model(risks_dict)
        explanations_model = self._build_explanation_model(explanations_raw)

        # =========================
        # 10. ASSEMBLE RESULT
        # =========================
        result: Dict[str, Any] = {
            "schema_version": self.config.config_version,
            # CFG-3 (v13/v14 audit): pull the model_version label from
            # the config object so a single edit propagates to every
            # downstream consumer, instead of the previous hard-coded
            # "truthlens-v2" string literal.
            "model_version": self.config.model_version,

            "scores": scores_model.model_dump(),
            "raw_scores": {
                k: float(v)
                for k, v in scores_raw.items()
                if isinstance(v, (int, float)) and np.isfinite(v)
            },

            "risks": risks_model.model_dump(),
            "explanations": explanations_model.model_dump(),

            "analysis_modules": {
                "weights": adaptive_weights,
                "entropy": entropy,
                "confidence": confidence,
                # Feature vector always included so callers can log it
                # for offline aggregator training data collection.
                "feature_vector": feature_vec.tolist(),
                # Neural aggregator details (None → key present but null
                # when neural path is disabled/errored — preserves JSON
                # schema stability for downstream consumers).
                "neural_aggregator": neural_meta,
            },
        }

        # =========================
        # 11. EXTERNAL ANALYSIS MODULES (CRIT-AG-1)
        # =========================
        if analysis_modules:
            # Stored under a namespaced key so it does not clobber the
            # pipeline-owned entries above.
            result["analysis_modules"]["external_analysis"] = dict(analysis_modules)

        # =========================
        # 12. GRAPH INTEGRATION
        # =========================
        graph_output = (
            source.get("graph_output") if isinstance(source, dict) else None
        )

        if graph_output is not None:
            try:
                if hasattr(graph_output, "to_dict"):
                    result["analysis_modules"]["graph"] = graph_output.to_dict()
                else:
                    result["analysis_modules"]["graph"] = graph_output

                if hasattr(graph_output, "explanation"):
                    result["analysis_modules"]["graph_explanation"] = graph_output.explanation
                elif isinstance(graph_output, dict):
                    result["analysis_modules"]["graph_explanation"] = graph_output.get("explanation")
            except Exception as e:
                logger.warning("[AggregationPipeline] Graph injection failed: %s", e)

        # =========================
        # 13. VALIDATION
        # =========================
        flat_scores = {
            "credibility_score": scores_model.credibility_score,
            "manipulation_risk": scores_model.manipulation_risk,
            "final_score": scores_model.final_score,
        }
        validation = self.validator.validate({"scores": flat_scores})
        result["analysis_modules"]["validation"] = validation

        if not validation["valid"]:
            logger.warning(
                "[AggregationPipeline] Validation issues: %s",
                validation["issues"],
            )

        # =========================
        # 13a. METRICS + UNCERTAINTY THRESHOLDS
        # =========================
        # UNUSED-AG: wire `AggregationMetrics` (rolling history) and the
        # uncertainty `track_percentiles` / `p95_threshold` /
        # `p99_threshold` config block — previously declared but never
        # read.
        if self.config.monitoring.enabled:
            self.metrics.update(flat_scores)

        if (
            self.config.uncertainty.enable_entropy
            and self.config.uncertainty.track_percentiles
            and entropy
        ):
            ent_arr = np.asarray(list(entropy.values()), dtype=np.float64)
            if ent_arr.size:
                p95 = float(np.percentile(ent_arr, 95))
                p99 = float(np.percentile(ent_arr, 99))
                result["analysis_modules"]["uncertainty"] = {
                    "p95": p95,
                    "p99": p99,
                    "exceeds_p95_threshold": p95
                    > self.config.uncertainty.p95_threshold,
                    "exceeds_p99_threshold": p99
                    > self.config.uncertainty.p99_threshold,
                }

        # =========================
        # 14. FINAL SCHEMA VALIDATION
        # =========================
        validated = TruthLensAggregationOutputModel(**result)
        return validated.model_dump()

    # =====================================================
    # HELPERS
    # =====================================================

    @staticmethod
    def _safe_unit(v: Any) -> float:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(fv):
            return 0.0
        return float(np.clip(fv, 0.0, 1.0))

    @staticmethod
    def _adapt_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a BiasProfileBuilder profile into Branch B input.

        Drops bookkeeping entries (``metadata``, ``bias_score``) and
        retains every dict-valued section verbatim. The FeatureMapper
        then forwards each section's already-numeric features straight
        through, so the calculator sees the analysis-side signal that
        the rest of the pipeline produced.
        """
        if not isinstance(profile, dict):
            return profile or {}

        out: Dict[str, Dict[str, Any]] = {}
        for k, v in profile.items():
            if k in _BUILDER_NON_FEATURE_KEYS:
                continue
            if isinstance(v, dict):
                out[k] = v
        return out

    def build_profile_from_prediction(
        self,
        prediction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Convert raw model outputs to a per-section feature profile.

        Provided for legacy callers (``inference_pipeline.py:339``)
        that expect the pipeline to expose this helper. CRIT-AG-1.
        """
        if not isinstance(prediction, dict):
            raise ValueError("prediction must be dict")
        return self.mapper.map_from_model_outputs(prediction)

    # =====================================================
    # ENTROPY — moved into FeatureMapper.extract_task_signals
    # (REC-AG-1). The previous in-pipeline `_compute_entropy` ran
    # the same softmax/nan_to_num pass that the mapper had already
    # done in `extract_confidence`, tripling work per article.
    # =====================================================

    # =====================================================
    # RISK MODEL BUILDER
    # =====================================================

    def _build_risk_model(self, risks_dict: Dict[str, Any]) -> TruthLensRiskModel:

        def _rv(data: Any) -> Optional[RiskValue]:
            if not isinstance(data, dict):
                return None
            level = data.get("level")
            score = data.get("score")
            if level not in ("LOW", "MEDIUM", "HIGH"):
                return None
            return RiskValue(level=level, score=score)

        return TruthLensRiskModel(
            manipulation_risk=_rv(risks_dict.get("manipulation_risk")),
            credibility_level=_rv(risks_dict.get("credibility_level")),
            overall_truthlens_rating=_rv(risks_dict.get("overall_truthlens_rating")),
        )

    # =====================================================
    # EXPLANATION MODEL BUILDER
    # =====================================================

    def _build_explanation_model(self, raw: Dict[str, Any]) -> ExplanationModel:

        if not raw:
            return ExplanationModel(sections={})

        section_scores = raw.get("section_scores", {})
        top_features = raw.get("top_features", [])

        method = self.config.attribution.method
        sections: Dict[str, Any] = {}

        for section, score in section_scores.items():

            section_feats = [
                (k, v)
                for s, k, v in top_features
                if s == section
            ]

            attributions = [
                {
                    "token": str(k),
                    "importance": float(abs(v)),
                    "contribution": float(v),
                    "direction": "positive" if v >= 0.0 else "negative",
                }
                for k, v in section_feats[: self.config.attribution.top_k]
            ]

            sections[section] = {
                "method": method,
                "top_features": [a["token"] for a in attributions],
                "attributions": attributions,
                "section_score": float(np.clip(score, 0.0, 1.0)),
            }

        return ExplanationModel(sections=sections)

    # =====================================================
    # BATCH (PERF-AG-5)
    # =====================================================

    def run_batch(self, batch_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run aggregation over a batch.

        Now that CRIT-AG-5 / CRIT-AG-6 removed the per-article
        mutations from the normalizer/calibrator, the per-article
        pipeline is effectively stateless. When
        ``config.batch_max_workers > 1`` the calls fan out to a
        thread pool — numpy releases the GIL during its hot loops
        so this gives a real speedup on CPU-bound batches without
        the correctness risk that the previous (mutating) version
        would have had.
        """
        workers = max(1, int(self.config.batch_max_workers))
        if workers <= 1 or len(batch_outputs) <= 1:
            return [self.run(x) for x in batch_outputs]

        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(self.run, batch_outputs))
