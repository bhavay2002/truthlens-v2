from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

from src.features.pipelines.feature_pipeline import (
    FeaturePipeline,
    partition_feature_sections,
)
from src.features.feature_schema_validator import FeatureSchemaValidator
from src.features.feature_statistics import FeatureStatistics
from src.features.fusion.feature_scaling import FeatureScalingPipeline
from src.features.fusion.feature_selection import FeatureSelectionPipeline
from src.features.feature_pruning import FeaturePruner              #  NEW
from src.features.feature_report import FeatureReport               #  NEW
from src.features.base.base_feature import FeatureContext

logger = logging.getLogger(__name__)


# =========================================================
# PIPELINE
# =========================================================

@dataclass
class FeatureEngineeringPipeline:

    pipeline: FeaturePipeline

    scaler: Optional[FeatureScalingPipeline] = None
    selector: Optional[FeatureSelectionPipeline] = None
    validator: Optional[FeatureSchemaValidator] = None

    #  NEW
    pruner: Optional[FeaturePruner] = None
    report_enabled: bool = False
    report_path: Optional[str] = None

    stats_enabled: bool = False

    # =====================================================
    # CORE PROCESS
    # =====================================================

    def process(
        self,
        contexts: List[FeatureContext],
        labels: Optional[List[int]] = None,
        fit: bool = False,
    ) -> List[Dict[str, float]]:

        if not contexts:
            raise ValueError("Input contexts cannot be empty")

        # -------------------------------------------------
        # 1. FEATURE EXTRACTION
        # -------------------------------------------------
        features = self.pipeline.batch_extract(contexts)

        # -------------------------------------------------
        # 2. SCHEMA VALIDATION
        # -------------------------------------------------
        if self.validator:
            features = self.validator.validate_batch(features)

        # -------------------------------------------------
        # 3.  FEATURE PRUNING
        # -------------------------------------------------
        # Audit fix §1.12 — pruning now runs BEFORE statistics. The
        # previous order (extract → validate → stats → prune → scale)
        # paid the full O(N²) correlation cost in ``FeatureStatistics``
        # on the un-pruned column set, only to throw 30-40 % of those
        # columns away in the very next step. Pruning first means stats
        # describe the post-prune matrix that the model will actually
        # train on, and the correlation work is bounded by the kept
        # column count.
        if self.pruner:
            try:
                if fit:
                    self.pruner.fit(features)
                    logger.info("Feature pruner fitted")

                features = self.pruner.transform(features)

            except Exception as e:
                logger.exception("Feature pruning failed: %s", e)
                raise

        # -------------------------------------------------
        # 4. FEATURE STATISTICS (post-prune)
        # -------------------------------------------------
        if self.stats_enabled and features:
            try:
                stats = FeatureStatistics()

                summary = stats.dataset_summary(features)
                variance = stats.compute_variance(features)
                skewness = stats.compute_skewness(features)

                logger.info(
                    "Feature stats | samples=%d features=%d mean_var=%.6f",
                    int(summary["num_samples"]),
                    int(summary["num_features"]),
                    summary["mean_variance"],
                )

                constant = stats.detect_constant_features(features)
                if constant:
                    logger.warning("Constant features: %s", constant[:10])

                low_variance = [k for k, v in variance.items() if v < 1e-6]
                if low_variance:
                    logger.warning("Low variance: %s", low_variance[:10])

                high_skew = [k for k, v in skewness.items() if abs(v) > 2.5]
                if high_skew:
                    logger.warning("High skew: %s", high_skew[:10])

            except Exception as e:
                logger.warning("Statistics failed: %s", e)

        # -------------------------------------------------
        # 5. SCALING
        # -------------------------------------------------
        if self.scaler:
            try:
                if fit:
                    self.scaler.fit(features)
                    logger.info("Scaler fitted")

                features = self.scaler.transform(features, return_array=False)

            except Exception as e:
                logger.exception("Scaling failed: %s", e)
                raise

        # -------------------------------------------------
        # 6. FEATURE SELECTION
        # -------------------------------------------------
        if self.selector:
            try:
                if fit:
                    self.selector.fit(features, labels)
                    logger.info("Selector fitted")

                features = self.selector.transform(features, return_array=False)

            except Exception as e:
                logger.exception("Feature selection failed: %s", e)
                raise

        # -------------------------------------------------
        # 7.  FEATURE REPORT (NEW)
        # -------------------------------------------------
        if self.report_enabled:
            try:
                reporter = FeatureReport()
                reporter.generate(features, save_path=self.report_path)
            except Exception as e:
                logger.warning("Feature report failed: %s", e)

        logger.info(
            "FeatureEngineeringPipeline complete | samples=%d features=%d",
            len(features),
            len(features[0]) if features else 0,
        )

        return features

    # =====================================================
    # SECTIONED OUTPUT
    # =====================================================

    def process_by_section(
        self,
        contexts: List[FeatureContext],
        labels: Optional[List[int]] = None,
        fit: bool = False,
    ) -> List[Dict[str, Dict[str, float]]]:

        flat = self.process(contexts, labels=labels, fit=fit)
        return [partition_feature_sections(f) for f in flat]

    # =====================================================
    # SINGLE SAMPLE
    # =====================================================

    def process_one(self, context: FeatureContext) -> Dict[str, float]:
        return self.process([context], fit=False)[0]

    # =====================================================
    # SINGLE SAMPLE (SECTIONED)
    # =====================================================

    def process_one_by_section(
        self,
        context: FeatureContext,
    ) -> Dict[str, Dict[str, float]]:
        return self.process_by_section([context], fit=False)[0]

    # =====================================================
    # DEBUG / INSPECTION
    # =====================================================

    def inspect_features(
        self,
        contexts: List[FeatureContext],
    ) -> Dict[str, Any]:

        features = self.process(contexts)

        stats = FeatureStatistics()

        return {
            "summary": stats.dataset_summary(features),
            "variance": stats.compute_variance(features),
            "skewness": stats.compute_skewness(features),
        }