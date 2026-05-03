from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

from src.aggregation.aggregation_pipeline import AggregationPipeline
from src.explainability.explanation_report_generator import ExplanationReportGenerator
from src.inference.constants import REPORT_VERSION
from src.utils import create_folder, save_json, timestamp

logger = logging.getLogger(__name__)


# =========================================================
# DATA CLASSES
# =========================================================

@dataclass
class ArticleSummary:
    title: Optional[str]
    source: Optional[str]
    word_count: Optional[int]
    analyzed_at: str


@dataclass
class ReportConfig:
    include_timestamp: bool = True
    pretty_json: bool = True
    validate_fields: bool = True

    save_explanation_artifacts: bool = False
    explanation_output_dir: str = "reports/explanations"

    include_evaluation: bool = True
    include_uncertainty: bool = True
    include_calibration: bool = True


# =========================================================
# MAIN GENERATOR
# =========================================================

class ReportGenerator:

    def __init__(
        self,
        config: Optional[ReportConfig] = None,
        aggregation_pipeline: Optional[AggregationPipeline] = None,
    ):
        self.config = config or ReportConfig()
        self.aggregation_pipeline = aggregation_pipeline or AggregationPipeline()

        self._explanation_reporter = ExplanationReportGenerator(
            output_dir=self.config.explanation_output_dir
        )

        logger.info("ReportGenerator initialized")

    def _current_timestamp(self) -> str:
        return timestamp().replace("_", "T") + "Z"

    def _safe(self, obj):
        if obj is None:
            return {}
        if not isinstance(obj, dict):
            raise TypeError("Expected dict")
        return obj

    # =====================================================
    # 🔥 MAIN REPORT (UPDATED)
    # =====================================================

    def generate_report(
        self,
        *,
        article_text: str,
        title: Optional[str] = None,
        source: Optional[str] = None,
        analysis: Optional[Dict[str, Any]] = None,
        predictions: Optional[Dict[str, Any]] = None,
        evaluation: Optional[Dict[str, Any]] = None,
        calibration: Optional[Dict[str, Any]] = None,
        uncertainty: Optional[Dict[str, Any]] = None,
        task_correlation: Optional[Dict[str, Any]] = None,
        explainability: Optional[Dict[str, Any]] = None,
        article_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        word_count = len(article_text.split())

        summary = ArticleSummary(
            title=title,
            source=source,
            word_count=word_count,
            analyzed_at=self._current_timestamp() if self.config.include_timestamp else "",
        )

        analysis = analysis or {}

        aggregation = analysis.get("aggregation")

        # REC-4: previously, when the caller passed a ``profile`` but no
        # ``aggregation``, this method silently re-ran the aggregation
        # pipeline on the report-generation hot path — duplicating work
        # the upstream analyzer had already done. Aggregation is now
        # mandatory: callers (``ArticleAnalyzer.analyze``, batch jobs,
        # etc.) must compute it once and pass it in. Re-aggregating here
        # would also bypass any analyzer-level caching/feature-flags.
        if not aggregation:
            if isinstance(analysis.get("profile"), dict):
                raise ValueError(
                    "ReportGenerator.generate_report: 'profile' was provided "
                    "without 'aggregation'. Recomputing aggregation here is "
                    "unsafe (duplicates upstream work and ignores caller "
                    "config). Run AggregationPipeline upstream and pass the "
                    "result via analysis['aggregation']."
                )
            aggregation = {}

        # =====================================================
        # 🔥 NEW EXTRACTION BLOCK
        # =====================================================

        analysis_modules = analysis.get("analysis_modules", {})

        graph = analysis.get("graph") or analysis_modules.get("graph")
        graph_expl = analysis.get("graph_explanation") or analysis_modules.get("graph_explanation")

        drift = analysis.get("drift") or analysis_modules.get("drift")
        monitoring = analysis.get("monitoring") or analysis_modules.get("monitoring")

        # =====================================================
        # MAIN REPORT STRUCTURE
        # =====================================================

        report: Dict[str, Any] = {
            "article_summary": asdict(summary),

            "predictions": self._safe(predictions),

            "analysis": {
                "bias_features": analysis.get("bias_features", {}),
                "emotion_features": analysis.get("emotion_features", {}),
                "narrative_features": analysis.get("narrative_features", {}),
                "graph_features": analysis.get("graph_features", {}),
                "analysis_modules": analysis_modules,
            },

            "aggregation": self._safe(aggregation),

            "evaluation": self._safe(evaluation) if self.config.include_evaluation else {},
            "calibration": self._safe(calibration) if self.config.include_calibration else {},
            "uncertainty": self._safe(uncertainty) if self.config.include_uncertainty else {},

            "task_correlation": self._safe(task_correlation),

            "explainability": self._safe(explainability),

            # =====================================================
            # 🔥 NEW SECTIONS (YOUR REQUIREMENT)
            # =====================================================
            "graph": graph,
            "graph_explanation": graph_expl,
            "drift": drift,
            "monitoring": monitoring,
        }

        # =====================================================
        # EXPLAINABILITY SAVE
        # =====================================================

        if explainability and self.config.save_explanation_artifacts and article_id:
            try:
                paths = self._explanation_reporter.generate(
                    article_id=article_id,
                    explanation=explainability,
                    save_json=True,
                    save_html=True,
                )
                report["explainability_artifacts"] = {
                    k: str(v) for k, v in paths.items()
                }
            except Exception as e:
                logger.warning("Explainability save failed: %s", e)

        # =====================================================
        # METADATA
        # =====================================================

        report["metadata"] = {
            # CFG-6: read from the constants module so the version bumps
            # in exactly one place when the report schema changes.
            "report_version": REPORT_VERSION,
            "generated_at": summary.analyzed_at,
            "tasks": list(predictions.keys()) if predictions else [],
        }

        # =====================================================
        # RISK FLAG
        # =====================================================

        if uncertainty and "mean_entropy" in uncertainty:
            report["risk_level"] = (
                "high" if uncertainty["mean_entropy"] > 1.5 else "normal"
            )

        logger.info("Report generated successfully")

        return report

    # =====================================================
    # JSON
    # =====================================================

    def to_json(self, report: Dict[str, Any]) -> str:
        return json.dumps(
            report,
            indent=4 if self.config.pretty_json else None,
            ensure_ascii=False,
        )

    def save_json(self, report: Dict[str, Any], path: str):

        path_obj = Path(path)
        create_folder(path_obj.parent)

        save_json(
            report,
            path_obj,
            indent=4 if self.config.pretty_json else 2,
        )

        logger.info("Report saved: %s", path)