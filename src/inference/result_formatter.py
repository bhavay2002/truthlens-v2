from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np

from src.utils import timestamp

logger = logging.getLogger(__name__)


# =========================================================
# DATA CLASSES (UPDATED)
# =========================================================

@dataclass
class TruthLensAPIResponse:
    predictions: Dict[str, Any]
    confidence: Dict[str, float]
    uncertainty: Optional[Dict[str, float]]
    credibility_score: Optional[float]

    # 🔥 NEW
    graph: Optional[Dict[str, Any]]
    graph_explanation: Optional[Dict[str, Any]]
    drift: Optional[Dict[str, Any]]
    monitoring: Optional[Dict[str, Any]]

    timestamp: str


@dataclass
class TruthLensDashboardReport:
    article_summary: Dict[str, Any]
    predictions: Dict[str, Any]
    evaluation: Dict[str, Any]
    calibration: Dict[str, Any]
    uncertainty: Dict[str, Any]
    aggregation: Dict[str, Any]

    # 🔥 NEW
    graph: Optional[Dict[str, Any]]
    graph_explanation: Optional[Dict[str, Any]]
    drift: Optional[Dict[str, Any]]
    monitoring: Optional[Dict[str, Any]]

    generated_at: str


@dataclass
class TruthLensResearchExport:
    article_summary: Dict[str, Any]
    predictions: Dict[str, Any]
    logits: Dict[str, Any]
    probabilities: Dict[str, Any]
    calibration: Dict[str, Any]
    uncertainty: Dict[str, Any]
    evaluation: Dict[str, Any]
    correlation: Dict[str, Any]

    # 🔥 NEW
    graph: Optional[Dict[str, Any]]
    graph_explanation: Optional[Dict[str, Any]]
    drift: Optional[Dict[str, Any]]
    monitoring: Optional[Dict[str, Any]]

    intermediate_features: Optional[Dict[str, Any]]
    model_metadata: Optional[Dict[str, Any]]
    generated_at: str


# =========================================================
# FORMATTER
# =========================================================

class ResultFormatter:

    def __init__(self):
        logger.info("ResultFormatter initialized")

    def _timestamp(self):
        return timestamp().replace("_", "T") + "Z"

    # =====================================================
    # 🔥 API FORMAT (UPDATED)
    # =====================================================

    def format_api_response(self, report: Dict[str, Any]):

        preds = report.get("predictions", {})
        uncertainty = report.get("uncertainty", {})

        formatted_preds = {}
        confidence = {}

        for task, out in preds.items():

            if not isinstance(out, dict):
                continue

            probs = out.get("probabilities")
            pred = out.get("predictions")

            if probs is not None:
                probs = np.asarray(probs)
                conf = float(np.max(probs[0]))
            else:
                conf = None

            formatted_preds[task] = int(pred[0]) if pred is not None else None
            confidence[task] = conf

        # 🔥 NEW EXTRACTION
        graph = report.get("graph") or report.get("analysis_modules", {}).get("graph")
        graph_expl = report.get("graph_explanation") or report.get("analysis_modules", {}).get("graph_explanation")
        drift = report.get("drift")
        monitoring = report.get("monitoring")

        response = TruthLensAPIResponse(
            predictions=formatted_preds,
            confidence=confidence,
            uncertainty=uncertainty,
            credibility_score=report.get("aggregation", {}).get("scores", {}).get("truthlens_credibility_score"),

            # 🔥 NEW
            graph=graph,
            graph_explanation=graph_expl,
            drift=drift,
            monitoring=monitoring,

            timestamp=self._timestamp(),
        )

        return asdict(response)

    # =====================================================
    # 🔥 DASHBOARD FORMAT (UPDATED)
    # =====================================================

    def format_dashboard_report(self, report: Dict[str, Any]):

        dashboard = TruthLensDashboardReport(
            article_summary=report.get("article_summary", {}),
            predictions=report.get("predictions", {}),
            evaluation=report.get("evaluation", {}),
            calibration=report.get("calibration", {}),
            uncertainty=report.get("uncertainty", {}),
            aggregation=report.get("aggregation", {}),

            # 🔥 NEW
            graph=report.get("graph") or report.get("analysis_modules", {}).get("graph"),
            graph_explanation=report.get("graph_explanation") or report.get("analysis_modules", {}).get("graph_explanation"),
            drift=report.get("drift"),
            monitoring=report.get("monitoring"),

            generated_at=self._timestamp(),
        )

        return asdict(dashboard)

    # =====================================================
    # 🔥 RESEARCH EXPORT (UPDATED)
    # =====================================================

    def format_research_export(
        self,
        report: Dict[str, Any],
        model_metadata: Optional[Dict[str, Any]] = None,
        features: Optional[Dict[str, Any]] = None,
    ):

        preds = report.get("predictions", {})

        logits = {}
        probs = {}

        for task, out in preds.items():
            if isinstance(out, dict):
                logits[task] = out.get("logits")
                probs[task] = out.get("probabilities")

        export = TruthLensResearchExport(
            article_summary=report.get("article_summary", {}),
            predictions=preds,
            logits=logits,
            probabilities=probs,
            calibration=report.get("calibration", {}),
            uncertainty=report.get("uncertainty", {}),
            evaluation=report.get("evaluation", {}),
            correlation=report.get("task_correlation", {}),

            # 🔥 NEW
            graph=report.get("graph") or report.get("analysis_modules", {}).get("graph"),
            graph_explanation=report.get("graph_explanation") or report.get("analysis_modules", {}).get("graph_explanation"),
            drift=report.get("drift"),
            monitoring=report.get("monitoring"),

            intermediate_features=features,
            model_metadata=model_metadata,
            generated_at=self._timestamp(),
        )

        return asdict(export)

    # =====================================================
    # JSON
    # =====================================================

    def to_json(self, data: Dict[str, Any], pretty=True):

        try:
            return json.dumps(
                data,
                indent=4 if pretty else None,
                ensure_ascii=False,
            )
        except Exception as e:
            logger.exception("Serialization failed")
            raise RuntimeError("JSON serialization failed") from e