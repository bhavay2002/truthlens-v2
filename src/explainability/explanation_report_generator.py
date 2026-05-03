from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)


class ExplanationReportGenerator:

    def __init__(self, output_dir: str | Path = "reports/explanations") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # UTILS
    # =====================================================

    def _safe_article_id(self, article_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", article_id.strip())
        return safe.strip("._") or "article"

    def _paths(self, article_id: str):
        sid = self._safe_article_id(article_id)
        return {
            "json": self.output_dir / f"{sid}.json",
            "html": self.output_dir / f"{sid}.html",
        }

    # =====================================================
    # TOKEN HIGHLIGHT
    # =====================================================

    def _highlight_tokens(self, tokens: List[str], scores: List[float]) -> str:

        if not tokens or not scores:
            return "<p>No token importance available</p>"

        max_score = max(scores) + 1e-12

        html_tokens = []
        for t, s in zip(tokens, scores):
            color = f"rgba(255,0,0,{s / max_score:.2f})"
            html_tokens.append(
                f'<span style="background:{color}; padding:2px; margin:1px;">{escape(t)}</span>'
            )

        return "<p>" + " ".join(html_tokens) + "</p>"

    # =====================================================
    # JSON
    # =====================================================

    def save_json(self, article_id: str, explanation: Dict[str, Any]) -> Path:

        path = self._paths(article_id)["json"]

        payload = {
            "article_id": article_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "version": "v3",
            "explanation": explanation,
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return path

    # =====================================================
    # HTML REPORT (UPGRADED)
    # =====================================================

    def save_html(self, article_id: str, explanation: Dict[str, Any]) -> Path:

        path = self._paths(article_id)["html"]

        prediction = explanation.get("prediction", {})
        scores = explanation.get("scores", {})
        risks = explanation.get("risks", {})
        metrics = explanation.get("metrics", {})
        agg = explanation.get("aggregated_explanation", {})
        monitoring = explanation.get("monitoring", {})

        tokens = agg.get("tokens", [])
        importance = agg.get("final_token_importance", [])

        highlighted = self._highlight_tokens(tokens, importance)

        # 🔥 NEW: entropy collection
        entropy_data = {
            "aggregated_entropy": agg.get("entropy"),
            "methods": {
                "shap": explanation.get("shap_explanation", {}).get("entropy") if explanation.get("shap_explanation") else None,
                "lime": explanation.get("lime_explanation", {}).get("entropy") if explanation.get("lime_explanation") else None,
                "attention": explanation.get("attention_explanation", {}).get("entropy") if explanation.get("attention_explanation") else None,
            }
        }

        html = f"""
        <html>
        <head>
        <title>TruthLens Report</title>
        <style>
            body {{ font-family: Arial; margin: 20px; }}
            h1 {{ color: #333; }}
            .section {{ margin-bottom: 30px; }}
            .card {{ padding: 10px; border: 1px solid #ddd; }}
        </style>
        </head>
        <body>

        <h1>TruthLens Explainability Report</h1>

        <div class="section card">
        <h2>Prediction</h2>
        <pre>{escape(json.dumps(prediction, indent=2))}</pre>
        </div>

        <div class="section card">
        <h2>Scores</h2>
        <pre>{escape(json.dumps(scores, indent=2))}</pre>
        </div>

        <div class="section card">
        <h2>Risk Assessment</h2>
        <pre>{escape(json.dumps(risks, indent=2))}</pre>
        </div>

        <div class="section card">
        <h2>Token Importance</h2>
        {highlighted}
        </div>

        <div class="section card">
        <h2>Explainability Metrics</h2>
        <pre>{escape(json.dumps(metrics, indent=2))}</pre>
        </div>

        <div class="section card">
        <h2>Method Contributions</h2>
        <pre>{escape(json.dumps(agg.get("method_weights", {}), indent=2))}</pre>
        </div>

        <div class="section card">
        <h2>Confidence</h2>
        <pre>{escape(json.dumps({
            "confidence_score": agg.get("confidence_score"),
            "agreement_score": agg.get("agreement_score")
        }, indent=2))}</pre>
        </div>

        <div class="section card">
        <h2>Entropy</h2>
        <pre>{escape(json.dumps(entropy_data, indent=2))}</pre>
        </div>

        <div class="section card">
        <h2>Monitoring</h2>
        <pre>{escape(json.dumps(monitoring, indent=2))}</pre>
        </div>

        </body>
        </html>
        """

        with path.open("w", encoding="utf-8") as f:
            f.write(html)

        return path

    # =====================================================
    # MAIN
    # =====================================================

    def generate(
        self,
        article_id: str,
        explanation: Dict[str, Any],
        *,
        save_json: bool = True,
        save_html: bool = True,
    ) -> Dict[str, Path]:

        outputs = {}

        if save_json:
            outputs["json"] = self.save_json(article_id, explanation)

        if save_html:
            outputs["html"] = self.save_html(article_id, explanation)

        return outputs