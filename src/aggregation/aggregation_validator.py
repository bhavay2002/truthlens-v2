# src/aggregation/aggregation_validator.py

import numpy as np
from typing import Dict, Any


class AggregationValidator:

    def validate(self, result: Dict[str, Any]) -> Dict[str, Any]:

        issues = []

        scores = result.get("scores", {})

        # -------------------------
        # range checks
        # -------------------------
        for k, v in scores.items():
            if isinstance(v, (int, float)):
                if not (0.0 <= v <= 1.0):
                    issues.append(f"{k} out of range")

        # -------------------------
        # logical consistency
        # -------------------------
        if "credibility_score" in scores and "manipulation_risk" in scores:
            if scores["credibility_score"] > 0.8 and scores["manipulation_risk"] > 0.8:
                issues.append("High credibility + high manipulation conflict")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
        }