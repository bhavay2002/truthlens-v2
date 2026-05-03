"""
File: run_inference.py (FINAL UPGRADED)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np

from src.inference.inference_engine import InferenceConfig, InferenceEngine
from src.evaluation.calibration import compute_calibration
from src.evaluation.uncertainty import uncertainty_statistics
from src.evaluation.task_correlation import compute_task_correlation
from src.evaluation.report_writer import save_report
from src.inference.report_generator import ReportGenerator
from src.inference.result_formatter import ResultFormatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================
# CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--article", type=str)
    parser.add_argument("--input_file", type=str)

    parser.add_argument("--labels_file", type=str)
    parser.add_argument("--evaluate", action="store_true")

    parser.add_argument("--output_format", default="api",
                        choices=["api", "dashboard", "research"])

    # 🔥 NEW FLAGS
    parser.add_argument("--save_logits", action="store_true")
    parser.add_argument("--save_probs", action="store_true")
    parser.add_argument("--save_uncertainty", action="store_true")
    parser.add_argument("--save_dir", default="outputs")

    return parser.parse_args()


# =========================================================
# DATA
# =========================================================

def load_texts(args):
    if args.article:
        return [args.article]

    if args.input_file:
        path = Path(args.input_file)
        return [l.strip() for l in path.read_text().splitlines() if l.strip()]

    raise ValueError("Provide article or input_file")


def load_labels(path) -> Dict:
    with open(path) as f:
        return json.load(f)


# =========================================================
# SAVE HELPERS
# =========================================================

def save_arrays(outputs, save_dir, *, save_logits, save_probs):

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for task, out in outputs.items():

        if save_logits and out.get("logits") is not None:
            np.save(save_path / f"{task}_logits.npy", out["logits"])

        if save_probs and out.get("probabilities") is not None:
            np.save(save_path / f"{task}_probabilities.npy", out["probabilities"])


def save_uncertainty(uncertainty, save_dir):

    if not uncertainty:
        return

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / "uncertainty.json", "w") as f:
        json.dump(uncertainty, f, indent=4)


# =========================================================
# MAIN
# =========================================================

def main():

    args = parse_args()

    try:
        texts = load_texts(args)

        engine = InferenceEngine(
            InferenceConfig(
                model_path=args.model_dir,
                device="auto"
            )
        )

        # -------------------------------------------------
        # INFERENCE
        # -------------------------------------------------
        outputs = engine.predict_for_evaluation(texts)

        # -------------------------------------------------
        # EVALUATION
        # -------------------------------------------------
        evaluation = {}
        calibration = {}
        uncertainty = {}
        correlation = {}

        if args.evaluate and args.labels_file:

            labels = load_labels(args.labels_file)

            for task, out in outputs.items():

                # RUN-META-GUARD: InferenceEngine.predict_for_evaluation returns
                # a ``"_meta"`` scratch key alongside per-task entries. Iterating
                # without this guard would call out["logits"] on
                # {"texts": [...]} and raise KeyError on every evaluation run.
                if not isinstance(out, dict) or "logits" not in out:
                    continue

                logits = out["logits"]
                probs = out["probabilities"]

                y_true = np.asarray(labels[task])

                calibration[task] = compute_calibration(
                    logits=logits,
                    y_true=y_true,
                    task_type="multiclass",
                )

                uncertainty[task] = uncertainty_statistics(probs)

            correlation = compute_task_correlation(
                {k: v["probabilities"] for k, v in outputs.items()}
            )

        # -------------------------------------------------
        # OPTIONAL SAVE RAW OUTPUTS 🔥
        # -------------------------------------------------
        if args.save_logits or args.save_probs:
            save_arrays(
                outputs,
                args.save_dir,
                save_logits=args.save_logits,
                save_probs=args.save_probs,
            )

        if args.save_uncertainty:
            save_uncertainty(uncertainty, args.save_dir)

        # -------------------------------------------------
        # REPORT
        # -------------------------------------------------
        report_gen = ReportGenerator()

        report = report_gen.generate_report(
            article_text=" ".join(texts),
            predictions=outputs,
            evaluation=evaluation,
            calibration=calibration,
            uncertainty=uncertainty,
            task_correlation=correlation,
        )

        # -------------------------------------------------
        # FORMAT OUTPUT
        # -------------------------------------------------
        formatter = ResultFormatter()

        if args.output_format == "api":
            final = formatter.format_api_response(report)

        elif args.output_format == "dashboard":
            final = formatter.format_dashboard_report(report)

        else:
            final = formatter.format_research_export(report)

        print(json.dumps(final, indent=2))

        # -------------------------------------------------
        # SAVE REPORT
        # -------------------------------------------------
        save_report(report, Path(args.save_dir) / "report.json")

    except Exception:
        logger.exception("Inference failed")
        sys.exit(1)


if __name__ == "__main__":
    main()