"""
Run EDA Analysis on Merged Dataset
This script loads, merges, and analyzes the fake news dataset
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.merge_datasets import merge_datasets
from src.data.eda import FakeNewsEDA
from src.utils.logging_utils import configure_logging
from src.visualization.visualize import plot_feature_importance
import logging

configure_logging()
logger = logging.getLogger(__name__)


def save_eda_report(
    eda,
    data_sources: dict,
    output_path: Path = Path("reports/eda_report.json"),
):
    """Save EDA summary report to JSON."""
    df = eda.df.copy()

    if "text_length" not in df.columns and "text" in df.columns:
        df["text_length"] = df["text"].astype(str).str.len()
    if "word_count" not in df.columns and "text" in df.columns:
        df["word_count"] = df["text"].astype(str).str.split().str.len()

    report = {
        "data_files": data_sources,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "label_distribution": {
            str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()
        } if "label" in df.columns else {},
        "text_length_stats": {
            "mean": float(df["text_length"].mean()),
            "median": float(df["text_length"].median()),
            "min": int(df["text_length"].min()),
            "max": int(df["text_length"].max()),
        } if "text_length" in df.columns and len(df) else {},
        "word_count_stats": {
            "mean": float(df["word_count"].mean()),
            "median": float(df["word_count"].median()),
            "min": int(df["word_count"].min()),
            "max": int(df["word_count"].max()),
        } if "word_count" in df.columns and len(df) else {},
        "figures_dir": str(Path("reports/figures")),
    }

    output_file = output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return output_file


def main():
    """Run full EDA on the dataset"""

    # Load and merge data
    logger.info("Loading datasets...")
    df = merge_datasets()
    logger.info(f"Total samples: {len(df)}")

    data_sources = {
        "isot_fake": str(Path("data/raw/isot/Fake.csv")),
        "isot_true": str(Path("data/raw/isot/True.csv")),
        "liar_train": str(Path("data/raw/liar_dataset/train.tsv")),
        "fakenewsnet_root": str(Path("data/raw/FakeNewsNet")),
    }
    
    # Run EDA
    logger.info("Starting EDA analysis...")
    eda = FakeNewsEDA(df)
    eda.run()

    if "label" in eda.df.columns and len(eda.df):
        label_counts = eda.df["label"].value_counts().to_dict()
        plot_feature_importance(
            features=[str(k) for k in label_counts.keys()],
            scores=[float(v) for v in label_counts.values()],
            top_k=len(label_counts),
            save_path=Path("reports/figures/label_distribution_visualize.png"),
        )

    report_path = save_eda_report(eda, data_sources)
    
    logger.info("\n" + "=" * 70)
    logger.info("EDA Complete! Check reports/figures/ for visualizations")
    logger.info(f"EDA report saved to: {report_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
