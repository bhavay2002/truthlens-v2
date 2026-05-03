from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors

logger = logging.getLogger(__name__)


# =========================================================
# UTILS
# =========================================================

def flatten_nested(data, parent_key="", sep="."):
    items = []

    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(flatten_nested(v, new_key, sep))
    elif isinstance(data, list):
        items.append((parent_key, str(data[:10]) + ("..." if len(data) > 10 else "")))
    else:
        items.append((parent_key, data))

    return items


def dict_to_table(data: Dict[str, Any]):

    rows = [["Metric", "Value"]]

    flat = flatten_nested(data)

    for k, v in flat:
        rows.append([k, str(v)])

    table = Table(rows, colWidths=[3 * inch, 3 * inch])

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ])
    )

    return table


# =========================================================
# TASK SECTION
# =========================================================

def render_tasks(elements, tasks, styles):

    elements.append(Paragraph("Task Performance", styles["Heading1"]))
    elements.append(Spacer(1, 12))

    for task, data in tasks.items():

        elements.append(Paragraph(f"Task: {task}", styles["Heading2"]))
        elements.append(Spacer(1, 6))

        # -------------------------
        # METRICS
        # -------------------------
        metrics = data.get("metrics", {})
        elements.append(Paragraph("Metrics", styles["Heading3"]))
        elements.append(dict_to_table(metrics))
        elements.append(Spacer(1, 8))

        # -------------------------
        # DATASET STATS
        # -------------------------
        stats = data.get("dataset_stats", {})
        if stats:
            elements.append(Paragraph("Dataset Statistics", styles["Heading3"]))
            elements.append(dict_to_table(stats))
            elements.append(Spacer(1, 8))

        elements.append(PageBreak())


# =========================================================
# GENERIC SECTION
# =========================================================

def render_section(elements, title, data, styles):

    if not data:
        return

    elements.append(Paragraph(title, styles["Heading1"]))
    elements.append(Spacer(1, 10))

    elements.append(dict_to_table(data))
    elements.append(Spacer(1, 12))


# =========================================================
# MAIN
# =========================================================

def generate_pdf_report(report: Dict[str, Any], output_path):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    elements = []

    # =====================================================
    # TITLE
    # =====================================================
    elements.append(Paragraph("TruthLens AI Evaluation Report", styles["Title"]))
    elements.append(Spacer(1, 20))

    # =====================================================
    # TASKS
    # =====================================================
    render_tasks(elements, report.get("tasks", {}), styles)

    # =====================================================
    # SUMMARY
    # =====================================================
    render_section(
        elements,
        "Overall Summary",
        report.get("summary"),
        styles,
    )

    # =====================================================
    # CALIBRATION
    # =====================================================
    render_section(
        elements,
        "Calibration",
        report.get("calibration"),
        styles,
    )

    # =====================================================
    #  NEW: ERROR ANALYSIS
    # =====================================================
    render_section(
        elements,
        "Error Analysis",
        report.get("error_analysis"),
        styles,
    )

    # =====================================================
    # NEW: THRESHOLDS
    # =====================================================
    render_section(
        elements,
        "Optimal Thresholds",
        report.get("optimal_thresholds"),
        styles,
    )

    # =====================================================
    #  NEW: GRAPH
    # =====================================================
    render_section(
        elements,
        "Graph Features",
        report.get("graph"),
        styles,
    )

    # =====================================================
    #  NEW: GRAPH EXPLANATION
    # =====================================================
    render_section(
        elements,
        "Graph Explanation",
        report.get("graph_explanation"),
        styles,
    )

    # =====================================================
    #  NEW: DRIFT
    # =====================================================
    render_section(
        elements,
        "Drift Detection",
        report.get("drift"),
        styles,
    )

    # =====================================================
    #  NEW: MONITORING
    # =====================================================
    render_section(
        elements,
        "Monitoring",
        report.get("monitoring"),
        styles,
    )

    # =====================================================
    # UNCERTAINTY
    # =====================================================
    render_section(
        elements,
        "Uncertainty",
        report.get("uncertainty"),
        styles,
    )

    # =====================================================
    # TASK CORRELATION
    # =====================================================
    render_section(
        elements,
        "Task Correlation",
        report.get("task_correlation"),
        styles,
    )

    # =====================================================
    # ADVANCED ANALYSIS
    # =====================================================
    render_section(
        elements,
        "Advanced Analysis",
        report.get("advanced_analysis"),
        styles,
    )

    # =====================================================
    # BUILD PDF
    # =====================================================
    doc = SimpleDocTemplate(str(output_path))
    doc.build(elements)

    logger.info(f"PDF report generated: {output_path}")

    return output_path