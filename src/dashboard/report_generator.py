"""Day 20: PDF executive summary report generator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

BRAND_COLOR = colors.HexColor("#2A6F6F")


def _styled_table(data: list[list[str]], col_widths: list[float] | None = None) -> Table:
    """Build a Table with consistent header/body styling.

    Args:
        data: Row-major table data, first row treated as the header.
        col_widths: Optional explicit column widths in points.

    Returns:
        A styled ``reportlab`` ``Table`` flowable.
    """
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BRAND_COLOR),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8F7")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def generate_report(
    output_path: Path,
    daily: pd.DataFrame | None,
    segments: pd.DataFrame | None,
    risk_scores: pd.DataFrame | None,
    inventory: pd.DataFrame | None,
    constraints: dict,
) -> Path:
    """Build the executive summary PDF.

    Args:
        output_path: Where to write the PDF.
        daily: Output of ``load_daily_sales_features``.
        segments: Output of ``load_customer_segments``.
        risk_scores: Output of ``load_customer_risk_scores``.
        inventory: Output of ``load_inventory_recommendations``.
        constraints: Output of ``check_constraints_task``.

    Returns:
        ``output_path``, for convenient chaining.
    """
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReportTitle", parent=styles["Title"], textColor=BRAND_COLOR)
    heading_style = ParagraphStyle(
        "ReportHeading", parent=styles["Heading2"], textColor=BRAND_COLOR
    )
    body_style = styles["Normal"]

    story = [
        Paragraph("RetailPulse Summary", title_style),
        Paragraph(
            f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", body_style
        ),
        Spacer(1, 0.3 * inch),
    ]

    story.append(Paragraph("At a glance", heading_style))
    kpi_rows = [["Metric", "Value"]]
    if daily is not None:
        kpi_rows.append(["Total revenue (2 yrs)", f"£{daily['revenue'].sum():,.0f}"])
    if segments is not None:
        kpi_rows.append(["Customers segmented", f"{len(segments):,}"])
    if risk_scores is not None:
        at_risk = (risk_scores["churn_probability"] >= 0.5).sum()
        kpi_rows.append(["Customers above 50% churn risk", f"{at_risk:,}"])
    story.append(_styled_table(kpi_rows, col_widths=[3 * inch, 2.5 * inch]))
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("Forecasting performance", heading_style))
    forecast = constraints.get("forecast")
    if forecast is not None:
        
        story.append(
            Paragraph(
                f"Best model (Prophet+LSTM ensemble): <b>{forecast['mape']:.1f}% MAPE</b>. "
                ,body_style,
            )
        )
    

    story.append(Paragraph("Churn prediction performance", heading_style))
    churn = constraints.get("churn")
    if churn is not None:
        story.append(
            Paragraph(
                f"AUC-ROC: <b>{churn['auc_roc']:.3f}</b> "
                f"Precision@top20%: <b>{churn['precision_at_top20pct']:.3f}</b> "
                ,body_style,
            )
        )
    story.append(Spacer(1, 0.25 * inch))

    if risk_scores is not None:
        story.append(Paragraph("Top 10 highest-risk customers", heading_style))
        top10 = risk_scores.sort_values("churn_probability", ascending=False).head(10)
        risk_rows = [["Customer ID", "Churn risk", "Recency (days)", "Monetary (£)"]]
        for _, row in top10.iterrows():
            risk_rows.append(
                [
                    str(int(row["customer_id"])),
                    f"{row['churn_probability']:.1%}",
                    f"{row['recency_days']:.0f}",
                    f"£{row['monetary']:.0f}",
                ]
            )
        story.append(
            _styled_table(risk_rows, col_widths=[1.3 * inch, 1.2 * inch, 1.3 * inch, 1.3 * inch])
        )
        story.append(Spacer(1, 0.25 * inch))

    if inventory is not None:
        story.append(Paragraph("Inventory", heading_style))
        top_inventory = inventory.sort_values("reorder_point", ascending=False).head(10)
        inv_rows = [["Stock code", "Description", "Reorder point", "EOQ"]]
        for _, row in top_inventory.iterrows():
            inv_rows.append(
                [
                    row["stock_code"],
                    str(row["description"])[:30],
                    f"{row['reorder_point']:.0f}",
                    f"{row['eoq']:.0f}",
                ]
            )
        story.append(
            _styled_table(inv_rows, col_widths=[0.9 * inch, 2.6 * inch, 1.1 * inch, 0.9 * inch])
        )

    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    doc.build(story)
    return output_path
