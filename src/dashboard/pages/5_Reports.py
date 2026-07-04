"""RetailPulse dashboard: reports and exports."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_root = Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

import streamlit as st  # noqa: E402

from src.dashboard.data_loader import (  # noqa: E402
    load_customer_risk_scores,
    load_customer_segments,
    load_daily_sales_features,
    load_forecast_comparison,
    load_inventory_recommendations,
    load_latest_mlflow_metrics,
)
from src.dashboard.report_generator import generate_report  # noqa: E402

st.set_page_config(page_title="Reports — RetailPulse", page_icon="📄", layout="wide")
st.title("📄 Reports & Exports")

daily = load_daily_sales_features()
segments = load_customer_segments()
forecast_comparison = load_forecast_comparison()
risk_scores = load_customer_risk_scores()
inventory = load_inventory_recommendations()
constraints = load_latest_mlflow_metrics()

st.subheader("Executive summary (PDF)")
st.caption(
    "Report consist of  KPIs, forecast and churn performance, top-risk customers, and inventory attention list."
)

if st.button("📄 Generate PDF report"):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        generate_report(
            Path(tmp.name),
            daily,
            segments,
            risk_scores,
            inventory,
            constraints,
        )
        pdf_bytes = Path(tmp.name).read_bytes()
    st.download_button(
        "⬇️ Download report.pdf",
        pdf_bytes,
        file_name="retailpulse_executive_summary.pdf",
        mime="application/pdf",
    )
    st.success("Report generated. Click the download button above.")

st.divider()
st.subheader("Raw data exports (CSV)")
st.caption(
    "Per-table exports are also available inline on the Customers & Churn and Inventory pages."
)

col1, col2, col3 = st.columns(3)
with col1:
    if segments is not None:
        st.download_button(
            "Customer segments (CSV)",
            segments.to_csv(index=False),
            file_name="customer_segments.csv",
            mime="text/csv",
        )
with col2:
    if risk_scores is not None:
        st.download_button(
            "Churn risk scores (CSV)",
            risk_scores.to_csv(index=False),
            file_name="churn_risk_scores.csv",
            mime="text/csv",
        )
with col3:
    if inventory is not None:
        st.download_button(
            "Inventory recommendations (CSV)",
            inventory.to_csv(index=False),
            file_name="inventory_recommendations.csv",
            mime="text/csv",
        )
