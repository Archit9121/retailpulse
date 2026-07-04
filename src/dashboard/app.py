"""RetailPulse dashboard: home page.
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

import streamlit as st  

from src.dashboard.data_loader import (  # noqa: E402
    AUC_TARGET,
    MAPE_TARGET,
    artifact_status,
    load_customer_segments,
    load_daily_sales_features,
    load_latest_mlflow_metrics,
)

st.set_page_config(page_title="RetailPulse", page_icon="🛒", layout="wide")

st.title("🛒 RetailPulse")
st.caption(
    "Retail analytics — demand forecasting, customer segmentation, "
    "churn risk, and inventory optimization."
)

daily = load_daily_sales_features()
segments = load_customer_segments()
metrics = load_latest_mlflow_metrics()

st.divider()
st.subheader("At a glance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if daily is not None:
        total_revenue = daily["revenue"].sum()
        st.metric("Total revenue (2 yrs)", f"£{total_revenue:,.0f}")
    else:
        st.metric("Total revenue", "—")
        st.caption("Run Day 2's pipeline")

with col2:
    if segments is not None:
        st.metric("Customers segmented", f"{len(segments):,}")
    else:
        st.metric("Customers segmented", "—")
        st.caption("Run Day 3's pipeline")

with col3:
    forecast = metrics.get("forecast")
    if forecast is not None:
        delta = forecast["mape"] - MAPE_TARGET
        st.metric(
            "Forecast MAPE",
            f"{forecast['mape']:.1f}%",
            delta=f"{delta:+.1f}pp vs {MAPE_TARGET:.0f}% target",
            delta_color="inverse",
        )
    else:
        st.metric("Forecast MAPE", "—")
        st.caption("Run Day 5-8's pipelines")

with col4:
    churn = metrics.get("churn")
    if churn is not None:
        delta = churn["auc_roc"] - AUC_TARGET
        st.metric(
            "Churn AUC-ROC",
            f"{churn['auc_roc']:.3f}",
            delta=f"{delta:+.3f} vs {AUC_TARGET:.2f} target",
        )
    else:
        st.metric("Churn AUC-ROC", "—")
        st.caption("Run Day 9-11's pipelines")

st.divider()

left, right = st.columns([2, 1])

with left:
    st.subheader("Revenue over time")
    if daily is not None:
        chart_df = daily.set_index("date")[["revenue", "revenue_roll30_mean"]].rename(
            columns={"revenue": "Daily revenue", "revenue_roll30_mean": "30-day rolling mean"}
        )
        st.line_chart(chart_df, color=["#B7D7D4", "#2A6F6F"])
    else:
        st.info("Run `python -m src.features.time_series` to generate daily sales features.")

with right:
    st.subheader("Pipeline")
    status = artifact_status()
    for name, ready in status.items():
        st.write(f"{name if ready else 'working'}")


