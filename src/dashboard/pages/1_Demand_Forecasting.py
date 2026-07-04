"""RetailPulse dashboard: demand forecasting 
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

_root = Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

import mlflow  # noqa: E402
import mlflow.prophet  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.dashboard.data_loader import (  # noqa: E402
    ROOT_DIR,
    load_daily_sales_features,
    load_forecast_comparison,
)

st.set_page_config(page_title="Demand Forecasting — RetailPulse", page_icon="📈", layout="wide")
st.title("📈 Demand Forecasting")

daily = load_daily_sales_features()
comparison = load_forecast_comparison()

st.subheader("Historical demand")
if daily is not None:
    date_range = st.slider(
        "Date range",
        min_value=daily["date"].min().to_pydatetime(),
        max_value=daily["date"].max().to_pydatetime(),
        value=(daily["date"].min().to_pydatetime(), daily["date"].max().to_pydatetime()),
        format="YYYY-MM-DD",
    )
    filtered = daily[(daily["date"] >= date_range[0]) & (daily["date"] <= date_range[1])]
    st.line_chart(
        filtered.set_index("date")[["revenue", "revenue_roll30_mean"]].rename(
            columns={"revenue": "Daily revenue", "revenue_roll30_mean": "30-day rolling mean"}
        ),
        color=["#B7D7D4", "#2A6F6F"],
    )
else:
    st.info("Run `python -m src.features.time_series` to generate daily sales features.")

st.divider()
st.subheader("Comparison: actual vs. naive / Prophet / LSTM / ensemble")
if comparison is not None:
    chart_cols = st.multiselect(
        "Show",
        ["actual", "prophet_pred", "lstm_pred", "ensemble_global", "ensemble_day_type"],
        default=["actual", "ensemble_day_type"],
    )
    if chart_cols:
        st.line_chart(comparison.set_index("ds")[chart_cols])

    naive_metrics = {"mape": 24.58, "mae": 5610.0, "rmse": 7711.7}

    def _eval(col: str) -> dict:
        err = comparison["actual"] - comparison[col]
        nonzero = comparison["actual"] != 0
        mape = (err[nonzero] / comparison.loc[nonzero, "actual"]).abs().mean() * 100
        return {"mape": mape, "mae": err.abs().mean(), "rmse": (err**2).mean() ** 0.5}

    metrics_table = pd.DataFrame(
        {
            "seasonal_naive": naive_metrics,
            "prophet": _eval("prophet_pred"),
            "lstm": _eval("lstm_pred"),
            "ensemble (global weight)": _eval("ensemble_global"),
            "ensemble (day-type weight)": _eval("ensemble_day_type"),
        }
    ).T
    metrics_table.columns = ["MAPE (%)", "MAE", "RMSE"]
    st.dataframe(
        metrics_table.style.format({"MAPE (%)": "{:.1f}", "MAE": "{:.0f}", "RMSE": "{:.0f}"})
    )
    st.caption(
        "Best result: day-type-weighted ensemble at 21.3% MAPE. "
    )
else:
    st.info("Run Day 5/6/8's forecasting pipelines to generate the comparison.")

st.divider()
st.subheader("Future prediction")


@st.cache_resource
def _load_prophet_model():
    mlflow.set_tracking_uri(f"file:{ROOT_DIR / 'mlruns'}")
    experiment = mlflow.get_experiment_by_name("demand_forecasting")
    if experiment is None:
        return None
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'day5_prophet_baseline'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        return None
    return mlflow.prophet.load_model(f"runs:/{runs.iloc[0]['run_id']}/prophet_model")


model = _load_prophet_model()
if model is None:
    st.info("Run `python -m src.models.run_forecasting` to train")
else:
    col1, col2 = st.columns(2)
    with col1:
        horizon_days = st.slider(
            "Days past the end of the historical data (2011-12-09)", 7, 180, 30
        )
    with col2:
        adjustment_pct = st.slider(
            "Hypothetical demand adjustment", -30, 50, 0, format="%d%%"
        )

    future = pd.DataFrame({"ds": pd.date_range("2011-12-10", periods=horizon_days)})
    forecast = model.predict(future)
    n_negative = int((forecast["yhat"] < 0).sum())

    baseline = forecast["yhat"].clip(lower=0)
    adjusted = baseline * (1 + adjustment_pct / 100)

    plot_df = pd.DataFrame(
        {"ds": future["ds"], "Baseline forecast": baseline, "Adjusted": adjusted}
    ).set_index("ds")
    st.line_chart(plot_df, color=["#B7D7D4", "#2A6F6F"])



    total_baseline = baseline.sum()
    total_adjusted = adjusted.sum()
    st.metric(
        f"Projected total demand over {horizon_days} days",
        f"{total_adjusted:,.0f} units",
        delta=f"{total_adjusted - total_baseline:+,.0f} vs. unadjusted baseline",
    )
    
