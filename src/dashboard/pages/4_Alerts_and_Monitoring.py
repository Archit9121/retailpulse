"""RetailPulse dashboard: alerts and monitoring."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

_root = Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.dashboard.data_loader import (  # noqa: E402
    AUC_TARGET,
    MAPE_TARGET,
    PRECISION_TARGET,
    ROOT_DIR,
    load_drift_summary,
    load_pipeline_run_log,
)
from src.orchestration.pipeline_tasks import check_drift_task  # noqa: E402

st.set_page_config(page_title="Alerts & Monitoring — RetailPulse", page_icon="🔔", layout="wide")
st.title("🔔 Alerts & Monitoring")


if st.button("🔄 Refresh now"):
    st.cache_data.clear()

st.caption(f"Last checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

drift = check_drift_task()



st.divider()
st.subheader("Drift status")
col1, col2 = st.columns(2)
with col1:
    churn_drift = drift["churn_drift"]
    label = (
        "Retraining recommended"
        if churn_drift["retrain_recommended"]
        else "No retraining needed"
    )
    st.metric("Churn feature drift", f"{churn_drift['drifted_share']:.0%} of columns", label)
with col2:
    demand_drift = drift["demand_drift"]
    label = (
        "Retraining recommended"
        if demand_drift["retrain_recommended"]
        else "No retraining needed"
    )
    st.metric("Demand feature drift", f"{demand_drift['drifted_share']:.0%} of columns", label)

churn_detail = load_drift_summary("churn")
demand_detail = load_drift_summary("demand")
with st.expander("Drift detail by column"):
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Churn features")
        if churn_detail is not None:
            st.dataframe(churn_detail)
    with col2:
        st.caption("Demand features")
        if demand_detail is not None:
            st.dataframe(demand_detail)

st.divider()
st.subheader("Model evolution across the project")


@st.cache_data
def _load_model_evolution() -> tuple[pd.DataFrame, pd.DataFrame]:
    mlflow.set_tracking_uri(f"file:{ROOT_DIR / 'mlruns'}")

    churn_exp = mlflow.get_experiment_by_name("churn_prediction")
    churn_evolution = pd.DataFrame(columns=["stage", "auc_roc"])
    if churn_exp is not None:
        try:
            churn_runs = mlflow.search_runs(
                experiment_ids=[churn_exp.experiment_id], order_by=["start_time ASC"]
            )
            day9 = churn_runs[churn_runs["tags.mlflow.runName"] == "day9_churn_xgboost"][
                "metrics.best_auc_roc"
            ].dropna()
            day11 = churn_runs[churn_runs["tags.mlflow.runName"] == "day11_churn_tuning"][
                "metrics.best_test_auc"
            ].dropna()
            if not day9.empty and not day11.empty:
                churn_evolution = pd.DataFrame(
                    {
                        "stage": ["manual sweep", "Optuna + features"],
                        "auc_roc": [day9.iloc[-1], day11.iloc[-1]],
                    }
                )
        except Exception:
            pass

    forecast_exp = mlflow.get_experiment_by_name("demand_forecasting")
    forecast_evolution = pd.DataFrame(columns=["stage", "mape"])
    if forecast_exp is not None:
        try:
            forecast_runs = mlflow.search_runs(
                experiment_ids=[forecast_exp.experiment_id], order_by=["start_time ASC"]
            )
            day5 = forecast_runs[
                forecast_runs["tags.mlflow.runName"] == "day5_prophet_baseline"
            ]["metrics.best_mape"].dropna()
            day6 = forecast_runs[forecast_runs["tags.mlflow.runName"] == "day6_lstm_baseline"][
                "metrics.best_mape"
            ].dropna()
            day8 = forecast_runs[forecast_runs["tags.mlflow.runName"] == "day8_ensemble"][
                "metrics.best_ensemble_mape"
            ].dropna()
            if not day5.empty and not day6.empty and not day8.empty:
                forecast_evolution = pd.DataFrame(
                    {
                        "stage": ["Prophet", "LSTM", "Ensemble"],
                        "mape": [day5.iloc[-1], day6.iloc[-1], day8.iloc[-1]],
                    }
                )
        except Exception:
            pass

    return churn_evolution, forecast_evolution


churn_evolution, forecast_evolution = _load_model_evolution()
col1, col2 = st.columns(2)
with col1:
    if churn_evolution.empty:
        st.info("No churn model evolution history found yet.")
    else:
        st.bar_chart(churn_evolution.set_index("stage"), color="#2A6F6F")
    
with col2:
    if forecast_evolution.empty:
        st.info("No demand model evolution history found yet.")
    else:
        st.bar_chart(forecast_evolution.set_index("stage"), color="#C44E52")

st.divider()

