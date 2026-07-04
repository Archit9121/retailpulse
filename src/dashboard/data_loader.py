"""Shared data loading utilities for the RetailPulse dashboard."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FEATURES_DIR = ROOT_DIR / "data" / "features"
FORECAST_DIR = FEATURES_DIR / "forecast_artifacts"
CHURN_DIR = FEATURES_DIR / "churn_artifacts"
MONITORING_DIR = FEATURES_DIR / "monitoring"

AUC_TARGET = 0.88
PRECISION_TARGET = 0.75
MAPE_TARGET = 12.0


def _load_csv_safe(path: Path) -> pd.DataFrame | None:
    """Load a csv file, returning None instead of raising if it's missing."""
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_daily_sales_features() -> pd.DataFrame | None:
    """Day 2's daily revenue/quantity/rolling-stats table."""
    df = _load_csv_safe(FEATURES_DIR / "daily_sales_features.csv")
    if df is not None:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_forecast_comparison() -> pd.DataFrame | None:
    """Day 8's ensemble holdout comparison (actual vs. Prophet/LSTM/ensemble)."""
    df = _load_csv_safe(FORECAST_DIR / "ensemble_holdout_comparison.csv")
    if df is not None:
        df["ds"] = pd.to_datetime(df["ds"])
    return df


@st.cache_data
def load_customer_segments() -> pd.DataFrame | None:
    """Day 3's K-Means customer segments with RFM values."""
    return _load_csv_safe(FEATURES_DIR / "customer_segments.csv")


@st.cache_data
def load_customer_risk_scores() -> pd.DataFrame | None:
    """Day 17's full-customer-base churn risk scores (Day 11's best tuned model)."""
    return _load_csv_safe(CHURN_DIR / "customer_risk_scores.csv")


@st.cache_data
def load_shap_importance() -> pd.DataFrame | None:
    """Day 11's SHAP feature importance for the tuned churn model."""
    return _load_csv_safe(CHURN_DIR / "shap_feature_importance_tuned.csv")


@st.cache_data
def load_inventory_recommendations() -> pd.DataFrame | None:
    """Day 10's ABC/safety-stock/reorder-point/EOQ recommendations."""
    return _load_csv_safe(FEATURES_DIR / "inventory_recommendations.csv")


@st.cache_data
def load_drift_summary(name: str) -> pd.DataFrame | None:
    """Day 12's drift summaries.

    Args:
        name: ``"churn"`` or ``"demand"``.
    """
    return _load_csv_safe(MONITORING_DIR / f"{name}_drift_summary.csv")


def load_pipeline_run_log() -> list[dict]:
    """Day 13's JSONL audit log of daily batch / retraining pipeline runs.

    Returns:
        List of run-summary dicts, most recent last. Empty list if no
        pipeline has run yet.
    """
    path = MONITORING_DIR / "pipeline_runs.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@st.cache_data
def load_latest_mlflow_metrics() -> dict:
    """The most recently logged churn and forecast metrics, checked against project targets.

    Returns:
        Dict with ``churn`` and ``forecast`` sub-dicts, mirroring
        ``src.orchestration.pipeline_tasks.check_constraints_task``.
    """
    from src.orchestration.pipeline_tasks import check_constraints_task

    return check_constraints_task()


def artifact_status() -> dict[str, bool]:
    """Check which pipeline outputs exist yet.

    Returns:
        Dict mapping a human-readable artifact name to whether its file
        exists on disk.
    """
    checks = {
        "Cleaned transactions": PROCESSED_DIR / "completed_sales.csv",
        "Customer segments": FEATURES_DIR / "customer_segments.csv",
        "Demand forecast ensemble": FORECAST_DIR / "ensemble_holdout_comparison.csv",
        "Churn predictions": CHURN_DIR / "shap_feature_importance_tuned.csv",
        "Inventory recommendations": FEATURES_DIR / "inventory_recommendations.csv",
        "Drift reports": MONITORING_DIR / "churn_drift_summary.csv",
    }
    return {name: path.exists() for name, path in checks.items()}


