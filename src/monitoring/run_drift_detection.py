"""Day 12 entry point: run drift detection for churn and demand features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.churn_labeling import build_churn_features
from src.monitoring.drift_detection import (
    MONITORING_DIR,
    extract_drift_summary,
    recommend_retraining,
    run_drift_report,
    save_drift_report,
    summarize_drift,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FEATURES_DIR = ROOT_DIR / "data" / "features"

CHURN_FEATURE_COLUMNS = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_order_value",
    "tenure_days",
    "monetary_per_day",
    "n_distinct_products",
]
DEMAND_FEATURE_COLUMNS = ["revenue", "quantity", "n_invoices", "n_customers"]


def check_churn_feature_drift(reference_cutoff: pd.Timestamp, current_cutoff: pd.Timestamp) -> dict:
    """Compare churn features computed at two different cutoffs.

    Args:
        reference_cutoff: The earlier (baseline) cutoff date.
        current_cutoff: The later cutoff date to check for drift against
            the baseline.

    Returns:
        Dict with the drift summary, per-column table, and retraining
        recommendation.
    """
    raw = pd.read_csv(PROCESSED_DIR / "customer_sales.csv")
    raw["invoice_date"] = pd.to_datetime(raw["invoice_date"])

    reference = build_churn_features(raw, cutoff=reference_cutoff)
    current = build_churn_features(raw, cutoff=current_cutoff)

    snapshot = run_drift_report(reference, current, CHURN_FEATURE_COLUMNS)
    summary = extract_drift_summary(snapshot)
    overall = summarize_drift(snapshot)
    save_drift_report(snapshot, "churn_features_drift")

    return {
        "summary": summary,
        "overall": overall,
        "retrain_recommended": recommend_retraining(overall),
    }


def check_demand_feature_drift(split_date: pd.Timestamp) -> dict:
    """Compare daily demand features before and after a split date.

    Args:
        split_date: Date dividing the "reference" and "current" periods.

    Returns:
        Dict with the drift summary, per-column table, and retraining
        recommendation.
    """
    daily = pd.read_csv(FEATURES_DIR / "demand_daily_flagged.csv")
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily[~daily["is_partial_day"]]

    reference = daily[daily["date"] < split_date]
    current = daily[daily["date"] >= split_date]

    snapshot = run_drift_report(reference, current, DEMAND_FEATURE_COLUMNS)
    summary = extract_drift_summary(snapshot)
    overall = summarize_drift(snapshot)
    save_drift_report(snapshot, "demand_features_drift")

    return {
        "summary": summary,
        "overall": overall,
        "retrain_recommended": recommend_retraining(overall),
    }


def main() -> None:
    churn_result = check_churn_feature_drift(
        reference_cutoff=pd.Timestamp("2011-06-01"), current_cutoff=pd.Timestamp("2011-09-10")
    )
    demand_result = check_demand_feature_drift(split_date=pd.Timestamp("2010-12-01"))

    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    churn_result["summary"].to_csv(MONITORING_DIR / "churn_drift_summary.csv", index=False)
    demand_result["summary"].to_csv(
        MONITORING_DIR / "demand_drift_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
