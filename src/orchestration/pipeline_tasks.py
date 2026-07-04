"""Day 13: Retraining pipeline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
MONITORING_DIR = ROOT_DIR / "data" / "features" / "monitoring"
RUN_LOG_PATH = MONITORING_DIR / "pipeline_runs.jsonl"

AUC_TARGET = 0.88
PRECISION_TARGET = 0.75
MAPE_TARGET = 12.0


def check_drift_task() -> dict:
    """Run the Day 12 drift checks for both monitored models.

    Returns:
        Dict with ``churn_drift`` and ``demand_drift``, each a dict with
        ``drifted_share`` and ``retrain_recommended`` (see
        ``src.monitoring.drift_detection.recommend_retraining``).
    """
    from src.monitoring.run_drift_detection import (
        check_churn_feature_drift,
        check_demand_feature_drift,
    )

    churn_result = check_churn_feature_drift(
        reference_cutoff=pd.Timestamp("2011-06-01"), current_cutoff=pd.Timestamp("2011-09-10")
    )
    demand_result = check_demand_feature_drift(split_date=pd.Timestamp("2010-12-01"))

    return {
        "churn_drift": {
            "drifted_share": churn_result["overall"]["drifted_share"],
            "retrain_recommended": churn_result["retrain_recommended"],
        },
        "demand_drift": {
            "drifted_share": demand_result["overall"]["drifted_share"],
            "retrain_recommended": demand_result["retrain_recommended"],
        },
    }


def get_latest_run_metrics(
    experiment_name: str, run_name: str, metric_names: list[str]
) -> dict | None:
    """Fetch metrics from the most recent MLflow run matching a name.

    Args:
        experiment_name: MLflow experiment to search.
        run_name: Exact ``mlflow.runName`` tag to match.
        metric_names: Which metric columns to pull.

    Returns:
        Dict of metric name to value for the most recent matching run, or
        None if no experiment or matching run exists yet (e.g. on a fresh
        checkout before any pipeline has run).
    """
    mlflow.set_tracking_uri(f"file:{ROOT_DIR / 'mlruns'}")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        return None

    row = runs.iloc[0]
    return {name: row.get(f"metrics.{name}") for name in metric_names}


def check_constraints_task() -> dict:
    """Check the most recently logged churn and forecast metrics against project targets.

    Returns:
        Dict with ``churn`` and ``forecast`` sub-dicts, each containing the
        raw metric values and a ``meets_target`` boolean per constraint.
    """
    churn_metrics = get_latest_run_metrics(
        "churn_prediction",
        "day11_churn_tuning",
        ["best_test_auc", "best_test_precision_at_top20pct"],
    )
    forecast_metrics = get_latest_run_metrics(
        "demand_forecasting", "day8_ensemble", ["best_ensemble_mape"]
    )

    result = {"churn": None, "forecast": None}
    if churn_metrics:
        auc = churn_metrics["best_test_auc"]
        precision = churn_metrics["best_test_precision_at_top20pct"]
        result["churn"] = {
            "auc_roc": float(auc) if auc is not None else None,
            "auc_meets_target": bool(auc >= AUC_TARGET) if auc is not None else None,
            "precision_at_top20pct": float(precision) if precision is not None else None,
            "precision_meets_target": (
                bool(precision >= PRECISION_TARGET) if precision is not None else None
            ),
        }
    if forecast_metrics:
        mape = forecast_metrics["best_ensemble_mape"]
        result["forecast"] = {
            "mape": float(mape) if mape is not None else None,
            "mape_meets_target": bool(mape <= MAPE_TARGET) if mape is not None else None,
        }
    return result


def decide_retrain_targets(drift_result: dict) -> list[str]:
    """Decide which models to retrain based on the drift check.

    Args:
        drift_result: Output of ``check_drift_task``.

    Returns:
        List of model names to retrain, a subset of
        ``["churn", "forecast"]``. 
    """
    targets = []
    if drift_result["churn_drift"]["retrain_recommended"]:
        targets.append("churn")
    if drift_result["demand_drift"]["retrain_recommended"]:
        targets.append("forecast")
    return targets


def retrain_churn_task() -> dict:
    """Retrain the churn model via the Day 11 pipeline (Optuna + SHAP).

    Returns:
        Dict with the retrained model's ``auc_roc`` and
        ``precision_at_top20pct``.
    """
    subprocess.run([sys.executable, "-m", "src.models.run_churn_tuning"], cwd=ROOT_DIR, check=True)
    metrics = get_latest_run_metrics(
        "churn_prediction",
        "day11_churn_tuning",
        ["best_test_auc", "best_test_precision_at_top20pct"],
    )
    return {
        "auc_roc": metrics["best_test_auc"],
        "precision_at_top20pct": metrics["best_test_precision_at_top20pct"],
    }


def retrain_forecast_task() -> dict:
    """Retrain the forecasting ensemble via the Day 5/6/8 pipelines.

    Returns:
        Dict with the retrained ensemble's ``mape``.
    """
    for module in (
        "src.models.run_forecasting",
        "src.models.run_lstm_forecasting",
        "src.models.run_ensemble_forecasting",
    ):
        subprocess.run([sys.executable, "-m", module], cwd=ROOT_DIR, check=True)

    metrics = get_latest_run_metrics("demand_forecasting", "day8_ensemble", ["best_ensemble_mape"])
    return {"mape": metrics["best_ensemble_mape"]}


def _json_default(obj):
    """JSON encoder

    Args:
        obj: An object ``json.dumps`` doesn't know how to serialize.

    Returns:
        A JSON-serializable native Python value.
    """
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def log_pipeline_run(run_summary: dict, log_path: Path = RUN_LOG_PATH) -> Path:
    """Append one pipeline run's summary to the audit log.

    Args:
        run_summary: Arbitrary JSON-serializable dict describing what this
            run did (drift results, retrain targets, validation outcome).
        log_path: Destination JSONL file, defaults to
            ``data/features/monitoring/pipeline_runs.jsonl``.

    Returns:
        Path the entry was appended to.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **run_summary}
    with open(log_path, "a") as f:
        f.write(json.dumps(entry, default=_json_default) + "\n")
    return log_path
