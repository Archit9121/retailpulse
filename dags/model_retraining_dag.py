"""Model retraining DAG"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airflow.decorators import dag, task
from airflow.models.param import Param

from src.orchestration.pipeline_tasks import (
    check_constraints_task,
    log_pipeline_run,
    retrain_churn_task,
    retrain_forecast_task,
)


@dag(
    dag_id="retailpulse_model_retraining",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 0},
    tags=["retailpulse", "retraining"],
    params={"retrain_targets": Param(default=[], type="array")},
)
def retailpulse_model_retraining():
    """Conditionally retrain the churn and/or forecasting models, then validate."""

    @task
    def retrain_churn_if_targeted(params: dict | None = None) -> dict | None:
        """Retrain the churn model only if it's in this run's target list.

        ``params`` is auto-injected by Airflow's TaskFlow API from the DAG
        run's runtime parameters 
        """
        retrain_targets = (params or {}).get("retrain_targets", [])
        if "churn" not in retrain_targets:
            return None
        return retrain_churn_task()

    @task
    def retrain_forecast_if_targeted(params: dict | None = None) -> dict | None:
        """Retrain the forecasting ensemble only if it's in this run's target list."""
        retrain_targets = (params or {}).get("retrain_targets", [])
        if "forecast" not in retrain_targets:
            return None
        return retrain_forecast_task()

    @task
    def validate_and_log(churn_result: dict | None, forecast_result: dict | None) -> dict:
        """Check refreshed metrics against project constraints and log the outcome."""
        constraints = check_constraints_task()
        summary = {
            "dag": "retailpulse_model_retraining",
            "retrained_churn": churn_result is not None,
            "retrained_forecast": forecast_result is not None,
            "constraints_after_retraining": constraints,
        }
        log_pipeline_run(summary)
        return summary

    churn_result = retrain_churn_if_targeted()
    forecast_result = retrain_forecast_if_targeted()
    validate_and_log(churn_result, forecast_result)


retailpulse_model_retraining()
