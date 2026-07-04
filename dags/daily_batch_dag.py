"""Daily batch DAG: drift detection + constraint monitoring."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from src.orchestration.pipeline_tasks import (
    check_constraints_task,
    check_drift_task,
    decide_retrain_targets,
    log_pipeline_run,
)


@dag(
    dag_id="retailpulse_daily_batch",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 1},
    tags=["retailpulse", "monitoring"],
)
def retailpulse_daily_batch():
    """Daily drift check + constraint monitoring, triggers retraining when needed."""

    @task
    def check_drift() -> dict:
        """Run the Day 12 drift checks for churn and demand features."""
        return check_drift_task()

    @task
    def check_constraints() -> dict:
        """Look up the most recently logged churn/forecast metrics against targets."""
        return check_constraints_task()

    @task
    def decide_targets(drift_result: dict) -> list[str]:
        """Decide which models, if any, need retraining."""
        targets = decide_retrain_targets(drift_result)
        return targets

    @task
    def log_summary(
        drift_result: dict, constraints_result: dict, retrain_targets: list[str]
    ) -> None:
        """Append this run's results to the audit log."""
        log_pipeline_run(
            {
                "dag": "retailpulse_daily_batch",
                "drift": drift_result,
                "constraints": constraints_result,
                "retrain_targets": retrain_targets,
            }
        )

    drift_result = check_drift()
    constraints_result = check_constraints()
    retrain_targets = decide_targets(drift_result)
    summary = log_summary(drift_result, constraints_result, retrain_targets)

    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining_if_needed",
        trigger_dag_id="retailpulse_model_retraining",
        conf={"retrain_targets": "{{ ti.xcom_pull(task_ids='decide_targets') }}"},
        trigger_rule="all_done",
    )

    summary >> trigger_retraining


retailpulse_daily_batch()
