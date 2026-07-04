"""Day 8 entry point: build and evaluate the Prophet + LSTM ensemble."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import pandas as pd

from src.models.ensemble_forecasting import (
    apply_day_type_weights,
    fit_day_type_weights,
    fit_global_weight,
    get_validation_predictions,
)
from src.models.forecasting import evaluate_forecast, load_demand_series
from src.models.lstm_forecasting import build_feature_frame


ROOT_DIR = Path(__file__).resolve().parents[2]
FORECAST_DIR = ROOT_DIR / "data" / "features" / "forecast_artifacts"


def main() -> None:
    mlflow.set_tracking_uri(f"file:{ROOT_DIR / 'mlruns'}")
    mlflow.set_experiment("demand_forecasting")

    df = load_demand_series(column="quantity")
    feature_df = build_feature_frame(df)

    prophet_test = pd.read_csv(FORECAST_DIR / "prophet_holdout_comparison.csv")
    lstm_test = pd.read_csv(FORECAST_DIR / "lstm_holdout_comparison.csv")
    test_df = pd.DataFrame(
        {
            "ds": prophet_test["ds"].to_numpy(),
            "actual": prophet_test["actual"].to_numpy(),
            "prophet_pred": prophet_test["prophet_yhat"].to_numpy(),
            "lstm_pred": lstm_test["lstm_yhat"].to_numpy(),
        }
    )

    with mlflow.start_run(run_name="day8_ensemble"):
        val_results = get_validation_predictions(df, feature_df)

        prophet_val_metrics = evaluate_forecast(val_results["actual"], val_results["prophet_pred"])
        lstm_val_metrics = evaluate_forecast(val_results["actual"], val_results["lstm_pred"])


        with mlflow.start_run(run_name="ensemble_global_weight", nested=True):
            best_weight, weight_sweep = fit_global_weight(val_results)
            mlflow.log_param("strategy", "global_weight")
            mlflow.log_param("lstm_weight", best_weight)
            global_test_pred = (
                best_weight * test_df["lstm_pred"] + (1 - best_weight) * test_df["prophet_pred"]
            )
            global_metrics = evaluate_forecast(test_df["actual"], global_test_pred)
            for k, v in global_metrics.items():
                mlflow.log_metric(k, v)


        with mlflow.start_run(run_name="ensemble_day_type_weights", nested=True):
            day_type_weights = fit_day_type_weights(val_results)
            mlflow.log_params({f"weight_{k}": v for k, v in day_type_weights.items()})
            day_type_test_pred = apply_day_type_weights(test_df, day_type_weights)
            day_type_metrics = evaluate_forecast(test_df["actual"], day_type_test_pred)
            for k, v in day_type_metrics.items():
                mlflow.log_metric(k, v)
    

        mlflow.log_metric("prophet_val_mape", prophet_val_metrics["mape"])
        mlflow.log_metric("lstm_val_mape", lstm_val_metrics["mape"])
        mlflow.log_metric("global_ensemble_test_mape", global_metrics["mape"])
        mlflow.log_metric("day_type_ensemble_test_mape", day_type_metrics["mape"])

        best_strategy = (
            "day_type" if day_type_metrics["mape"] < global_metrics["mape"] else "global"
        )
        best_metrics = day_type_metrics if best_strategy == "day_type" else global_metrics
        mlflow.log_param("best_strategy", best_strategy)
        mlflow.log_metric("best_ensemble_mape", best_metrics["mape"])

        out_dir = FORECAST_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        comparison = test_df.copy()
        comparison["ensemble_global"] = global_test_pred
        comparison["ensemble_day_type"] = day_type_test_pred
        comparison.to_csv(out_dir / "ensemble_holdout_comparison.csv", index=False)
        weight_sweep.to_csv(out_dir / "ensemble_weight_sweep.csv", index=False)
        val_results.to_csv(out_dir / "ensemble_validation_predictions.csv", index=False)

        mape_target_met = best_metrics["mape"] <= 12.0
        mlflow.log_param("meets_12pct_mape_target", mape_target_met)


if __name__ == "__main__":
    main()
