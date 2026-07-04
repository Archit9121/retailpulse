"""Day 5 : fit the Prophet baseline, evaluate, log to MLflow."""

from __future__ import annotations


import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.prophet

from src.models.forecasting import (
    evaluate_forecast,
    fit_prophet,
    load_demand_series,
    naive_seasonal_forecast,
    predict,
    train_test_split_ts,
    write_forecast_artifacts,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
TARGET_COLUMN = "quantity"
TEST_DAYS = 56

# Hyperparameter
SWEEP_CONFIGS = [
    {"changepoint_prior_scale": 0.05, "seasonality_mode": "additive"},
    {"changepoint_prior_scale": 0.05, "seasonality_mode": "multiplicative"},
    {"changepoint_prior_scale": 0.5, "seasonality_mode": "additive"},
    {"changepoint_prior_scale": 0.01, "seasonality_mode": "additive"},
    {
        "changepoint_prior_scale": 0.05,
        "seasonality_mode": "additive",
        "add_saturday_regressor": True,
    },
]


def main() -> None:
    mlflow.set_tracking_uri(f"file:{ROOT_DIR / 'mlruns'}")
    mlflow.set_experiment("demand_forecasting")

    df = load_demand_series(column=TARGET_COLUMN)
    train_df, test_df = train_test_split_ts(df, test_days=TEST_DAYS)

    naive = naive_seasonal_forecast(train_df, test_df, season_length=7)
    naive_metrics = evaluate_forecast(test_df["y"], naive)

    with mlflow.start_run(run_name="day5_prophet_baseline"):
        mlflow.log_param("target_column", TARGET_COLUMN)
        mlflow.log_param("test_days", TEST_DAYS)
        mlflow.log_param("n_train_days", len(train_df))

        with mlflow.start_run(run_name="seasonal_naive_baseline", nested=True):
            mlflow.log_param("method", "seasonal_naive_lag7")
            for k, v in naive_metrics.items():
                mlflow.log_metric(k, v)

        results = []
        for i, config in enumerate(SWEEP_CONFIGS):
            with mlflow.start_run(run_name=f"prophet_config_{i}", nested=True):
                mlflow.log_params(config)
                model = fit_prophet(train_df, **config)
                forecast = predict(model, test_df)
                metrics = evaluate_forecast(test_df["y"], forecast["yhat"])
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)
                results.append({**config, **metrics, "model": model, "forecast": forecast})


        best = min(results, key=lambda r: r["mape"])
        

        mlflow.log_param("best_changepoint_prior_scale", best["changepoint_prior_scale"])
        mlflow.log_param("best_seasonality_mode", best["seasonality_mode"])
        mlflow.log_metric("best_mape", best["mape"])
        mlflow.log_metric("best_mae", best["mae"])
        mlflow.log_metric("best_rmse", best["rmse"])
        mlflow.log_metric("best_vs_naive_mape_delta", best["mape"] - naive_metrics["mape"])
        mlflow.prophet.log_model(best["model"], name="prophet_model")
        
       
        write_forecast_artifacts(test_df, best["forecast"]["yhat"], naive)

        



if __name__ == "__main__":
    main()
