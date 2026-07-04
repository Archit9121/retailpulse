"""Day 8: hybrid Prophet + LSTM ensemble for demand forecasting.

Combines Day 5's Prophet baseline and Day 6's LSTM, both trained on the
identical 56-day split. The ensemble weight is selected on a
validation window.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.models.forecasting import evaluate_forecast, fit_prophet, predict
from src.models.lstm_forecasting import (
    DemandLSTM,
    build_windowed_splits,
    predict_test_set,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
TEST_DAYS = 56
VAL_DAYS = 56

BEST_PROPHET_CONFIG = {"changepoint_prior_scale": 0.05, "seasonality_mode": "additive"}
BEST_LSTM_CONFIG = {"lookback": 14, "hidden_size": 16, "num_layers": 1}


def get_validation_predictions(df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    """Produce out-of-sample Prophet and LSTM predictions on the validation window.

    Args:
        df: Output of ``load_demand_series`` (``ds``/``y`` columns).
        feature_df: Output of ``build_feature_frame`` on the same series.

    Returns:
        DataFrame with ``ds``, ``actual``, ``prophet_pred``, ``lstm_pred``
        for the validation window.
    """
    n = len(df)
    test_start = n - TEST_DAYS
    val_start = test_start - VAL_DAYS

    train_inner = df.iloc[:val_start]
    val_df = df.iloc[val_start:test_start]

    prophet_model = fit_prophet(train_inner, **BEST_PROPHET_CONFIG)
    prophet_val_forecast = predict(prophet_model, val_df)
    prophet_val_preds = prophet_val_forecast["yhat"].to_numpy()

    pl.seed_everything(42, verbose=False)
    splits = build_windowed_splits(
        feature_df, lookback=BEST_LSTM_CONFIG["lookback"], test_days=TEST_DAYS, val_days=VAL_DAYS
    )
    train_loader = DataLoader(splits["train_ds"], batch_size=16, shuffle=True)
    val_loader = DataLoader(splits["val_ds"], batch_size=16, shuffle=False)
    lstm_model = DemandLSTM(
        n_features=len(splits["train_ds"].features[0]),
        hidden_size=BEST_LSTM_CONFIG["hidden_size"],
        num_layers=BEST_LSTM_CONFIG["num_layers"],
    )
    from pytorch_lightning.callbacks import EarlyStopping

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="cpu",
        callbacks=[EarlyStopping(monitor="val_loss", patience=10, mode="min")],
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    trainer.fit(lstm_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    lstm_val_preds = predict_test_set(lstm_model, splits["val_ds"], splits["scaler"])

    return pd.DataFrame(
        {
            "ds": val_df["ds"].to_numpy(),
            "actual": val_df["y"].to_numpy(),
            "prophet_pred": prophet_val_preds,
            "lstm_pred": lstm_val_preds,
        }
    )


def fit_global_weight(
    val_results: pd.DataFrame, weight_grid: np.ndarray = np.arange(0, 1.01, 0.05)
) -> tuple[float, pd.DataFrame]:
    """Grid-search a single ensemble weight minimizing validation MAPE.

    Args:
        val_results: Output of ``get_validation_predictions``.
        weight_grid: Candidate LSTM weights to try, 0 = pure Prophet, 1 =
            pure LSTM. Defaults to 0.0-1.0 in steps of 0.05.

    Returns:
        Tuple of (best weight, DataFrame of weight -> validation MAPE for
        every grid point, for inspection/plotting).
    """
    rows = []
    for w in weight_grid:
        combined = w * val_results["lstm_pred"] + (1 - w) * val_results["prophet_pred"]
        metrics = evaluate_forecast(val_results["actual"], combined)
        rows.append({"weight": w, "mape": metrics["mape"]})
    sweep = pd.DataFrame(rows)
    best_weight = float(sweep.loc[sweep["mape"].idxmin(), "weight"])
    return best_weight, sweep


def fit_day_type_weights(
    val_results: pd.DataFrame, weight_grid: np.ndarray = np.arange(0, 1.01, 0.05)
) -> dict:
    """Grid-search separate ensemble weights for weekday, Saturday, and Sunday.

    Args:
        val_results: Output of ``get_validation_predictions``.
        weight_grid: Candidate LSTM weights to try per bucket.

    Returns:
        Dict with keys ``"weekday"``, ``"saturday"``, ``"sunday"``, each
        mapping to the best weight for that bucket. Saturday is selected by
        MAE (its actual value is always 0 in this data, so MAPE is
        undefined there), the other two by MAPE.
    """
    dow = pd.to_datetime(val_results["ds"]).dt.dayofweek
    buckets = {
        "weekday": val_results[dow < 5],
        "saturday": val_results[dow == 5],
        "sunday": val_results[dow == 6],
    }
    weights = {}
    for name, bucket in buckets.items():
        if bucket.empty:
            weights[name] = 0.5
            continue
        scores = []
        for w in weight_grid:
            combined = w * bucket["lstm_pred"] + (1 - w) * bucket["prophet_pred"]
            metrics = evaluate_forecast(bucket["actual"], combined)
            score = metrics["mae"] if name == "saturday" else metrics["mape"]
            scores.append(score)
        weights[name] = float(weight_grid[int(np.argmin(scores))])
    return weights


def apply_day_type_weights(test_df: pd.DataFrame, weights: dict) -> np.ndarray:
    """Combine test-set Prophet/LSTM predictions using per-day-type weights.

    Args:
        test_df: DataFrame with ``ds``, ``prophet_pred``, ``lstm_pred``.
        weights: Output of ``fit_day_type_weights``.

    Returns:
        Array of combined predictions, one per row of ``test_df``.
    """
    dow = pd.to_datetime(test_df["ds"]).dt.dayofweek
    w = np.where(
        dow == 5, weights["saturday"], np.where(dow == 6, weights["sunday"], weights["weekday"])
    )
    return w * test_df["lstm_pred"].to_numpy() + (1 - w) * test_df["prophet_pred"].to_numpy()
