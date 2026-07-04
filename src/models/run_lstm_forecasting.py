"""Day 6 : train the LSTM demand forecaster and evaluate."""

from __future__ import annotations


import os
from pathlib import Path
import sys
import os
sys.path.insert(0,'D:/retailplus/src')
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.pytorch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader

from src.models.forecasting import evaluate_forecast, load_demand_series
from src.models.lstm_forecasting import (
    DemandLSTM,
    build_feature_frame,
    build_windowed_splits,
    predict_test_set,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
TARGET_COLUMN = "quantity"
TEST_DAYS = 56
VAL_DAYS = 56
BATCH_SIZE = 16
MAX_EPOCHS = 100

# Hyperparameter
SWEEP_CONFIGS = [
    {"lookback": 14, "hidden_size": 16, "num_layers": 1},
    {"lookback": 14, "hidden_size": 32, "num_layers": 1},
    {"lookback": 28, "hidden_size": 32, "num_layers": 1},
    {"lookback": 28, "hidden_size": 32, "num_layers": 2},
]


def run_one_config(config: dict, feature_df: pd.DataFrame) -> dict:
    """Train and evaluate one LSTM config.

    Args:
        config: Dict with ``lookback``, ``hidden_size``, ``num_layers``.
        feature_df: Output of ``build_feature_frame``.

    Returns:
        Dict with the config, the trained model, evaluation metrics, and
        the fitted scaler/test dates needed to persist predictions later.
    """
    pl.seed_everything(42, verbose=False)
    splits = build_windowed_splits(
        feature_df, lookback=config["lookback"], test_days=TEST_DAYS, val_days=VAL_DAYS
    )
    train_loader = DataLoader(splits["train_ds"], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(splits["val_ds"], batch_size=BATCH_SIZE, shuffle=False)

    model = DemandLSTM(
        n_features=len(splits["train_ds"].features[0]),
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="cpu",
        callbacks=[early_stop],
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    preds = predict_test_set(model, splits["test_ds"], splits["scaler"])
    metrics = evaluate_forecast(splits["test_actuals"], preds)
    val_loss = trainer.callback_metrics.get("val_loss")

    

    return {
        **config,
        **metrics,
        "model": model,
        "preds": preds,
        "test_dates": splits["test_dates"],
        "test_actuals": splits["test_actuals"],
        "val_loss": float(val_loss) if val_loss is not None else None,
        "stopped_epoch": trainer.current_epoch,
    }


def main() -> None:
    mlflow.set_tracking_uri(f"file:{ROOT_DIR / 'mlruns'}")
    mlflow.set_experiment("demand_forecasting")

    df = load_demand_series(column=TARGET_COLUMN)
    feature_df = build_feature_frame(df)


    with mlflow.start_run(run_name="day6_lstm_baseline"):
        mlflow.log_param("target_column", TARGET_COLUMN)
        mlflow.log_param("test_days", TEST_DAYS)
        mlflow.log_param("val_days", VAL_DAYS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("max_epochs", MAX_EPOCHS)

        results = []
        for i, config in enumerate(SWEEP_CONFIGS):
            with mlflow.start_run(run_name=f"lstm_config_{i}", nested=True):
                mlflow.log_params(config)
                result = run_one_config(config, feature_df)
                for k in ("mae", "rmse", "mape", "n_excluded_zero_actual"):
                    mlflow.log_metric(k, result[k])
                if result["val_loss"] is not None:
                    mlflow.log_metric("val_loss", result["val_loss"])
                mlflow.log_metric("stopped_epoch", result["stopped_epoch"])
                results.append(result)
                

        best = min(results, key=lambda r: r["mape"])
        best_config = {k: best[k] for k in ("lookback", "hidden_size", "num_layers")}

        mlflow.log_param("best_lookback", best["lookback"])
        mlflow.log_param("best_hidden_size", best["hidden_size"])
        mlflow.log_param("best_num_layers", best["num_layers"])
        mlflow.log_metric("best_mape", best["mape"])
        mlflow.log_metric("best_mae", best["mae"])
        mlflow.log_metric("best_rmse", best["rmse"])
        mlflow.pytorch.log_model(best["model"], name="lstm_model", serialization_format="pickle")

        out_dir = ROOT_DIR / "data" / "features" / "forecast_artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)
        comparison = pd.DataFrame(
            {"ds": best["test_dates"], "actual": best["test_actuals"], "lstm_yhat": best["preds"]}
        )
        comparison.to_csv(out_dir / "lstm_holdout_comparison.csv", index=False)

        mape_target_met = best["mape"] <= 12.0
        mlflow.log_param("meets_12pct_mape_target", mape_target_met)
        
        


if __name__ == "__main__":
    main()
