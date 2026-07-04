"""Day 6: LSTM demand forecasting model.

Built on the Day 4-cleaned daily ``quantity`` series and the same
56-day holdout split used for Day 5's Prophet baseline.

"""

from __future__ import annotations


from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import Dataset
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))


FEATURES_DIR = Path(__file__).resolve().parents[2] / "data" / "features"
FEATURE_COLUMNS = ("y_scaled", "dow_sin", "dow_cos", "is_saturday", "is_uk_holiday")


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Attach day-of-week, Saturday, and UK holiday features to a demand series.

    Args:
        df: DataFrame with ``ds`` (date) and ``y`` (demand) columns.

    Returns:
        ``df`` with added ``dow_sin``, ``dow_cos`` (cyclic day-of-week
        encoding), ``is_saturday``, and ``is_uk_holiday`` columns.
    """
    out = df.copy()
    dow = out["ds"].dt.dayofweek
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["is_saturday"] = (dow == 5).astype(float)

    uk_holidays = holidays.UnitedKingdom(
        years=range(out["ds"].dt.year.min(), out["ds"].dt.year.max() + 1)
    )
    out["is_uk_holiday"] = out["ds"].dt.date.isin(uk_holidays).astype(float)
    return out


class DemandScaler:
    """Standardizes the target column using train-only statistics."""

    def __init__(self) -> None:
        """Create an unfitted scaler."""
        self.mean_: float | None = None
        self.std_: float | None = None

    def fit(self, y: np.ndarray) -> "DemandScaler":
        """Fit mean/std from training-only target values.

        Args:
            y: 1D array of target values from the training split only.

        Returns:
            self, fitted.
        """
        self.mean_ = float(np.mean(y))
        self.std_ = float(np.std(y)) or 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Apply the fitted standardization.

        Args:
            y: Array of target values, any split.

        Returns:
            Standardized values.
        """
        return (y - self.mean_) / self.std_

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        """Undo the standardization.

        Args:
            y_scaled: Standardized values.

        Returns:
            Values back on the original demand scale.
        """
        return y_scaled * self.std_ + self.mean_


class DemandWindowDataset(Dataset):
    """Sliding-window dataset: ``lookback`` days of features -> next day's target."""

    def __init__(
        self, features: np.ndarray, targets: np.ndarray, lookback: int, target_indices: np.ndarray
    ):
        """Build the dataset.

        Args:
            features: Array of shape (T, n_features), full series.
            targets: Array of shape (T,), the scaled target, full series.
            lookback: Number of preceding days used as input context.
            target_indices: Which time indices (into the full T-length
                arrays) belong to this split. Each index ``t`` must satisfy
                ``t >= lookback`` so a full window exists.
        """
        assert (
            target_indices >= lookback
        ).all(), "Every target index needs `lookback` days of prior context"
        self.features = features
        self.targets = targets
        self.lookback = lookback
        self.target_indices = target_indices

    def __len__(self) -> int:
        """Number of windows in this split."""
        return len(self.target_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the (window, target) pair at position idx."""
        t = self.target_indices[idx]
        x = self.features[t - self.lookback : t]
        y = self.targets[t]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class DemandLSTM(pl.LightningModule):
    """LSTM regressor: a window of daily features -> next day's scaled demand."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
    ):
        """Build the model.

        Args:
            n_features: Number of input features per timestep.
            hidden_size: LSTM hidden state size.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout between LSTM layers (ignored if num_layers=1).
            learning_rate: Adam learning rate.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the next-step scaled demand for a batch of windows.

        Args:
            x: Tensor of shape (batch, lookback, n_features).

        Returns:
            Tensor of shape (batch,), the predicted scaled target.
        """
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        return self.head(last_step).squeeze(-1)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """One training step: MSE loss on the scaled target."""
        x, y = batch
        yhat = self(x)
        loss = self.loss_fn(yhat, y)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """validation step: MSE loss on the scaled target."""
        x, y = batch
        yhat = self(x)
        loss = self.loss_fn(yhat, y)
        self.log("val_loss", loss, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Adam optimizer"""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def build_windowed_splits(
    feature_df: pd.DataFrame, lookback: int, test_days: int = 56, val_days: int = 56
) -> dict:
    """Build train/val/test windowed datasets and the fitted target scaler.

    Args:
        feature_df: Output of ``build_feature_frame``, full series.
        lookback: Number of preceding days per input window.
        test_days: Size of the held-out test tail, defaults to 56 (matches
            Day 5's Prophet evaluation exactly).
        val_days: Size of the validation slice carved from the remaining
            training data, defaults to 56 (also a clean multiple of 7).

    Returns:
        Dict with ``train_ds``, ``val_ds``, ``test_ds``, ``scaler`` , and ``test_dates``
    """
    n = len(feature_df)
    test_start = n - test_days
    val_start = test_start - val_days

    y_raw = feature_df["y"].to_numpy()
    scaler = DemandScaler().fit(y_raw[:val_start])  # fit on train-inner only, no leakage
    y_scaled = scaler.transform(y_raw)

    feature_df = feature_df.copy()
    feature_df["y_scaled"] = y_scaled
    features = feature_df[list(FEATURE_COLUMNS)].to_numpy()

    train_idx = np.arange(lookback, val_start)
    val_idx = np.arange(val_start, test_start)
    test_idx = np.arange(test_start, n)

    return {
        "train_ds": DemandWindowDataset(features, y_scaled, lookback, train_idx),
        "val_ds": DemandWindowDataset(features, y_scaled, lookback, val_idx),
        "test_ds": DemandWindowDataset(features, y_scaled, lookback, test_idx),
        "scaler": scaler,
        "test_dates": feature_df["ds"].to_numpy()[test_idx],
        "test_actuals": y_raw[test_idx],
    }


def predict_test_set(
    model: DemandLSTM, test_ds: DemandWindowDataset, scaler: DemandScaler
) -> np.ndarray:
    """Run inference on every window in the test dataset.

    Args:
        model: A trained ``DemandLSTM``.
        test_ds: The test ``DemandWindowDataset``.
        scaler: The same ``DemandScaler`` used to build the dataset's
            targets, used here to inverse-transform predictions back to
            real demand units.

    Returns:
        Array of predicted demand values (original scale, clipped at 0
        since demand can't be negative), one per test window, in order.
    """
    model.eval()
    preds_scaled = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            x, _ = test_ds[i]
            yhat = model(x.unsqueeze(0))
            preds_scaled.append(yhat.item())
    preds = scaler.inverse_transform(np.array(preds_scaled))
    return np.maximum(preds, 0.0)
