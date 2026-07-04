"""Day 5 baseline: Prophet model for daily demand forecasting.

Built on the cleaned, partial-day-flagged daily series from Day 4.
Targets ``quantity``.
"""

from __future__ import annotations


from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import numpy as np
import pandas as pd
from prophet import Prophet


FEATURES_DIR = Path(__file__).resolve().parents[2] / "data" / "features"
MODELS_DIR = Path(__file__).resolve().parents[2] / "data" / "features" / "forecast_artifacts"


def load_demand_series(column: str = "quantity") -> pd.DataFrame:
    """Load the Day 4 cleaned daily series in Prophet's expected format.

    Args:
        column: Which demand column to forecast, e.g. ``"quantity"``.

    Returns:
        DataFrame with columns ``ds`` (date) and ``y`` (the demand column),
        sorted by date, with the Day 4 partial boundary day excluded.
    """
    daily = pd.read_csv(FEATURES_DIR / "demand_daily_flagged.csv")
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily[~daily["is_partial_day"]].sort_values("date")
    df = daily[["date", column]].rename(columns={"date": "ds", column: "y"})
    return df.reset_index(drop=True)


def train_test_split_ts(df: pd.DataFrame, test_days: int = 56) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered DataFrame into train/test by holding out the tail.

    Args:
        df: DataFrame with a ``ds`` column, sorted ascending by date.
        test_days: Number of trailing days to hold out, defaults to 56
            (8 weeks), a multiple of 7 so the holdout always contains
            a whole number of each weekday.

    Returns:
        Tuple of (train_df, test_df).
    """
    train = df.iloc[:-test_days].copy()
    test = df.iloc[-test_days:].copy()
    return train, test


def fit_prophet(
    train_df: pd.DataFrame,
    changepoint_prior_scale: float = 0.05,
    seasonality_mode: str = "additive",
    add_uk_holidays: bool = True,
    add_saturday_regressor: bool = False,
) -> Prophet:
    """Fit a Prophet model with explicit weekly seasonality and UK holidays.

    Args:
        train_df: Training data with ``ds``/``y`` columns.
        changepoint_prior_scale: Prophet's trend flexibility parameter.
        seasonality_mode: ``"additive"`` or ``"multiplicative"``.
        add_uk_holidays: If True, add UK public holidays (Christmas,
            Boxing Day, etc.)
        add_saturday_regressor: If True, add an explicit binary
            ``is_saturday`` regressor on top of the default Fourier-based
            weekly seasonality.

    Returns:
        The fitted ``Prophet`` model.
    """
    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
    )
    if add_uk_holidays:
        model.add_country_holidays(country_name="UK")
    if add_saturday_regressor:
        train_df = train_df.copy()
        train_df["is_saturday"] = (train_df["ds"].dt.dayofweek == 5).astype(int)
        model.add_regressor("is_saturday")
    model.fit(train_df)
    return model


def predict(model: Prophet, test_df: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions for the dates in ``test_df``.

    Args:
        model: A fitted ``Prophet`` model.
        test_df: DataFrame with a ``ds`` column for the dates to predict.

    Returns:
        Prophet's full forecast DataFrame.
    """
    future = test_df[["ds"]].copy()
    if "is_saturday" in model.extra_regressors:
        future["is_saturday"] = (future["ds"].dt.dayofweek == 5).astype(int)
    return model.predict(future)


def evaluate_forecast(actual: pd.Series, predicted: pd.Series) -> dict:
    """Compute MAE, RMSE, and zero-excluding MAPE.

    Args:
        actual: True values, indexed the same as ``predicted``.
        predicted: Forecasted values.

    Returns:
        Dict with ``mae``, ``rmse``, ``mape`` (computed only over points
        where ``actual != 0``, since the formula is undefined otherwise),
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    nonzero_mask = actual != 0
    n_excluded = int((~nonzero_mask).sum())
    mape = float(np.mean(np.abs(errors[nonzero_mask] / actual[nonzero_mask])) * 100)

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "n_excluded_zero_actual": n_excluded,
        "n_total": len(actual),
    }


def naive_seasonal_forecast(
    train_df: pd.DataFrame, test_df: pd.DataFrame, season_length: int = 7
) -> pd.Series:
    """Seasonal-naive baseline: forecast each day as the value `season_length` days earlier.

    Args:
        train_df: Training data with ``ds``/``y`` columns, used as the
            source of historical values for the start of the test period.
        test_df: Dates to forecast, with ``ds``/``y`` columns.
        season_length: Lag in days, defaults to 7 (last week's same day).

    Returns:
        Series of naive forecasts aligned to ``test_df``'s row order.
    """
    full = (
        pd.concat([train_df, test_df], ignore_index=True).sort_values("ds").reset_index(drop=True)
    )
    full["naive"] = full["y"].shift(season_length)
    return full.iloc[-len(test_df) :]["naive"].reset_index(drop=True)


def write_forecast_artifacts(
    test_df: pd.DataFrame, forecast: pd.DataFrame, naive: pd.Series, out_dir: Path = MODELS_DIR
) -> Path:
    """save actual vs. Prophet vs. naive predictions for the holdout period.

    Args:
        test_df: Holdout data with ``ds``/``y``.
        forecast: Output of ``predict``.
        naive: Output of ``naive_seasonal_forecast``.
        out_dir: Destination directory.

    Returns:
        Path the comparison table was written to.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(
        {
            "ds": test_df["ds"].values,
            "actual": test_df["y"].values,
            "prophet_yhat": forecast.values,
            "naive_seasonal": naive.values,
        }
    )
    path = out_dir / "prophet_holdout_comparison.csv"
    comparison.to_csv(path, index=False)
    return path
