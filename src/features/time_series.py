"""Daily aggregate and rolling time-series features.

Builds a daily store-level demand table from ``completed_sales`` 
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


FEATURES_DIR = Path(__file__).resolve().parents[2] / "data" / "features"
ROLLING_WINDOWS = (7, 14, 30)


def build_daily_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate completed sales daily.

    Args:
        df: ``completed_sales`` DataFrame with ``invoice_date``,
            ``quantity``, ``total``, ``invoice``, and ``customer_id``.

    Returns:
        DataFrame indexed by date with columns ``revenue``, ``quantity``,
        ``n_invoices``, ``n_customers``.Days with zero
        transactions are filled with 0 rather than left missing.
    """
    daily = (
        df.set_index("invoice_date")
        .resample("D")
        .agg(
            revenue=("total", "sum"),
            quantity=("quantity", "sum"),
            n_invoices=("invoice", "nunique"),
            n_customers=("customer_id", "nunique"),
        )
    )
    daily.index.name = "date"
    return daily


def add_rolling_features(
    daily: pd.DataFrame, windows: tuple[int, ...] = ROLLING_WINDOWS
) -> pd.DataFrame:
    """Add rolling mean and standard deviation features for revenue and quantity.

    Args:
        daily: Output of ``build_daily_aggregates``.
        windows: Window sizes in days, defaults to (7, 14, 30).

    Returns:
        ``daily`` with added columns
        ``{revenue,quantity}_roll{window}_{mean,std}`` for each window.
    """
    daily = daily.copy()
    for window in windows:
        for col in ("revenue", "quantity"):
            roll = daily[col].rolling(window=window, min_periods=window)
            daily[f"{col}_roll{window}_mean"] = roll.mean()
            daily[f"{col}_roll{window}_std"] = roll.std()
    return daily


def write_daily_features(daily: pd.DataFrame, out_dir: Path = FEATURES_DIR) -> Path:
    """save dily feature table as csv.

    Args:
        daily: Output of ``add_rolling_features``.
        out_dir: Destination directory, defaults to ``data/features``.

    Returns:
        Path the table was written to.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "daily_sales_features.csv"
    daily.reset_index().to_csv(path, index=False)
    return path


def main() -> None:
    processed_dir = FEATURES_DIR.parent / "processed"
    df = pd.read_csv(processed_dir / "completed_sales.csv")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    daily = build_daily_aggregates(df)
    daily = add_rolling_features(daily)
    write_daily_features(daily)


if __name__ == "__main__":
    main()
