"""Day 9: churn label and feature engineering."""

from __future__ import annotations


from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
FEATURES_DIR = Path(__file__).resolve().parents[2] / "data" / "features"

CUTOFF_DATE = pd.Timestamp("2011-09-10")
CHURN_WINDOW_DAYS = 90


def build_churn_labels(
    df: pd.DataFrame, cutoff: pd.Timestamp = CUTOFF_DATE, window_days: int = CHURN_WINDOW_DAYS
) -> pd.DataFrame:
    """Label each customer active as of the cutoff as churned or retained.

    Args:
        df: ``customer_sales`` DataFrame with ``customer_id`` and
            ``invoice_date`` columns.
        cutoff: The observation cutoff date.
        window_days: Length of the forward-looking window used to decide
            churn.

    Returns:
        DataFrame indexed by ``customer_id`` with a single ``churned``
        column (1 = no purchase in the window after cutoff, 0 = at least
        one). Only includes customers with a purchase on or before cutoff;
        a customer who hasn't bought anything yet isn't a churn candidate.
    """
    window_end = cutoff + pd.Timedelta(days=window_days)
    active_mask = df["invoice_date"] <= cutoff
    active_customers = df.loc[active_mask, "customer_id"].unique()

    future_mask = (df["invoice_date"] > cutoff) & (df["invoice_date"] <= window_end)
    returned_customers = set(df.loc[future_mask, "customer_id"].unique())

    labels = pd.DataFrame({"customer_id": active_customers})
    labels["churned"] = (~labels["customer_id"].isin(returned_customers)).astype(int)
    return labels.set_index("customer_id")


def build_churn_features(df: pd.DataFrame, cutoff: pd.Timestamp = CUTOFF_DATE) -> pd.DataFrame:
    """Build cutoff behavioral features per customer.

    Args:
        df: ``customer_sales`` DataFrame with ``customer_id``,
            ``invoice``, ``invoice_date``, ``total``, ``quantity``,
            ``stock_code``, ``country`` columns.
        cutoff: The observation cutoff date. Every row used here must have
            ``invoice_date <= cutoff``.

    Returns:
        DataFrame indexed by ``customer_id`` with columns:
        ``recency_days``, ``frequency``, ``monetary``, ``avg_order_value``,
        ``total_quantity``, ``n_distinct_products``, ``tenure_days``,
        ``avg_days_between_purchases``, ``monetary_per_day``, ``is_uk``.
    """
    pre_cutoff = df[df["invoice_date"] <= cutoff].copy()

    grouped = pre_cutoff.groupby("customer_id")
    features = grouped.agg(
        recency_days=("invoice_date", lambda s: (cutoff - s.max()).days),
        frequency=("invoice", "nunique"),
        monetary=("total", "sum"),
        total_quantity=("quantity", "sum"),
        n_distinct_products=("stock_code", "nunique"),
        first_purchase=("invoice_date", "min"),
        last_purchase=("invoice_date", "max"),
    )

    features["avg_order_value"] = features["monetary"] / features["frequency"]
    features["tenure_days"] = (cutoff - features["first_purchase"]).dt.days
    # A customer with frequency=1 has no "between purchases" interval
    features["avg_days_between_purchases"] = features["tenure_days"] / features["frequency"].clip(
        lower=1
    )
    features.loc[features["frequency"] == 1, "avg_days_between_purchases"] = features.loc[
        features["frequency"] == 1, "tenure_days"
    ]
    # tenure_days can be 0 for a customer whose only purchase happened on
    # the cutoff date itself
    features["monetary_per_day"] = features["monetary"] / features["tenure_days"].clip(lower=1)

    is_uk = pre_cutoff.groupby("customer_id")["country"].agg(
        lambda s: (s == "United Kingdom").mean()
    )
    features["is_uk"] = (is_uk >= 0.5).astype(int)

    features = features.drop(columns=["first_purchase", "last_purchase"])
    return features


def build_churn_dataset(
    df: pd.DataFrame, cutoff: pd.Timestamp = CUTOFF_DATE, window_days: int = CHURN_WINDOW_DAYS
) -> pd.DataFrame:
    """Join features and labels.

    Args:
        df: ``customer_sales`` DataFrame.
        cutoff: Observation cutoff date.
        window_days: Churn window length in days.

    Returns:
        DataFrame indexed by ``customer_id`` with all feature columns plus
        ``churned``.
    """
    features = build_churn_features(df, cutoff=cutoff)
    labels = build_churn_labels(df, cutoff=cutoff, window_days=window_days)
    dataset = features.join(labels, how="inner")
    return dataset


def write_churn_dataset(dataset: pd.DataFrame, out_dir: Path = FEATURES_DIR) -> Path:
    """save the churn modeling dataset as csv.

    Args:
        dataset: Output of ``build_churn_dataset``.
        out_dir: Destination directory, defaults to ``data/features``.

    Returns:
        Path the table was written to.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "churn_dataset.csv"
    dataset.reset_index().to_csv(path, index=False)
    return path


def main() -> None:
    df = pd.read_csv(PROCESSED_DIR / "customer_sales.csv")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    dataset = build_churn_dataset(df)
    write_churn_dataset(dataset)


if __name__ == "__main__":
    main()
