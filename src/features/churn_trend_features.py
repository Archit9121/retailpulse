"""Day 11: trend-based churn features,by Day 9's SHAP findings.
"""

from __future__ import annotations


import numpy as np
import pandas as pd



def add_recency_ratio(features: pd.DataFrame) -> pd.DataFrame:
    """Add recency normalized by the customer's own typical purchase interval.

    Args:
        features: A churn feature DataFrame with ``recency_days`` and
            ``avg_days_between_purchases`` columns (the output of
            ``src.features.churn_labeling.build_churn_features``).

    Returns:
        ``features`` with an added ``recency_ratio`` column.
    """
    features = features.copy()
    denom = features["avg_days_between_purchases"]
    ratio = features["recency_days"] / denom.replace(0, np.nan)
    features["recency_ratio"] = ratio.fillna(0.0)
    return features


def add_trend_features(
    transactions: pd.DataFrame, features: pd.DataFrame, cutoff: pd.Timestamp
) -> pd.DataFrame:
    """Add frequency and monetary trend features.

    Args:
        transactions: Pre-cutoff ``customer_sales`` transactions with
            ``customer_id``, ``invoice``, ``invoice_date``, ``total``.
            Rows after ``cutoff`` are filtered out internally.
        features: A churn feature DataFrame indexed by ``customer_id`` to
            join the new columns onto.
        cutoff: The observation cutoff date.

    Returns:
        ``features`` with added ``frequency_trend`` and ``monetary_trend``
        columns: ``(recent_half_activity) / (older_half_activity + 1)``,
        split at each customer's own history midpoint. Single-purchase
        customers get a neutral value of 1.0 for both.
    """
    pre_cutoff = transactions[transactions["invoice_date"] <= cutoff].copy()
    first_purchase = pre_cutoff.groupby("customer_id")["invoice_date"].min()
    midpoint = first_purchase + (cutoff - first_purchase) / 2
    pre_cutoff["midpoint"] = pre_cutoff["customer_id"].map(midpoint)
    pre_cutoff["is_recent_half"] = pre_cutoff["invoice_date"] > pre_cutoff["midpoint"]

    grouped = pre_cutoff.groupby(["customer_id", "is_recent_half"])
    freq_by_half = grouped["invoice"].nunique().unstack(fill_value=0)
    monetary_by_half = grouped["total"].sum().unstack(fill_value=0)

    freq_trend = (freq_by_half.get(True, 0) / (freq_by_half.get(False, 0) + 1)).rename(
        "frequency_trend"
    )
    monetary_trend = (monetary_by_half.get(True, 0) / (monetary_by_half.get(False, 0) + 1)).rename(
        "monetary_trend"
    )

    out = features.copy()
    out = out.join(freq_trend, how="left").join(monetary_trend, how="left")

    single_purchase = out["frequency"] == 1
    out.loc[single_purchase, "frequency_trend"] = 1.0
    out.loc[single_purchase, "monetary_trend"] = 1.0
    out["frequency_trend"] = out["frequency_trend"].fillna(1.0)
    out["monetary_trend"] = out["monetary_trend"].fillna(1.0)

    
    return out


def build_enriched_features(
    transactions: pd.DataFrame, features: pd.DataFrame, cutoff: pd.Timestamp
) -> pd.DataFrame:
    """Apply all Day 11 feature.

    Args:
        transactions: Pre-cutoff ``customer_sales`` transactions.
        features: Base churn feature DataFrame from Day 9.
        cutoff: The observation cutoff date.

    Returns:
        ``features`` with ``recency_ratio``, ``frequency_trend``, and
        ``monetary_trend`` added.
    """
    enriched = add_recency_ratio(features)
    enriched = add_trend_features(transactions, enriched, cutoff)
    return enriched
