"""RFM (Recency, Frequency, Monetary) feature engineering.

Builds per-customer RFM features from ``customer_sales`` 
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FEATURES_DIR = Path(__file__).resolve().parents[2] / "data" / "features"


def compute_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Compute Recency, Frequency, and Monetary value per customer.

    Args:
        df: ``customer_sales`` DataFrame with ``customer_id``, ``invoice``,
            ``invoice_date``, and ``total`` columns.
        snapshot_date: Reference date to measure recency.one day after the 
        latest invoice date in ``df``

    Returns:
        DataFrame indexed by ``customer_id`` with columns ``recency_days``,
        ``frequency``, and ``monetary``.
    """
    if snapshot_date is None:
        snapshot_date = df["invoice_date"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("customer_id").agg(
        recency_days=("invoice_date", lambda s: (snapshot_date - s.max()).days),
        frequency=("invoice", "nunique"),
        monetary=("total", "sum"),
    )
    return rfm


def add_rfm_scores(rfm: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Add quintile-based R, F, M scores and a combined RFM segment label.

    Args:
        rfm: Output of ``compute_rfm``.
        n_bins: Number of quantile bins per dimension.

    Returns:
        ``rfm`` with added columns ``r_score``, ``f_score``, ``m_score``
    """
    rfm = rfm.copy()
    # Lower recency is better, so scores are reversed (most recent = highest score).
    rfm["r_score"] = pd.qcut(rfm["recency_days"].rank(method="first"), n_bins, labels=False)
    rfm["r_score"] = n_bins - rfm["r_score"]
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), n_bins, labels=False) + 1
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), n_bins, labels=False) + 1

    rfm["rfm_segment"] = rfm.apply(_rule_based_segment, axis=1)
    return rfm


def _rule_based_segment(row: pd.Series) -> str:
    """Map an RFM score with human-readable label.

    Args:
        row: A row with ``r_score``, ``f_score``, ``m_score`` columns.

    Returns:
        One of a small set of coarse segment labels.
    """
    r, f, m = row["r_score"], row["f_score"], row["m_score"]
    if r >= 4 and f >= 4 and m >= 4:
        return "champions"
    if r >= 4 and f <= 2:
        return "new_customers"
    if r <= 2 and f >= 4 and m >= 4:
        return "at_risk_high_value"
    if r <= 2 and f <= 2:
        return "lost"
    return "regular"


def write_rfm(rfm: pd.DataFrame, out_dir: Path = FEATURES_DIR) -> Path:
    """save the RFM feature table as csv.

    Args:
        rfm: Output of ``add_rfm_scores``.
        out_dir: Destination directory, to ``data/features``.

    Returns:
        Path the table was written to.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "rfm.csv"
    rfm.reset_index().to_csv(path, index=False)
    return path


def main() -> None:
    processed_dir = FEATURES_DIR.parent / "processed"
    df = pd.read_csv(processed_dir / "customer_sales.csv")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    rfm = compute_rfm(df)
    rfm = add_rfm_scores(rfm)
    write_rfm(rfm)


if __name__ == "__main__":
    main()
