"""Day 10: inventory optimization"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm



PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
FEATURES_DIR = Path(__file__).resolve().parents[2] / "data" / "features"

# Assumption
LEAD_TIME_DAYS = 7
SERVICE_LEVEL = 0.95
ORDER_COST = 20.0
HOLDING_COST_RATE = 0.20
DEMAND_LOOKBACK_DAYS = 90


def compute_abc_classification(
    df: pd.DataFrame, tier_a_cutoff: float = 0.80, tier_b_cutoff: float = 0.95
) -> pd.DataFrame:
    """Classify products into A/B/C tiers by cumulative revenue contribution.

    Args:
        df: ``completed_sales`` DataFrame with ``stock_code`` and
            ``total`` columns.
        tier_a_cutoff: Cumulative revenue share at which tier A ends.
        tier_b_cutoff: Cumulative revenue share at which tier B ends.

    Returns:
        DataFrame indexed by ``stock_code`` with ``total_revenue``,
        ``revenue_share``, ``cumulative_share``, and ``tier`` (one of
        ``"A"``, ``"B"``, ``"C"``).
    """
    revenue = df.groupby("stock_code")["total"].sum().sort_values(ascending=False)
    total = revenue.sum()
    cumulative_share = revenue.cumsum() / total

    tier = pd.Series("C", index=revenue.index)
    tier[cumulative_share <= tier_a_cutoff] = "A"
    tier[(cumulative_share > tier_a_cutoff) & (cumulative_share <= tier_b_cutoff)] = "B"

    result = pd.DataFrame(
        {
            "total_revenue": revenue,
            "revenue_share": revenue / total,
            "cumulative_share": cumulative_share,
            "tier": tier,
        }
    )
    return result


def compute_product_demand_stats(
    df: pd.DataFrame,
    lookback_days: int = DEMAND_LOOKBACK_DAYS,
    reference_date: pd.Timestamp | None = None,
    quantity_cap_percentile: float = 0.999,
) -> pd.DataFrame:
    """Compute per-product daily demand mean/std over a recent lookback window.

    Args:
        df: ``completed_sales`` DataFrame with ``stock_code``,
            ``invoice_date``, ``quantity`` columns.
        lookback_days: Number of trailing days to compute demand
            statistics over.
        reference_date: The date the lookback window ends at, defaults to
            the latest date in ``df``.
        quantity_cap_percentile: Per-transaction quantity is winsorized at
            this percentile of the *global* (whole-dataset) quantity
            distribution before aggregating into daily demand, defaults to
            0.999.

    Returns:
        DataFrame indexed by ``stock_code`` with ``daily_demand_mean``,
        ``daily_demand_std``, ``description`` , and ``avg_unit_price``
    """
    if reference_date is None:
        reference_date = df["invoice_date"].max()
    window_start = reference_date - pd.Timedelta(days=lookback_days)

    quantity_cap = df["quantity"].quantile(quantity_cap_percentile)

    window = df[(df["invoice_date"] > window_start) & (df["invoice_date"] <= reference_date)].copy()
    window["quantity"] = window["quantity"].clip(upper=quantity_cap)

    daily = (
        window.groupby(["stock_code", window["invoice_date"].dt.date])["quantity"]
        .sum()
        .rename("daily_qty")
        .reset_index()
    )
    all_dates = pd.date_range(window_start + pd.Timedelta(days=1), reference_date, freq="D").date
    stats_rows = []
    for stock_code, grp in daily.groupby("stock_code"):
        full_series = pd.Series(0.0, index=all_dates)
        full_series.update(grp.set_index("invoice_date")["daily_qty"])
        stats_rows.append(
            {
                "stock_code": stock_code,
                "daily_demand_mean": full_series.mean(),
                "daily_demand_std": full_series.std(ddof=0) if len(full_series) > 1 else 0.0,
            }
        )
    stats = pd.DataFrame(stats_rows).set_index("stock_code")

    desc_lookup = window.groupby("stock_code")["description"].agg(lambda s: s.mode().iat[0])
    price_lookup = window.groupby("stock_code")["unit_price"].mean()
    stats["description"] = desc_lookup
    stats["avg_unit_price"] = price_lookup
    return stats


def compute_safety_stock(
    daily_demand_std: pd.Series, lead_time_days: int, service_level: float
) -> pd.Series:
    """Safety stock needed to cover demand variability during lead time.

    Args:
        daily_demand_std: Per-product standard deviation of daily demand.
        lead_time_days: Supplier lead time in days.
        service_level: Target probability of not stocking out during lead
            time.

    Returns:
        Safety stock in units, ``z * daily_demand_std * sqrt(lead_time_days)``,
        where ``z`` is the inverse normal CDF at ``service_level``. The
        square-root scaling assumes day-to-day demand is independent.
    """
    z = norm.ppf(service_level)
    return z * daily_demand_std * np.sqrt(lead_time_days)


def compute_reorder_point(
    daily_demand_mean: pd.Series, lead_time_days: int, safety_stock: pd.Series
) -> pd.Series:
    """Stock level at which a new order should be placed.

    Args:
        daily_demand_mean: Per-product mean daily demand.
        lead_time_days: Assumed supplier lead time in days.
        safety_stock: Output of ``compute_safety_stock``.

    Returns:
        Reorder point in units: expected demand during lead time plus
        safety stock.
    """
    return daily_demand_mean * lead_time_days + safety_stock


def compute_eoq(
    daily_demand_mean: pd.Series, order_cost: float, holding_cost_per_unit_per_year: pd.Series
) -> pd.Series:
    """Economic order quantity: the order size minimizing total ordering + holding cost.

    Args:
        daily_demand_mean: Per-product mean daily demand.
        order_cost: Cost per purchase order, in GBP.
        holding_cost_per_unit_per_year: Per-product annual holding cost
            per unit, typically ``HOLDING_COST_RATE * avg_unit_price``.

    Returns:
        EOQ in units: ``sqrt(2 * annual_demand * order_cost / holding_cost)``.
        Products with zero holding cost (e.g. a product given away free)
        are guarded against division by zero and return 0 rather than inf.
    """
    annual_demand = daily_demand_mean * 365
    safe_holding_cost = holding_cost_per_unit_per_year.replace(0, np.nan)
    eoq = np.sqrt(2 * annual_demand * order_cost / safe_holding_cost)
    return eoq.fillna(0.0)


def build_inventory_recommendations(
    df: pd.DataFrame,
    tiers_to_optimize: tuple[str, ...] = ("A", "B"),
    lead_time_days: int = LEAD_TIME_DAYS,
    service_level: float = SERVICE_LEVEL,
    order_cost: float = ORDER_COST,
    holding_cost_rate: float = HOLDING_COST_RATE,
    lookback_days: int = DEMAND_LOOKBACK_DAYS,
    quantity_cap_percentile: float = 0.999,
) -> pd.DataFrame:
    """Build the full per-product inventory recommendation table.

    Args:
        df: ``completed_sales`` DataFrame.
        tiers_to_optimize: Which ABC tiers to compute recommendations for,
            defaults to A and B. Tier C (the long tail of low-revenue
            products) is excluded by default
        lead_time_days: Assumed supplier lead time.
        service_level: Target service level.
        order_cost: Assumed fixed cost per order.
        holding_cost_rate: Assumed annual holding cost as a fraction of
            unit price.
        lookback_days: Demand statistics lookback window.
        quantity_cap_percentile: Passed through to
            ``compute_product_demand_stats``.

    Returns:
        DataFrame indexed by ``stock_code`` with ABC tier, demand stats,
        safety stock, reorder point, and EOQ, restricted to
        ``tiers_to_optimize``.
    """
    abc = compute_abc_classification(df)
    demand_stats = compute_product_demand_stats(
        df, lookback_days=lookback_days, quantity_cap_percentile=quantity_cap_percentile
    )

    combined = abc.join(demand_stats, how="inner")
    combined = combined[combined["tier"].isin(tiers_to_optimize)].copy()

    combined["safety_stock"] = compute_safety_stock(
        combined["daily_demand_std"], lead_time_days, service_level
    )
    combined["reorder_point"] = compute_reorder_point(
        combined["daily_demand_mean"], lead_time_days, combined["safety_stock"]
    )
    holding_cost_per_unit = holding_cost_rate * combined["avg_unit_price"]
    combined["eoq"] = compute_eoq(combined["daily_demand_mean"], order_cost, holding_cost_per_unit)

    combined["lead_time_days"] = lead_time_days
    combined["service_level"] = service_level
    
    return combined


def simulate_reorder_trigger(current_stock: float, reorder_point: float, eoq: float) -> dict:
    """Demonstrate the reorder decision against current stock level.

    Args:
        current_stock: A current stock level, in units.
        reorder_point: Output of ``compute_reorder_point`` for this product.
        eoq: Output of ``compute_eoq`` for this product.

    Returns:
        Dict with ``should_reorder`` (bool) and ``recommended_order_qty``
        (the EOQ if reordering, else 0).
    """
    should_reorder = current_stock < reorder_point
    return {
        "should_reorder": bool(should_reorder),
        "recommended_order_qty": float(eoq) if should_reorder else 0.0,
    }


def write_recommendations(recommendations: pd.DataFrame, out_dir: Path = FEATURES_DIR) -> Path:
    """save the inventory recommendation table as csv.

    Args:
        recommendations: Output of ``build_inventory_recommendations``.
        out_dir: Destination directory, defaults to ``data/features``.

    Returns:
        Path the table was written to.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "inventory_recommendations.csv"
    recommendations.reset_index().to_csv(path, index=False)
    return path


def main() -> None:
    df = pd.read_csv(PROCESSED_DIR / "completed_sales.csv")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    recommendations = build_inventory_recommendations(df)
    write_recommendations(recommendations)


if __name__ == "__main__":
    main()
