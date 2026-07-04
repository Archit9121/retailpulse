"""Time-series preparation for demand forecasting: stationarity and decomposition."""

from __future__ import annotations


from pathlib import Path

import pandas as pd
from statsmodels.tsa.seasonal import STL, DecomposeResult
from statsmodels.tsa.stattools import adfuller, kpss


FEATURES_DIR = Path(__file__).resolve().parents[2] / "data" / "features"


def flag_partial_boundary_days(
    raw: pd.DataFrame, daily_index: pd.DatetimeIndex, close_hour: int = 17
) -> pd.Series:
    """Flag the first and/or last day of the series if its timestamps don't reach a normal close.

    Args:
        raw: Transaction-level DataFrame with an ``invoice_date`` column
            (datetime), used to check what time of day each boundary date's
            transactions actually span.
        daily_index: The daily-resampled index to flag against.
        close_hour: Hour of day (24h) a normal trading day is expected to
            reach. Defaults to 17 (5pm), comfortably below this dataset's
            observed normal closing times (typically 18:00-20:00), so only
            a day cut off well before closing gets flagged, not an
            ordinarily early-closing day.

    Returns:
        Boolean Series aligned to ``daily_index``, True only for a boundary
        day (first or last) whose latest transaction is earlier than
        ``close_hour``. Interior days are never flagged, since a gap inside
        the range is a missing-data question, not a partial-collection one.
    """
    flags = pd.Series(False, index=daily_index)
    if len(daily_index) == 0:
        return flags

    for boundary_date in (daily_index[0], daily_index[-1]):
        day_txns = raw[raw["invoice_date"].dt.date == boundary_date.date()]
        if day_txns.empty:
            continue
        latest_hour = day_txns["invoice_date"].max().hour
        if latest_hour < close_hour:
            flags.loc[boundary_date] = True
    return flags


def get_clean_series(
    daily: pd.DataFrame, column: str = "quantity", exclude_partial: bool = True
) -> pd.Series:
    """Extract a single demand column,dropping partial boundary days.

    Args:
        daily: Daily DataFrame with an ``is_partial_day`` column.
        exclude_partial: If True (default), drop rows flagged as partial
            boundary days.

    Returns:
        A Series indexed by date, ready for ``run_adf_test``,
        ``run_kpss_test``, and ``decompose_series``.
    """
    df = daily
    if exclude_partial and "is_partial_day" in df.columns:
        n_before = len(df)
        df = df[~df["is_partial_day"]]
    return df[column].copy()


def run_adf_test(series: pd.Series) -> dict:
    """Run the Augmented Dickey-Fuller stationarity test.

    Args:
        series: A time-ordered numeric series with no missing values.

    Returns:
        Dict with ``statistic``, ``p_value``, ``n_lags``, ``critical_values``,
        and ``is_stationary`` (True if p_value < 0.05, i.e. the null
        hypothesis of a unit root is rejected).
    """
    statistic, p_value, n_lags, n_obs, critical_values, _ = adfuller(series, autolag="AIC")
    return {
        "test": "ADF",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_lags": int(n_lags),
        "n_obs": int(n_obs),
        "critical_values": {k: float(v) for k, v in critical_values.items()},
        "is_stationary": bool(p_value < 0.05),
    }


def run_kpss_test(series: pd.Series) -> dict:
    """Run the KPSS stationarity test.

    Args:
        series: A time-ordered numeric series with no missing values.

    Returns:
        Dict with ``statistic``, ``p_value``, ``n_lags``, ``critical_values``,
        and ``is_stationary`` (True if p_value >= 0.05; KPSS's null
        hypothesis is stationarity, the opposite orientation from ADF, so
        a low p-value here means non-stationary, not stationary).
    """
    statistic, p_value, n_lags, critical_values = kpss(series, regression="c", nlags="auto")
    return {
        "test": "KPSS",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_lags": int(n_lags),
        "critical_values": {k: float(v) for k, v in critical_values.items()},
        "is_stationary": bool(p_value >= 0.05),
    }


def summarize_stationarity(adf_result: dict, kpss_result: dict) -> str:
    """Combine ADF and KPSS verdicts into one interpretation.

    Args:
        adf_result: Output of ``run_adf_test``.
        kpss_result: Output of ``run_kpss_test``.

    Returns:
        One of four standard combined verdicts. ADF and KPSS test opposite
        null hypotheses, so they can (and sometimes do) disagree; reporting
        both rather than picking one avoids overstating confidence.
    """
    adf_stationary = adf_result["is_stationary"]
    kpss_stationary = kpss_result["is_stationary"]
    if adf_stationary and kpss_stationary:
        return "stationary"
    if not adf_stationary and not kpss_stationary:
        return "non_stationary"
    if adf_stationary and not kpss_stationary:
        return "conflicting_trend_stationary"
    return "conflicting_difference_stationary"


def decompose_series(series: pd.Series, period: int = 7) -> DecomposeResult:
    """Run STL (Seasonal-Trend decomposition using LOESS) on a demand series.

    Args:
        series: A time-ordered numeric series with no missing values.
        period: Seasonal period in observations, defaults to 7 (weekly
            seasonality on a daily series).

    Returns:
        A ``statsmodels`` ``DecomposeResult`` with ``.trend``,
        ``.seasonal``, and ``.resid`` components.
    """
    stl = STL(series, period=period, robust=True)
    return stl.fit()


def build_demand_diagnostics(
    raw: pd.DataFrame, daily: pd.DataFrame, column: str = "quantity"
) -> dict:
    """Run the full Day 4 diagnostic suite on one demand column.

    Args:
        raw: Transaction-level ``completed_sales`` DataFrame.
        daily: Daily-aggregated DataFrame.
        column: Which demand column to diagnose.

    Returns:
        Dict with ``adf_level``, ``kpss_level``, ``adf_diff1``,
        ``kpss_diff1`` (test results on the level series and its first
        difference), ``level_verdict``, ``diff1_verdict`` (combined
        ADF+KPSS interpretations), and ``n_observations``.
    """
    daily = daily.copy()
    daily["is_partial_day"] = flag_partial_boundary_days(raw, daily.index)
    clean = get_clean_series(daily, column=column)

    adf_level = run_adf_test(clean)
    kpss_level = run_kpss_test(clean)
    diff1 = clean.diff().dropna()
    adf_diff1 = run_adf_test(diff1)
    kpss_diff1 = run_kpss_test(diff1)

    return {
        "column": column,
        "n_observations": len(clean),
        "adf_level": adf_level,
        "kpss_level": kpss_level,
        "level_verdict": summarize_stationarity(adf_level, kpss_level),
        "adf_diff1": adf_diff1,
        "kpss_diff1": kpss_diff1,
        "diff1_verdict": summarize_stationarity(adf_diff1, kpss_diff1),
    }


def write_diagnostics(diagnostics: dict, out_dir: Path = FEATURES_DIR) -> Path:
    """save the diagnostic summary as JSON.

    Args:
        diagnostics: Output of ``build_demand_diagnostics``.
        out_dir: Destination directory, defaults to ``data/features``.

    Returns:
        Path the report was written to.
    """
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"stationarity_report_{diagnostics['column']}.json"
    path.write_text(json.dumps(diagnostics, indent=2, default=str))
    return path


def write_clean_demand_series(daily: pd.DataFrame, out_dir: Path = FEATURES_DIR) -> Path:
    """save the partial-day-flagged daily demand table as csv.

    Args:
        daily: Daily DataFrame with the ``is_partial_day`` flag attached.
        out_dir: Destination directory, defaults to ``data/features``.

    Returns:
        Path the table was written to.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "demand_daily_flagged.csv"
    daily.reset_index().to_csv(path, index=False)
    return path


def write_decomposition(
    series: pd.Series, result: DecomposeResult, column: str, out_dir: Path = FEATURES_DIR
) -> Path:
    """save STL trend/seasonal/residual components as csv.

    Args:
        series: The original series that was decomposed (kept alongside
            the components for easy plotting/validation downstream).
        result: Output of ``decompose_series``.
        column: Name of the demand column that was decomposed, used in the
            output filename.
        out_dir: Destination directory, defaults to ``data/features``.

    Returns:
        Path the table was written to.
    """
    out = pd.DataFrame(
        {
            "observed": series,
            "trend": result.trend,
            "seasonal": result.seasonal,
            "resid": result.resid,
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"decomposition_{column}.csv"
    out.reset_index(names="date").to_csv(path, index=False)
    return path


def main() -> None:
    """Run Day 4 time-series diagnostics on the daily demand series."""
    root = FEATURES_DIR.parent.parent
    raw = pd.read_csv(root / "data" / "processed" / "completed_sales.csv")
    raw["invoice_date"] = pd.to_datetime(raw["invoice_date"])

    daily = pd.read_csv(root / "data" / "features" / "daily_sales_features.csv")
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.set_index("date")
    daily["is_partial_day"] = flag_partial_boundary_days(raw, daily.index)
    write_clean_demand_series(daily)

    for column in ("quantity", "revenue"):
        diagnostics = build_demand_diagnostics(raw, daily, column=column)
        write_diagnostics(diagnostics)


        clean_series = get_clean_series(daily, column=column)
        decomposition = decompose_series(clean_series, period=7)
        write_decomposition(clean_series, decomposition, column=column)


if __name__ == "__main__":
    main()
