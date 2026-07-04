"""Day 12: drift detection using Evidently AI.

Two realistic monitoring scenarios, each comparing a "reference" snapshot
against a later "current" one using the same feature-engineering.

- **Churn features**: the same ``build_churn_features`` logic from Day 9,
  computed at an earlier cutoff (reference) and the Day 9 cutoff
  (current).
- **Demand features**: the daily aggregate/rolling features from Day 2,
  split into an earlier and a later period of the same series. 

"""

from __future__ import annotations
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FEATURES_DIR = ROOT_DIR / "data" / "features"
MONITORING_DIR = ROOT_DIR / "data" / "features" / "monitoring"


def run_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame, columns: list[str]):
    """Run an Evidently data drift report comparing two feature snapshots.

    Args:
        reference_df: The baseline snapshot.
        current_df: The later snapshot to check for drift against the
            baseline.
        columns: Which columns to include in the comparison. Restricting
            to a known, deliberate column list rather than passing
            everything keeps the report focused on actual model features.

    Returns:
        The Evidently ``Snapshot`` (report result).
    """
    ref_ds = Dataset.from_pandas(reference_df[columns], data_definition=DataDefinition())
    cur_ds = Dataset.from_pandas(current_df[columns], data_definition=DataDefinition())
    report = Report(metrics=[DataDriftPreset()])
    return report.run(cur_ds, ref_ds)


def extract_drift_summary(snapshot) -> pd.DataFrame:
    """Parse an Evidently snapshot into a clean per-column drift table.

    Args:
        snapshot: Output of ``run_drift_report``.

    Returns:
        DataFrame with columns ``column``, ``method``, ``value``, and
        ``drifted``. Excludes the overall ``DriftedColumnsCount`` summary
        metric, which is handled separately by ``summarize_drift``.

    """
    rows = []
    for metric in snapshot.dict()["metrics"]:
        config = metric.get("config", {})
        if config.get("type") != "evidently:metric_v2:ValueDrift":
            continue
        method = config.get("method", "")
        value = metric["value"]
        threshold = config.get("threshold", 0.05)
        is_pvalue_method = "p_value" in method
        drifted = (value < threshold) if is_pvalue_method else (value > threshold)
        rows.append(
            {"column": config["column"], "method": method, "value": value, "drifted": drifted}
        )
    return pd.DataFrame(rows).sort_values("column").reset_index(drop=True)


def summarize_drift(snapshot) -> dict:
    """Pull the overall drifted-column count from a snapshot.

    Args:
        snapshot: Output of ``run_drift_report``.

    Returns:
        Dict with ``n_drifted`` and ``drifted_share``
    """
    for metric in snapshot.dict()["metrics"]:
        if metric.get("config", {}).get("type") == "evidently:metric_v2:DriftedColumnsCount":
            return {
                "n_drifted": metric["value"]["count"],
                "drifted_share": metric["value"]["share"],
            }
    return {"n_drifted": 0, "drifted_share": 0.0}


def recommend_retraining(drift_summary: dict, share_threshold: float = 0.5) -> bool:
    """Turn a drift summary into a retraining recommendation.

    Args:
        drift_summary: Output of ``summarize_drift``.
        share_threshold: If the drifted column share meets or exceeds this,
            recommend retraining, defaults to 0.5 (half or more of the
            monitored features have shifted).

    Returns:
        True if retraining is recommended.
    """
    return drift_summary["drifted_share"] >= share_threshold


def save_drift_report(snapshot, name: str, out_dir: Path = MONITORING_DIR) -> Path:
    """save an Evidently snapshot as HTML for human review.

    Args:
        snapshot: Output of ``run_drift_report``.
        name: Base filename (without extension).
        out_dir: Destination directory, defaults to
            ``data/features/monitoring``.

    Returns:
        Path the HTML report was written to.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.html"
    snapshot.save_html(str(path))
    return path
