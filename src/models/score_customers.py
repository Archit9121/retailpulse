"""Score the full active customer base with the best tuned churn model."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.xgboost
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT_DIR / "data" / "features"


def load_best_churn_model(run_name: str = "day11_churn_tuning") -> tuple:
    """Load the best tuned churn model and its expected feature order from MLflow.

    Args:
        run_name: The parent run name to search for, defaults to Day 11's
            tuning run.

    Returns:
        Tuple of (fitted XGBClassifier, list of feature names in the order
        the model expects them).

    Raises:
        ValueError: If no matching run with a logged model is found.
    """
    mlflow.set_tracking_uri(f"file:{ROOT_DIR / 'mlruns'}")
    experiment = mlflow.get_experiment_by_name("churn_prediction")
    if experiment is None:
        raise ValueError("No churn_prediction experiment found; run Day 9-11's pipelines first.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError(f"No run named '{run_name}' found in churn_prediction.")

    run_id = runs.iloc[0]["run_id"]
    model = mlflow.xgboost.load_model(f"runs:/{run_id}/tuned_churn_model")
    feature_names = model.get_booster().feature_names
    return model, feature_names


def score_customers(
    model, feature_names: list[str], customer_features: pd.DataFrame
) -> pd.DataFrame:
    """Score every customer in a feature table with churn probability.

    Args:
        model: A fitted classifier with ``predict_proba``.
        feature_names: Feature columns in the exact order the model
            expects them.
        customer_features: DataFrame indexed by ``customer_id``.

    Returns:
        DataFrame with ``customer_id``, every column in ``feature_names``,
        and ``churn_probability``, sorted descending by risk.
    """
    df = customer_features.copy()
    if "customer_id" in df.columns:
        df = df.set_index("customer_id")

    missing = set(feature_names) - set(df.columns)
    if missing:
        raise ValueError(f"customer_features is missing required columns: {missing}")

    X = df[feature_names]
    df["churn_probability"] = model.predict_proba(X)[:, 1]
    return df.reset_index().sort_values("churn_probability", ascending=False)


def write_risk_scores(scores: pd.DataFrame, out_dir: Path = FEATURES_DIR) -> Path:
    """save the full customer risk-score table.

    Args:
        scores: Output of ``score_customers``.
        out_dir: Destination directory, defaults to ``data/features``.

    Returns:
        Path the table was written to.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "churn_artifacts" / "customer_risk_scores.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(path, index=False)
    return path


def main() -> None:
    customer_features = pd.read_csv(
        FEATURES_DIR / "churn_artifacts" / "churn_dataset_enriched.csv"
    )
    model, feature_names = load_best_churn_model()
    scores = score_customers(model, feature_names, customer_features)
    write_risk_scores(scores)
    


if __name__ == "__main__":
    main()
