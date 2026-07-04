"""Day 11 entry point: Optuna tuning + feature importance analysis for churn.

Usage:
    python -m src.models.run_churn_tuning
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from src.features.churn_labeling import CUTOFF_DATE, build_churn_dataset
from src.features.churn_trend_features import build_enriched_features
from src.models.churn_model import FEATURE_COLUMNS
from src.models.churn_tuning import evaluate_on_test, fit_best_model, run_optuna_search

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FEATURES_DIR = ROOT_DIR / "data" / "features"
LOCAL_CHURN_MODEL_DIR = FEATURES_DIR / "churn_artifacts" / "models" / "tuned_churn_model"
N_TRIALS = 60
AUC_TARGET = 0.88
PRECISION_TARGET = 0.75

ENRICHED_FEATURE_COLUMNS = list(FEATURE_COLUMNS) + [
    "recency_ratio",
    "frequency_trend",
    "monetary_trend",
]


def run_experiment(name: str, X: pd.DataFrame, y: pd.Series, n_trials: int) -> dict:
    """Run one full Optuna-tuned experiment (split, search, refit, evaluate).

    Args:
        name: Experiment label, used in logging and the nested MLflow run name.
        X: Full feature matrix for this experiment's feature set.
        y: Labels.
        n_trials: Number of Optuna trials.

    Returns:
        Dict with the fitted model, test metrics, and the feature columns used.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    with mlflow.start_run(run_name=name, nested=True):
        mlflow.log_param("feature_set", name)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_trials", n_trials)

        study = run_optuna_search(X_tr, y_tr, X_val, y_val, n_trials=n_trials)
        model = fit_best_model(study.best_params, X_tr, y_tr, X_val, y_val)
        metrics = evaluate_on_test(model, X_test, y_test)

        mlflow.log_params(study.best_params)
        mlflow.log_metric("val_auc", study.best_value)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        logger.info(
            "[%s] val_auc=%.4f, test_auc=%.4f, precision@20%%=%.4f",
            name,
            study.best_value,
            metrics["auc_roc"],
            metrics["precision_at_top20pct"],
        )

    return {
        "model": model,
        "metrics": metrics,
        "feature_columns": list(X.columns),
        "X_test": X_test,
        "y_test": y_test,
        "best_params": study.best_params,
    }


def main() -> None:
    """Run the Day 11 churn tuning pipeline end to end."""
    mlflow.set_tracking_uri(f"file:{ROOT_DIR / 'mlruns'}")
    mlflow.set_experiment("churn_prediction")

    raw = pd.read_csv(PROCESSED_DIR / "customer_sales.csv")
    raw["invoice_date"] = pd.to_datetime(raw["invoice_date"])
    dataset = build_churn_dataset(raw, cutoff=CUTOFF_DATE)
    enriched = build_enriched_features(
        raw[raw["invoice_date"] <= CUTOFF_DATE], dataset, CUTOFF_DATE
    )

    y = enriched["churned"]

    with mlflow.start_run(run_name="day11_churn_tuning"):
        mlflow.log_param("cutoff_date", str(CUTOFF_DATE.date()))

        baseline_result = run_experiment(
            "optuna_baseline_features", enriched[list(FEATURE_COLUMNS)], y, n_trials=N_TRIALS
        )
        enriched_result = run_experiment(
            "optuna_enriched_features", enriched[ENRICHED_FEATURE_COLUMNS], y, n_trials=N_TRIALS
        )

        best_name, best_result = max(
            [("baseline", baseline_result), ("enriched", enriched_result)],
            key=lambda kv: kv[1]["metrics"]["auc_roc"],
        )
        other_name = "enriched" if best_name == "baseline" else "baseline"
        other_auc = (
            enriched_result["metrics"]["auc_roc"]
            if best_name == "baseline"
            else baseline_result["metrics"]["auc_roc"]
        )
        logger.info(
            "Best feature set: %s (test AUC=%.4f vs. %s AUC=%.4f)",
            best_name,
            best_result["metrics"]["auc_roc"],
            other_name,
            other_auc,
        )

        mlflow.log_param("best_feature_set", best_name)
        mlflow.log_metric("best_test_auc", best_result["metrics"]["auc_roc"])
        mlflow.log_metric(
            "best_test_precision_at_top20pct", best_result["metrics"]["precision_at_top20pct"]
        )
        mlflow.xgboost.log_model(best_result["model"], name="tuned_churn_model")
        if LOCAL_CHURN_MODEL_DIR.exists():
            shutil.rmtree(LOCAL_CHURN_MODEL_DIR)
        LOCAL_CHURN_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
        mlflow.xgboost.save_model(best_result["model"], path=str(LOCAL_CHURN_MODEL_DIR))

        auc_met = best_result["metrics"]["auc_roc"] >= AUC_TARGET
        precision_met = best_result["metrics"]["precision_at_top20pct"] >= PRECISION_TARGET
        mlflow.log_param("meets_auc_target", auc_met)
        mlflow.log_param("meets_precision_target", precision_met)
        logger.info(
            "AUC target (>=%.2f): %s. Precision@20%% target (>=%.2f): %s",
            AUC_TARGET,
            "MET" if auc_met else "NOT MET",
            PRECISION_TARGET,
            "MET" if precision_met else "NOT MET",
        )

        explainer = shap.TreeExplainer(best_result["model"])
        shap_values = explainer(best_result["X_test"])
        mean_abs_shap = pd.DataFrame(
            {
                "feature": best_result["X_test"].columns,
                "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)

        out_dir = FEATURES_DIR / "churn_artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)
        mean_abs_shap.to_csv(out_dir / "shap_feature_importance_tuned.csv", index=False)
        enriched.reset_index().to_csv(out_dir / "churn_dataset_enriched.csv", index=False)

        logger.info("Top 5 features (tuned model): %s", mean_abs_shap.head(5).to_dict("records"))


if __name__ == "__main__":
    main()
