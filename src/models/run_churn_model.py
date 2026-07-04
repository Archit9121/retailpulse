"""Day 9 : train the XGBoost churn model.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap

from src.models.churn_model import (
    FEATURE_COLUMNS,
    evaluate_churn_model,
    fit_xgboost_churn,
    train_test_split_churn,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT_DIR / "data" / "features"

# Hyperparamenter
SWEEP_CONFIGS = [
    {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 300},
    {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 500},
    {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 500},
    {"max_depth": 8, "learning_rate": 0.1, "n_estimators": 300},
]

AUC_TARGET = 0.88
PRECISION_TARGET = 0.75


def main() -> None:
    mlflow.set_tracking_uri(f"file:{ROOT_DIR / 'mlruns'}")
    mlflow.set_experiment("churn_prediction")

    dataset = pd.read_csv(FEATURES_DIR / "churn_dataset.csv").set_index("customer_id")
    X_train, X_test, y_train, y_test = train_test_split_churn(dataset, test_size=0.2)
    X_tr, X_val, y_tr, y_val = train_test_split_churn(
        pd.concat([X_train, y_train], axis=1), test_size=0.15
    )

    with mlflow.start_run(run_name="day9_churn_xgboost"):
        mlflow.log_param("n_customers", len(dataset))
        mlflow.log_param("churn_rate", float(dataset["churned"].mean()))
        mlflow.log_param("cutoff_date", "2011-09-10")
        mlflow.log_param("churn_window_days", 90)
        mlflow.log_param("features", list(FEATURE_COLUMNS))

        results = []
        for i, config in enumerate(SWEEP_CONFIGS):
            with mlflow.start_run(run_name=f"xgboost_config_{i}", nested=True):
                mlflow.log_params(config)
                model = fit_xgboost_churn(X_tr, y_tr, X_val, y_val, **config)
                metrics = evaluate_churn_model(model, X_test, y_test)
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)
                results.append({**config, **metrics, "model": model})
                

        best = max(results, key=lambda r: r["auc_roc"])
        best_config = {k: best[k] for k in ("max_depth", "learning_rate", "n_estimators")}
        

        mlflow.log_params({f"best_{k}": v for k, v in best_config.items()})
        mlflow.log_metric("best_auc_roc", best["auc_roc"])
        mlflow.log_metric("best_precision_at_top20pct", best["precision_at_top20pct"])
        mlflow.xgboost.log_model(best["model"], name="churn_model")

        auc_met = best["auc_roc"] >= AUC_TARGET
        precision_met = best["precision_at_top20pct"] >= PRECISION_TARGET
        mlflow.log_param("meets_auc_target", auc_met)
        mlflow.log_param("meets_precision_target", precision_met)
        

        # SHAP explainability on the test set, using the best model.
        explainer = shap.TreeExplainer(best["model"])
        shap_values = explainer(X_test)
        mean_abs_shap = pd.DataFrame(
            {
                "feature": X_test.columns,
                "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)

        out_dir = FEATURES_DIR / "churn_artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)
        mean_abs_shap.to_csv(out_dir / "shap_feature_importance.csv", index=False)

        test_predictions = X_test.copy()
        test_predictions["actual_churned"] = y_test.values
        test_predictions["predicted_proba"] = best["model"].predict_proba(X_test)[:, 1]
        test_predictions.reset_index().to_csv(
            out_dir / "churn_test_predictions.csv", index=False
        )



if __name__ == "__main__":
    main()
