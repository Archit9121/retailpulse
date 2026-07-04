"""Day 9: XGBoost churn prediction with SHAP explainability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = (
    "recency_days",
    "frequency",
    "monetary",
    "total_quantity",
    "n_distinct_products",
    "avg_order_value",
    "tenure_days",
    "avg_days_between_purchases",
    "monetary_per_day",
    "is_uk",
)


def train_test_split_churn(
    dataset: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split on the churn label.

    Args:
        dataset: Output of ``src.features.churn_labeling.build_churn_dataset``.
        test_size: Fraction of customers held out for testing.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test), where X is restricted
        to ``FEATURE_COLUMNS``.
    """
    X = dataset[list(FEATURE_COLUMNS)]
    y = dataset["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def fit_xgboost_churn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    n_estimators: int = 300,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
) -> xgb.XGBClassifier:
    """Fit an XGBoost binary classifier with early stopping on a validation split.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features, used only for early stopping.
        y_val: Validation labels.
        max_depth: Max tree depth.
        learning_rate: Boosting learning rate.
        n_estimators: Max number of boosting rounds (early stopping may
            stop before reaching this).
        subsample: Row subsample ratio per tree.
        colsample_bytree: Column subsample ratio per tree.
        random_state: Seed for reproducibility.

    Returns:
        The fitted ``XGBClassifier``.
    """
    model = xgb.XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        eval_metric="auc",
        early_stopping_rounds=20,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def precision_at_top_k(y_true: pd.Series, y_scores: np.ndarray, k_frac: float = 0.2) -> float:
    """Precision among the top ``k_frac`` fraction of customers by predicted churn score.

    Args:
        y_true: True binary labels.
        y_scores: Predicted churn probabilities, same order as ``y_true``.
        k_frac: Fraction of the population to consider.

    Returns:
        Precision (fraction of flagged customers who are truly churned)
        among the top ``k_frac`` highest-scored customers.
    """
    n = len(y_true)
    k = max(1, int(np.ceil(n * k_frac)))
    y_true_arr = np.asarray(y_true)
    order = np.argsort(-y_scores)
    top_k_labels = y_true_arr[order[:k]]
    return float(top_k_labels.mean())


def evaluate_churn_model(model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate against the project's churn constraints and standard classification metrics.

    Args:
        model: A fitted churn classifier.
        X_test: Held-out test features.
        y_test: Held-out test labels.

    Returns:
        Dict with ``auc_roc``, ``precision_at_top20pct`` (the two
        non-negotiable constraints), plus ``precision_at_0.5``,
        
    """
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_scores >= 0.5).astype(int)

    return {
        "auc_roc": float(roc_auc_score(y_test, y_scores)),
        "precision_at_top20pct": precision_at_top_k(y_test, y_scores, k_frac=0.2),
        "precision_at_0.5": float(precision_score(y_test, y_pred_default)),
        "recall_at_0.5": float(recall_score(y_test, y_pred_default)),
    }


def compute_shap_values(model: xgb.XGBClassifier, X: pd.DataFrame) -> shap.Explanation:
    """Compute SHAP values for the given feature matrix using a TreeExplainer.

    Args:
        model: A fitted ``XGBClassifier``.
        X: Feature matrix to explain (typically the test set).

    Returns:
        A SHAP ``Explanation`` object covering every row in ``X``.
    """
    explainer = shap.TreeExplainer(model)
    return explainer(X)
