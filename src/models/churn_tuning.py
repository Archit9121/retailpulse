"""Day 11: Optuna hyperparameter search for the churn model."""

from __future__ import annotations


import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from src.models.churn_model import precision_at_top_k



def objective(
    trial: optuna.Trial, X_tr: pd.DataFrame, y_tr: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
) -> float:
    """ validation AUC-ROC for one hyperparameter draw.

    Args:
        trial: The current Optuna trial, used to sample hyperparameters.
        X_tr: Training features.
        y_tr: Training labels.
        X_val: Validation features.
        y_val: Validation labels.

    Returns:
        Validation AUC-ROC for this trial's hyperparameter draw.
    """
    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    model = xgb.XGBClassifier(
        **params,
        random_state=42,
        eval_metric="auc",
        early_stopping_rounds=20,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    y_scores = model.predict_proba(X_val)[:, 1]
    return float(roc_auc_score(y_val, y_scores))


def run_optuna_search(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    seed: int = 42,
) -> optuna.Study:
    """Run the Optuna search and return the completed study.

    Args:
        X_tr: Training features.
        y_tr: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        n_trials: Number of trials to run, defaults to 50.
        seed: Sampler seed for reproducibility.

    Returns:
        The completed Optuna ``Study``. ``study.best_params`` and
        ``study.best_value`` give the winning hyperparameters and the
        validation AUC they achieved.
    """
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective(trial, X_tr, y_tr, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    return study


def fit_best_model(
    best_params: dict, X_tr: pd.DataFrame, y_tr: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
) -> xgb.XGBClassifier:
    """Refit an XGBoost model with the best Optuna hyperparameters.

    Args:
        best_params: ``study.best_params`` from ``run_optuna_search``.
        X_tr: Training features.
        y_tr: Training labels.
        X_val: Validation features, used for early stopping only.
        y_val: Validation labels.

    Returns:
        The fitted ``XGBClassifier``.
    """
    model = xgb.XGBClassifier(
        **best_params,
        random_state=42,
        eval_metric="auc",
        early_stopping_rounds=20,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def evaluate_on_test(model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate a tuned model on the test set.

    Args:
        model: A fitted churn classifier.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dict with ``auc_roc`` and ``precision_at_top20pct``.
    """
    y_scores = model.predict_proba(X_test)[:, 1]
    return {
        "auc_roc": float(roc_auc_score(y_test, y_scores)),
        "precision_at_top20pct": precision_at_top_k(y_test, y_scores, k_frac=0.2),
    }
