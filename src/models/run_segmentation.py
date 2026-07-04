"""Day 3: run the full customer segmentation pipeline."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from src.models.segmentation import (
    choose_k,
    fit_dbscan,
    fit_kmeans,
    label_segments,
    prepare_features,
    sweep_dbscan_eps,
    sweep_kmeans_k,
    write_segments,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
TARGET_RANGE = (6, 8)


def log_kmeans_sweep(sweep: pd.DataFrame) -> None:
    """Log one MLflow run per k tried in the K-Means sweep.

    Args:
        sweep: Output of ``sweep_kmeans_k``.
    """
    for k, row in sweep.iterrows():
        with mlflow.start_run(run_name=f"kmeans_k{k}", nested=True):
            mlflow.log_param("algorithm", "kmeans")
            mlflow.log_param("k", k)
            mlflow.log_metric("inertia", row["inertia"])
            mlflow.log_metric("silhouette", row["silhouette"])
            mlflow.log_metric("davies_bouldin", row["davies_bouldin"])


def log_dbscan_sweep(sweep: pd.DataFrame, min_samples: int) -> None:
    """Log one MLflow run per eps tried in the DBSCAN sweep.

    Args:
        sweep: Output of ``sweep_dbscan_eps``.
        min_samples: The fixed min_samples used for the whole sweep.
    """
    for eps, row in sweep.iterrows():
        with mlflow.start_run(run_name=f"dbscan_eps{eps:.2f}", nested=True):
            mlflow.log_param("algorithm", "dbscan")
            mlflow.log_param("eps", eps)
            mlflow.log_param("min_samples", min_samples)
            mlflow.log_metric("n_clusters", row["n_clusters"])
            mlflow.log_metric("n_noise", row["n_noise"])
            mlflow.log_metric("noise_pct", row["noise_pct"])
            if not np.isnan(row["silhouette"]):
                mlflow.log_metric("silhouette", row["silhouette"])


def main() -> None:
    mlflow.set_tracking_uri(f"file:{ROOT_DIR / 'mlruns'}")
    mlflow.set_experiment("customer_segmentation")

    rfm = pd.read_csv(ROOT_DIR / "data" / "features" / "rfm.csv").set_index("customer_id")
    X, scaler = prepare_features(rfm)
    

    with mlflow.start_run(run_name="day3_segmentation_pipeline"):
        mlflow.log_param("n_customers", len(rfm))
        mlflow.log_param("target_k_range", str(TARGET_RANGE))

        # --- K-Means sweep ---
        kmeans_sweep = sweep_kmeans_k(X, k_range=range(2, 13))
        log_kmeans_sweep(kmeans_sweep)
        chosen_k = choose_k(kmeans_sweep, target_range=TARGET_RANGE)
        kmeans_model = fit_kmeans(X, chosen_k)
        kmeans_labels = kmeans_model.labels_
        kmeans_sil = kmeans_sweep.loc[chosen_k, "silhouette"]

        # --- DBSCAN sweep ---
        min_samples = 10
        eps_range = np.arange(0.2, 2.01, 0.1)
        dbscan_sweep = sweep_dbscan_eps(X, eps_range, min_samples=min_samples)
        log_dbscan_sweep(dbscan_sweep, min_samples)

        in_target = dbscan_sweep[
            dbscan_sweep["n_clusters"].between(TARGET_RANGE[0], TARGET_RANGE[1])
        ]
        dbscan_viable = not in_target.empty
        if dbscan_viable:
            best_eps = in_target["silhouette"].idxmax()
            dbscan_model = fit_dbscan(X, eps=best_eps, min_samples=min_samples)
            dbscan_sil = in_target.loc[best_eps, "silhouette"]
            mlflow.sklearn.log_model(dbscan_model, name="dbscan_model_not_selected")
            

        # --- Model selection ---
        # K-Means is chosen as the production segmentation: it reliably hits
        # the 6-8 segment target by construction, while DBSCAN's cluster
        # count is an emergent property of eps and isn't guaranteed to land
        # in range.
        final_model_name = "kmeans"
        final_labels = kmeans_labels
        mlflow.log_param("final_model", final_model_name)
        mlflow.log_param("final_k", chosen_k)
        mlflow.log_metric("final_silhouette", kmeans_sil)
        mlflow.log_metric("final_davies_bouldin", kmeans_sweep.loc[chosen_k, "davies_bouldin"])
        mlflow.sklearn.log_model(kmeans_model, name="kmeans_model")

        segmented = label_segments(rfm, final_labels)
        segment_counts = segmented["segment_name"].value_counts()
        mlflow.log_param("n_segments", segmented["segment_name"].nunique())
        for name, count in segment_counts.items():
            mlflow.log_metric(f"segment_size_{name}", count)

        write_segments(segmented)


if __name__ == "__main__":
    main()
