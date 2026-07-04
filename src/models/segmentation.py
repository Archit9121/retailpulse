"""Customer segmentation via K-Means and DBSCAN on RFM features."""

from __future__ import annotations


from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


FEATURES_DIR = Path(__file__).resolve().parents[2] / "data" / "features"
RFM_COLS = ("recency_days", "frequency", "monetary")


def prepare_features(rfm: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Log-transform and standardize R, F, M for clustering.

    Args:
        rfm: DataFrame with ``recency_days``, ``frequency``, ``monetary``
            columns (output of ``src.features.rfm.compute_rfm``).

    Returns:
        Tuple of (scaled feature array, fitted scaler). The scaler is
        returned so cluster centers can be inverse-transformed back to
        interpretable RFM units later.

    Note:
        ``frequency`` and ``monetary`` are heavily right-skewed in this
        dataset (skew 12.0 and 25.6 respectively, vs. 0.9 for recency).
        
    """
    log_features = np.log1p(rfm[list(RFM_COLS)].to_numpy())
    scaler = StandardScaler()
    scaled = scaler.fit_transform(log_features)
    return scaled, scaler


def sweep_kmeans_k(
    X: np.ndarray, k_range: range = range(2, 13), random_state: int = 42
) -> pd.DataFrame:
    """Fit K-Means across a range of k and collect selection metrics.

    Args:
        X: Scaled feature array (output of ``prepare_features``).
        k_range: Values of k to try.
        random_state: Seed for reproducibility.

    Returns:
        DataFrame indexed by k with columns ``inertia``, ``silhouette``,
        ``davies_bouldin``.
    """
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        rows.append(
            {
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(X, labels),
                "davies_bouldin": davies_bouldin_score(X, labels),
            }
        )
    return pd.DataFrame(rows).set_index("k")


def choose_k(sweep: pd.DataFrame, target_range: tuple[int, int] = (6, 8)) -> int:
    """Pick the best k inside the target range by silhouette score.

    Args:
        sweep: Output of ``sweep_kmeans_k``.
        target_range: Inclusive (min, max) k values to choose from.

    Returns:
        The k in ``target_range`` with the highest silhouette score.

    Raises:
        ValueError: If no k in ``sweep`` falls inside ``target_range``.
    """
    lo, hi = target_range
    in_range = sweep.loc[lo:hi]
    if in_range.empty:
        raise ValueError(f"No k in {target_range} present in sweep index {list(sweep.index)}")
    return int(in_range["silhouette"].idxmax())


def fit_kmeans(X: np.ndarray, k: int, random_state: int = 42) -> KMeans:
    """Fit a final K-Means model at the chosen k.

    Args:
        X: Scaled feature array.
        k: Number of clusters.
        random_state: Seed for reproducibility.

    Returns:
        The fitted ``KMeans`` estimator.
    """
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    km.fit(X)
    return km


def sweep_dbscan_eps(X: np.ndarray, eps_range: np.ndarray, min_samples: int = 10) -> pd.DataFrame:
    """Fit DBSCAN across a range of eps and collect selection metrics.

    Args:
        X: Scaled feature array.
        eps_range: Array of eps values to try.
        min_samples: Fixed min_samples for every run in the sweep.

    Returns:
        DataFrame indexed by eps with columns ``n_clusters``, ``n_noise``,
        ``noise_pct``, and ``silhouette`` (computed on non-noise points
        only; NaN if fewer than 2 clusters survive).
    """
    rows = []
    for eps in eps_range:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)
        n_clusters = len(set(labels) - {-1})
        n_noise = int((labels == -1).sum())
        sil = np.nan
        if n_clusters >= 2:
            mask = labels != -1
            if mask.sum() > n_clusters:
                sil = silhouette_score(X[mask], labels[mask])
        rows.append(
            {
                "eps": eps,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_pct": n_noise / len(labels) * 100,
                "silhouette": sil,
            }
        )
    return pd.DataFrame(rows).set_index("eps")


def fit_dbscan(X: np.ndarray, eps: float, min_samples: int = 10) -> DBSCAN:
    """Fit a final DBSCAN model at the chosen eps/min_samples.

    Args:
        X: Scaled feature array.
        eps: Neighborhood radius.
        min_samples: Minimum points to form a dense region.

    Returns:
        The fitted ``DBSCAN`` estimator (labels in ``.labels_``, -1 = noise).
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(X)
    return db


def label_segments(rfm: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
    """Attach cluster labels.

    Args:
        rfm: Original (unscaled) RFM DataFrame.
        cluster_labels: Cluster assignment per row`.

    Returns:
        ``rfm`` with an added ``cluster`` column and a ``segment_name``
        column derived from each cluster's mean recency/frequency/monetary,
        ranked against the other clusters (see ``_name_clusters``). DBSCAN
        noise points (label -1) are named ``"noise"`` directly rather than
        scored, since a noise point's "profile" isn't a coherent segment.
    """
    out = rfm.copy()
    out["cluster"] = cluster_labels

    real_clusters = out[out["cluster"] != -1]
    cluster_profiles = real_clusters.groupby("cluster")[list(RFM_COLS)].mean()
    names = _name_clusters(cluster_profiles)
    if (out["cluster"] == -1).any():
        names[-1] = "noise"

    out["segment_name"] = out["cluster"].map(names)
    return out


def _name_clusters(cluster_profiles: pd.DataFrame) -> dict[int, str]:
    """Derive distinct labels for a set of clusters.

    Args:
        cluster_profiles: DataFrame indexed by cluster id, with mean
            ``recency_days``, ``frequency``, ``monetary`` columns.

    Returns:
        Dict mapping cluster id to a unique segment name.
    """
    median_recency = cluster_profiles["recency_days"].median()
    median_frequency = cluster_profiles["frequency"].median()
    median_monetary = cluster_profiles["monetary"].median()

    base_names = {}
    for cluster_id, row in cluster_profiles.iterrows():
        recent = row["recency_days"] <= median_recency
        frequent = row["frequency"] >= median_frequency
        high_value = row["monetary"] >= median_monetary
        base_names[cluster_id] = _octant_label(recent, frequent, high_value)

    final_names: dict[int, str] = {}
    for base_name, cluster_ids in _group_by_value(base_names).items():
        if len(cluster_ids) == 1:
            final_names[cluster_ids[0]] = base_name
            continue
        ranked = sorted(
            cluster_ids, key=lambda cid: cluster_profiles.loc[cid, "monetary"], reverse=True
        )
        qualifiers = ["premium", "core", "emerging", "marginal"]
        for rank, cid in enumerate(ranked):
            qualifier = qualifiers[rank] if rank < len(qualifiers) else f"tier{rank + 1}"
            final_names[cid] = f"{base_name}_{qualifier}"
    return final_names


def _group_by_value(mapping: dict[int, str]) -> dict[str, list[int]]:
    """Invert a {key: value} dict into {value: [keys]}, preserving order.

    Args:
        mapping: Dict of cluster id to base name.

    Returns:
        Dict of base name to the list of cluster ids sharing that name.
    """
    groups: dict[str, list[int]] = {}
    for key, value in mapping.items():
        groups.setdefault(value, []).append(key)
    return groups


def _octant_label(recent: bool, frequent: bool, high_value: bool) -> str:
    """Map a recency/frequency/monetary direction triple to a base label.

    Args:
        recent: True if recency is at or better than the cluster-level median.
        frequent: True if frequency is at or above the cluster-level median.
        high_value: True if monetary is at or above the cluster-level median.

    Returns:
        One of 8 base segment labels, one per octant.
    """
    if recent and frequent and high_value:
        return "champions"
    if recent and frequent and not high_value:
        return "loyal_low_spend"
    if recent and not frequent and high_value:
        return "promising_big_spenders"
    if recent and not frequent and not high_value:
        return "new_or_occasional"
    if not recent and frequent and high_value:
        return "at_risk_high_value"
    if not recent and frequent and not high_value:
        return "at_risk_low_spend"
    if not recent and not frequent and high_value:
        return "hibernating_high_value"
    return "lost"


def write_segments(segmented: pd.DataFrame, out_dir: Path = FEATURES_DIR) -> Path:
    """save the segmented customer table as csv.

    Args:
        segmented: Output of ``label_segments``.
        out_dir: Destination directory, defaults to ``data/features``.

    Returns:
        Path the table was written to.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "customer_segments.csv"
    segmented.reset_index().to_csv(path, index=False)
    return path
