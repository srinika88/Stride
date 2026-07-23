"""
StrideIQ — Clustering (Drift-Onset Detection)
K-Means across a single run's feature windows, in time order, to find
where the dominant cluster assignment shifts — this is the flagship
"when does form change during a run" capability that snapshot-based
tools structurally can't do. Unsupervised, so it needs no labels —
usable even before the classifier has enough labeled data to train on.
"""

import logging

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from backend import config 
from backend.database_manager import DatabaseManager
from ml.features import FEATURE_COLUMNS

logger = logging.getLogger("strideiq.ml.clustering")


def cluster_run(run_id: int, db: DatabaseManager, n_clusters: int = None) -> list[dict]:
    """
    Returns a list of {window_start, cluster_id, is_drift_point} dicts,
    one per window, in time order. is_drift_point marks windows where
    the cluster assignment changed from the previous window — the
    actual "form started changing here" signal.
    """
    n_clusters = n_clusters or config.N_CLUSTERS

    features = db.get_features(run_id)
    if not features:
        logger.warning(f"No feature rows for run {run_id} — cannot cluster")
        return []

    rows = [dict(f) for f in features]
    rows.sort(key=lambda r: r["window_start"])  # must be in time order for drift detection to mean anything

    if len(rows) < n_clusters:
        logger.warning(
            f"Run {run_id} has only {len(rows)} windows, fewer than "
            f"n_clusters={n_clusters} — reducing cluster count to match."
        )
        n_clusters = max(1, len(rows))

    X = np.array([[r.get(col) for col in FEATURE_COLUMNS] for r in rows], dtype=float)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(X_scaled)

    results = []
    prev_cluster = None
    for row, cluster_id in zip(rows, cluster_ids):
        is_drift = prev_cluster is not None and cluster_id != prev_cluster
        results.append({
            "run_id": run_id,
            "window_start": row["window_start"],
            "cluster_id": int(cluster_id),
            "is_drift_point": is_drift,
        })
        prev_cluster = cluster_id

    n_drift_points = sum(r["is_drift_point"] for r in results)
    logger.info(f"Run {run_id}: {n_clusters} clusters, {n_drift_points} drift transition(s) detected")

    return results


def cluster_and_store(run_id: int, db: DatabaseManager, n_clusters: int = None):
    results = cluster_run(run_id, db, n_clusters)
    if results:
        db.insert_cluster_results(results)
    return results


if __name__ == "__main__":
    db = DatabaseManager()
    runs = db.get_all_runs()
    if not runs:
        print("No runs found.")
    else:
        latest_run_id = runs[0]["run_id"]
        results = cluster_and_store(latest_run_id, db)
        drift_points = [r for r in results if r["is_drift_point"]]
        print(f"Found {len(drift_points)} drift point(s) in run {latest_run_id}:")
        for d in drift_points:
            print(f"  Window starting at {d['window_start']}: shifted to cluster {d['cluster_id']}") 



