"""Optional employee-segmentation baseline used by the CLI scaffold."""

import json
from typing import Any

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.load_data.loader import split_features_target
from src.ml.preprocessing import build_preprocessor
from src.utils.constants import ID_COLUMN, RANDOM_STATE
from src.utils.paths import REPORTS_DIR, ensure_artifact_dirs


def run_kmeans_clustering(
    frame: pd.DataFrame,
    *,
    n_clusters: int = 4,
    random_state: int = RANDOM_STATE,
    save_artifacts: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if n_clusters < 2:
        raise ValueError("클러스터 수는 2 이상이어야 합니다.")
    features, _ = split_features_target(frame)
    preprocessor = build_preprocessor(features, dense_output=True)
    transformed = preprocessor.fit_transform(features)
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = model.fit_predict(transformed)
    sample_size = min(5000, len(frame))
    metrics = {
        "n_clusters": n_clusters,
        "inertia": float(model.inertia_),
        "silhouette": float(
            silhouette_score(
                transformed,
                labels,
                sample_size=sample_size,
                random_state=random_state,
            )
        ),
    }
    assignments = pd.DataFrame({ID_COLUMN: frame[ID_COLUMN], "cluster": labels})
    if save_artifacts:
        ensure_artifact_dirs()
        assignments.to_csv(REPORTS_DIR / "cluster_assignments.csv", index=False)
        (REPORTS_DIR / "clustering_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return assignments, metrics

