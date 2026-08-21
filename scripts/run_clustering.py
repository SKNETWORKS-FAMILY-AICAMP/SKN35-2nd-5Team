#!/usr/bin/env python3
"""
CLI Script to run K-Means customer behavior clustering.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.loader import load_feature_dataset
from src.clustering.kmeans import perform_kmeans_clustering, get_cluster_summary


def main():
    print("=== Running K-Means User Clustering ===")
    df = load_feature_dataset()
    df_clustered, kmeans, scaler = perform_kmeans_clustering(df, n_clusters=4)
    summary = get_cluster_summary(df_clustered)
    print("\n[Cluster Profiles & Churn Rates]")
    print(summary.to_string())
    print("\n[OK] Clustering complete!")


if __name__ == "__main__":
    main()
