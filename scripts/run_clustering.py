"""Run the optional K-Means employee segmentation baseline."""

import argparse
import json

from src.load_data.loader import load_train_data
from src.ml.clustering import run_kmeans_clustering


def main() -> None:
    parser = argparse.ArgumentParser(description="직원 K-Means 군집화")
    parser.add_argument("--clusters", type=int, default=4)
    args = parser.parse_args()
    _, metrics = run_kmeans_clustering(load_train_data(), n_clusters=args.clusters)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
