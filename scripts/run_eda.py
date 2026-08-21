#!/usr/bin/env python3
"""
CLI Script to run comprehensive EDA and print dataset summary.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.loader import load_feature_dataset
from src.analysis.eda import compute_dataset_overview, compare_churn_groups, compute_correlations


def main():
    print("=== Running Exploratory Data Analysis ===")
    df = load_feature_dataset()
    overview = compute_dataset_overview(df)
    print("\n[Overview]")
    for k, v in overview.items():
        print(f"  - {k}: {v}")
        
    print("\n[Top Correlations with Churn]")
    corrs = compute_correlations(df)
    print(corrs.tail(10).to_string())
    print("\n[OK] EDA completed successfully!")


if __name__ == "__main__":
    main()
