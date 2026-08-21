#!/usr/bin/env python3
"""
CLI Script to compare and rank all trained ML and DL models.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.comparison.model_comparison import build_comparison_dataframe


def main():
    print("=== Model Benchmark Leaderboard ===")
    df_comp = build_comparison_dataframe()
    if df_comp.empty:
        print("[WARN] No trained models found in artifacts/results/. Please run train_ml.py or train_dl.py first.")
    else:
        print(df_comp.to_string(index=False))
        
    print("\n[OK] Evaluation ranking complete!")


if __name__ == "__main__":
    main()
