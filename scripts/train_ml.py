#!/usr/bin/env python3
"""
CLI Script to train and evaluate all 4 Machine Learning models.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.ml.trainer import train_and_save_all_ml_models


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rf-trials",
        type=int,
        default=20,
        help="Number of Optuna trials for Random Forest (default: 20)",
    )
    parser.add_argument(
        "--rf-cv-splits",
        type=int,
        default=5,
        help="Stratified CV folds for Random Forest tuning (default: 5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=== Training All ML Models (DT, RF, XGB, LGBM) ===")
    print(
        f"Random Forest tuning: {args.rf_trials} Optuna trials x "
        f"{args.rf_cv_splits}-fold CV"
    )
    results = train_and_save_all_ml_models(
        random_forest_trials=args.rf_trials,
        random_forest_cv_splits=args.rf_cv_splits,
    )
    for m_id, m_metrics in results.items():
        print(f"\n[{m_metrics.get('model_name', m_id)}]")
        print(f"  - Train Acc: {m_metrics.get('train_accuracy', 0):.4f}")
        print(f"  - Test Acc:  {m_metrics.get('test_accuracy', 0):.4f}")
        print(f"  - F1-Score:  {m_metrics.get('f1_score', 0):.4f}")
        print(f"  - ROC-AUC:   {m_metrics.get('roc_auc', 0):.4f}")
        print(f"  - Train Time:{m_metrics.get('train_time_sec', 0):.2f}s")
    print("\n[OK] All ML models trained and saved to artifacts/models/ml/ and artifacts/results/!")


if __name__ == "__main__":
    main()
