#!/usr/bin/env python3
"""
CLI Script to train and evaluate all 4 Machine Learning models.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.ml.trainer import train_and_save_all_ml_models


def main():
    print("=== Training All ML Models (DT, RF, XGB, LGBM) ===")
    results = train_and_save_all_ml_models()
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
