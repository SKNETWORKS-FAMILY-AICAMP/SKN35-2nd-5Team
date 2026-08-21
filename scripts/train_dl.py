#!/usr/bin/env python3
"""
CLI Script to train and evaluate Deep Learning (MLP) model.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dl.trainer import train_and_save_dl_model


def main():
    print("=== Training Deep Learning (MLP) Model ===")
    results = train_and_save_dl_model()
    for m_id, m_metrics in results.items():
        print(f"\n[{m_metrics.get('model_name', m_id)}]")
        print(f"  - Train Acc: {m_metrics.get('train_accuracy', 0):.4f}")
        print(f"  - Test Acc:  {m_metrics.get('test_accuracy', 0):.4f}")
        print(f"  - F1-Score:  {m_metrics.get('f1_score', 0):.4f}")
        print(f"  - ROC-AUC:   {m_metrics.get('roc_auc', 0):.4f}")
        print(f"  - Train Time:{m_metrics.get('train_time_sec', 0):.2f}s")
    print("\n[OK] DL model trained and saved to artifacts/models/dl/ and artifacts/results/!")


if __name__ == "__main__":
    main()
