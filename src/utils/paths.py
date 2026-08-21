"""
Project Paths and Directory Management.
"""

from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "ednet_payment_users.csv"
FEATURE_DATA_PATH = DATA_DIR / "churn_modeling_features.csv"
SUMMARY_DATA_PATH = DATA_DIR / "kt4_pass_expiry_repurchase_analysis.csv"
TRANSACTIONS_DATA_PATH = DATA_DIR / "kt4_payment_transactions.csv"

# Artifacts Directories
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PROCESSED_DIR = ARTIFACTS_DIR / "processed"
MODELS_DIR = ARTIFACTS_DIR / "models"
ML_MODELS_DIR = MODELS_DIR / "ml"
DL_MODELS_DIR = MODELS_DIR / "dl"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
RESULTS_DIR = ARTIFACTS_DIR / "results"

# Ensure essential artifact directories exist
for p in [PROCESSED_DIR, ML_MODELS_DIR, DL_MODELS_DIR, FIGURES_DIR, RESULTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)
