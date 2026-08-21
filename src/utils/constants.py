"""
Global Project Constants and Hyperparameters.
"""

# Random Seed
RANDOM_STATE = 42

# Data & Churn Settings
TARGET_COLUMN = "is_churn"
OBSERVATION_WINDOW_DAYS = 14
PREDICTION_HORIZON_DAYS = 30

# Feature engineering columns to exclude from training
LEAKAGE_AND_ID_COLUMNS = [
    "user_id",
    "first_pay_ts",
    "first_pay_item",
    "is_refund_churn",
    "is_non_renewal_churn",
    "is_churn",
]

# Risk Types for Retention
RISK_TYPES = {
    "ENGAGEMENT_DROP": "2주차 학습량 급감형 (활동 소멸 위험)",
    "DORMANT_USER": "최근 장기 미접속형 (휴면 이탈 위험)",
    "LOW_ACHIEVEMENT": "문제 풀이 저조/어려움 호소형 (학습 정체 위험)",
    "REFUND_PRONE": "초기 불만/환불 위험형",
}
