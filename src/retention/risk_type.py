"""
Churn Risk Typology Classification.
"""

from typing import Dict, Any
from src.utils.constants import RISK_TYPES


def diagnose_user_risk_type(user_features: Dict[str, Any]) -> Dict[str, str]:
    """
    Diagnose a customer's specific churn risk pattern.
    """
    decay_ratio = float(user_features.get("obs_decay_ratio", 1.0))
    recency_days = float(user_features.get("obs_recency_days", 0.0))
    solve_count = float(user_features.get("obs_solve_count", 0.0))
    active_days = float(user_features.get("obs_active_days", 0.0))
    
    if decay_ratio < 0.1 and active_days >= 2:
        risk_code = "ENGAGEMENT_DROP"
        desc = "1주차 대비 2주차 학습량이 90% 이상 급감하여 활동 소멸 위험이 매우 높습니다."
        action = "학습 독려 리마인드 푸시 및 가벼운 5분 복습 챌린지 제공"
    elif recency_days >= 7.0:
        risk_code = "DORMANT_USER"
        desc = f"최근 {recency_days:.1f}일간 접속이 없어 휴면 이탈 상태에 진입했습니다."
        action = "복귀 환영 쿠폰(포인트) 지급 및 미완료 문제 이어풀기 제안"
    elif solve_count < 10.0:
        risk_code = "LOW_ACHIEVEMENT"
        desc = "초기 14일간 문제 풀이량이 10회 미만으로 서비스 정착에 실패했습니다."
        action = "수준별 기초 진단 가이드 및 1:1 맞춤형 추천 학습 로드맵 발송"
    else:
        risk_code = "GENERAL_RISK"
        desc = "일반적인 이용 패턴 감퇴 징후가 관측되었습니다."
        action = "정기 학습 진도 알림 및 프리미엄 혜택 리마인드"
        
    return {
        "risk_code": risk_code,
        "risk_title": RISK_TYPES.get(risk_code, "일반 이탈 위험"),
        "description": desc,
        "recommended_action": action,
    }
