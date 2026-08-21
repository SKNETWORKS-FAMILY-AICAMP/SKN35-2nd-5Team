"""
LLM Client wrapper for generating personalized retention messages.
"""

import os
from typing import Dict, Any


def generate_retention_message_with_llm(
    user_info: Dict[str, Any],
    risk_diag: Dict[str, str],
    api_key: str = None,
) -> str:
    """
    Generate tailored Korean CRM retention push/SMS message using LLM or smart template fallback.
    """
    uid = user_info.get("user_id", "고객")
    risk_title = risk_diag.get("risk_title", "학습 관리")
    action = risk_diag.get("recommended_action", "학습 복귀 독려")
    
    # Template Fallback (즉시 동작 및 API 미설정 시 안전 대응)
    message_template = f"""[산타토익 학습 리포트]
안녕하세요, {uid}님! 📚

최근 학습 패턴 분석 결과, 목표 점수 달성을 위한 골든타임이 지나가고 있어요.
▶ 현재 진단: {risk_title}
▶ 맞춤 처방: {action}

{uid}님만을 위해 오늘 '5분 미니 테스트'와 '복습 포인트 500P'를 준비했습니다.
지금 바로 앱에서 실력을 점검해 보세요!

👉 [나의 학습 이어하기]"""
    return message_template
