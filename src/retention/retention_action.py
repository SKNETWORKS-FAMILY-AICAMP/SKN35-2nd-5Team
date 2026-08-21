"""
End-to-End Retention Action Orchestrator.
"""

from typing import Dict, Any
from src.retention.risk_type import diagnose_user_risk_type
from src.retention.llm_client import generate_retention_message_with_llm


def generate_user_retention_plan(user_features: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a comprehensive retention diagnosis and personalized message."""
    risk_diag = diagnose_user_risk_type(user_features)
    crm_message = generate_retention_message_with_llm(user_features, risk_diag)
    
    return {
        "user_id": user_features.get("user_id", "unknown"),
        "risk_diagnosis": risk_diag,
        "crm_push_message": crm_message,
    }
