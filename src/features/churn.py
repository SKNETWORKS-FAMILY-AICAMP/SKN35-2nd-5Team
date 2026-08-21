"""
Churn label definition and classification logic.
"""

from typing import Dict, Any


def classify_user_churn_status(
    has_pay: bool,
    pay_count: int,
    has_refund: bool,
) -> Dict[str, int]:
    """
    Classify a customer's churn status based on payment and refund history.
    
    - is_refund_churn: Refund requested (1)
    - is_non_renewal_churn: Paid once, no refund, but did not renew/repurchase (1)
    - is_churn: Either refund churn or non-renewal churn (1) vs Retained/Repurchased (0)
    """
    if not has_pay:
        return {
            "is_refund_churn": 0,
            "is_non_renewal_churn": 0,
            "is_churn": 0,
        }
        
    is_refund_churn = 1 if has_refund else 0
    is_repurchase = 1 if pay_count >= 2 else 0
    is_non_renewal_churn = 1 if (not is_repurchase and not has_refund) else 0
    is_churn = 1 if (is_refund_churn or is_non_renewal_churn) else 0
    
    return {
        "is_refund_churn": is_refund_churn,
        "is_non_renewal_churn": is_non_renewal_churn,
        "is_churn": is_churn,
    }
