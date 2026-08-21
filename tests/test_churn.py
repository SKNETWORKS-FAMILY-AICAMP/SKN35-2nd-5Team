import unittest
from src.features.churn import classify_user_churn_status


class TestChurn(unittest.TestCase):
    def test_classify_refund_churn(self):
        res = classify_user_churn_status(has_pay=True, pay_count=1, has_refund=True)
        self.assertEqual(res["is_refund_churn"], 1)
        self.assertEqual(res["is_churn"], 1)

    def test_classify_non_renewal_churn(self):
        res = classify_user_churn_status(has_pay=True, pay_count=1, has_refund=False)
        self.assertEqual(res["is_non_renewal_churn"], 1)
        self.assertEqual(res["is_churn"], 1)

    def test_classify_retained_user(self):
        res = classify_user_churn_status(has_pay=True, pay_count=2, has_refund=False)
        self.assertEqual(res["is_non_renewal_churn"], 0)
        self.assertEqual(res["is_refund_churn"], 0)
        self.assertEqual(res["is_churn"], 0)


if __name__ == "__main__":
    unittest.main()
