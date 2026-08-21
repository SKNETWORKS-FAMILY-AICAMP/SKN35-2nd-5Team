import unittest
from tests.fixtures import create_mock_feature_df
from src.data.validator import validate_feature_dataset


class TestDataValidator(unittest.TestCase):
    def test_validate_mock_feature_dataset(self):
        df = create_mock_feature_df(50)
        res = validate_feature_dataset(df)
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["total_rows"], 50)
        self.assertFalse(res["has_nulls"])


if __name__ == "__main__":
    unittest.main()
