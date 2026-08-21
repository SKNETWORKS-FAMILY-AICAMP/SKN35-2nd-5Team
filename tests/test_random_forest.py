import unittest

import pandas as pd

from src.ml.random_forest import build_pipeline, tune_hyperparameters


class TestAdvancedRandomForest(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(
            {
                "activity": [1, 2, None, 4, 5, 6, 7, 8, 9, 10],
                "recency": [14, 12, 11, 9, 8, 6, 5, 3, 2, 0],
                "product": ["A", "A", "B", "B", None, "A", "B", "A", "B", "A"],
            }
        )
        self.y = pd.Series([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    def test_pipeline_handles_missing_and_unseen_category(self):
        pipeline = build_pipeline(
            num_features=["activity", "recency"],
            cat_features=["product"],
            params={"n_estimators": 10, "max_depth": 3, "n_jobs": 1},
        )
        pipeline.fit(self.X, self.y)
        new_data = pd.DataFrame(
            {"activity": [None], "recency": [4], "product": ["NEW_PRODUCT"]}
        )
        prediction = pipeline.predict(new_data)
        self.assertEqual(prediction.shape, (1,))

    def test_tuning_rejects_empty_feature_lists(self):
        with self.assertRaises(ValueError):
            tune_hyperparameters(
                self.X,
                self.y,
                num_features=[],
                cat_features=[],
                n_trials=1,
                cv_splits=2,
            )


if __name__ == "__main__":
    unittest.main()
