import unittest
from src.features.feature_engineering import extract_user_observation_features


class TestFeatures(unittest.TestCase):
    def test_extract_user_observation_features(self):
        user_rows = [
            {"timestamp": 1000, "action_type": "respond"},
            {"timestamp": 1000 + 86400000 * 2, "action_type": "submit"},
            {"timestamp": 1000 + 86400000 * 8, "action_type": "play_video"},
        ]
        features = extract_user_observation_features(user_rows, first_pay_ts=1000, obs_window_days=14)
        self.assertEqual(features["obs_total_events"], 3)
        self.assertEqual(features["obs_solve_count"], 1)
        self.assertEqual(features["obs_submit_count"], 1)
        self.assertEqual(features["obs_play_video_count"], 1)
        self.assertEqual(features["obs_active_days"], 3)


if __name__ == "__main__":
    unittest.main()
