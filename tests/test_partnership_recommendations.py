import unittest

import pandas as pd

from analytics.partnership_recommendations import build_pair_recommendations


class PartnershipRecommendationTests(unittest.TestCase):
    def test_recommendations_follow_deterministic_contract_ordering(self):
        pairs = pd.DataFrame([
            {"Player1": "A", "Player2": "B", "Partnership Index": 81, "expected_xgd_lift_per_match": 0.22, "confidence_level": "High", "sample_count": 8, "Pressure Escape": 70, "Rotation Fit": 68, "primary_driver_label": "Value Lift"},
            {"Player1": "A", "Player2": "C", "Partnership Index": 78, "expected_xgd_lift_per_match": 0.27, "confidence_level": "Medium", "sample_count": 6, "Pressure Escape": 58, "Rotation Fit": 60, "primary_driver_label": "Handoff Quality"},
            {"Player1": "B", "Player2": "C", "Partnership Index": 79, "expected_xgd_lift_per_match": 0.18, "confidence_level": "High", "sample_count": 7, "Pressure Escape": 83, "Rotation Fit": 80, "primary_driver_label": "Pressure Escape"},
            {"Player1": "A", "Player2": "D", "Partnership Index": 88, "expected_xgd_lift_per_match": 0.35, "confidence_level": "Low", "sample_count": 2, "Pressure Escape": 62, "Rotation Fit": 59, "primary_driver_label": "Value Lift"},
        ])

        recs = build_pair_recommendations(pairs, min_samples=4)

        self.assertEqual(recs["aggressive"]["label"], "A + C")
        self.assertEqual(recs["stabilizer"]["label"], "B + C")
        self.assertEqual(recs["high_confidence"]["label"], "A + B")
        self.assertEqual(recs["watchlist"]["label"], "A + D")


if __name__ == "__main__":
    unittest.main()
