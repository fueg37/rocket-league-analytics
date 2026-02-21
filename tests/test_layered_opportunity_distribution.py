import unittest

import pandas as pd

import app


class LayeredOpportunityDistributionTests(unittest.TestCase):
    def test_primary_layer_prefers_coach_report_schema(self):
        coach_report_df = pd.DataFrame(
            {
                "Time": [12.0],
                "Role": ["first man"],
                "RecommendedAction": ["challenge_now"],
                "MissedSwing": [0.12],
            }
        )

        result = app._build_layered_opportunity_distribution(
            coach_report_df,
            replay_states=pd.DataFrame({"Frame": [100]}),
            momentum_series=pd.Series([0.2], index=[12.0]),
            win_prob_df=pd.DataFrame({"Time": [12.0], "WinProb": [55.0]}),
            rotation_timeline=pd.DataFrame({"Frame": [100], "Team": ["Blue"], "Role": ["1st"]}),
        )

        self.assertFalse(result.empty)
        self.assertEqual(result.iloc[0]["Layer"], "Primary")
        self.assertEqual(result.iloc[0]["RecommendedAction"], "challenge_now")

    def test_secondary_layer_falls_back_to_decision_moments(self):
        result = app._build_layered_opportunity_distribution(
            coach_report_df=pd.DataFrame(),
            replay_states=pd.DataFrame({"Frame": [150, 300]}),
            momentum_series=pd.Series([0.1, -0.2, 0.3], index=[5.0, 10.0, 15.0]),
            win_prob_df=pd.DataFrame({"Time": [5.0, 10.0, 15.0], "WinProb": [48.0, 50.0, 54.0]}),
            rotation_timeline=pd.DataFrame({"Frame": [150, 300], "Team": ["Blue", "Blue"], "Role": ["2nd", "3rd"]}),
        )

        self.assertFalse(result.empty)
        self.assertIn("Layer", result.columns)
        self.assertEqual(result.iloc[0]["Layer"], "Secondary")
        self.assertTrue({"Role", "RecommendedAction", "MissedSwing"}.issubset(set(result.columns)))


if __name__ == "__main__":
    unittest.main()
