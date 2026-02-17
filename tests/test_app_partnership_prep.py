import unittest

import pandas as pd

import app


class AppPartnershipPrepIntegrationTests(unittest.TestCase):
    def test_prepare_partnership_intelligence_tables_has_complete_schema(self):
        season = pd.DataFrame([
            {"MatchID": "1", "Team": "Blue", "Name": "A", "xG": 0.8, "xGA": 0.2, "Assists": 1, "Time_1st%": 40, "Time_2nd%": 20, "Possession": 52, "Pressure Time (s)": 22, "Avg Recovery (s)": 1.2},
            {"MatchID": "1", "Team": "Blue", "Name": "B", "xG": 0.6, "xGA": 0.3, "Assists": 0, "Time_1st%": 25, "Time_2nd%": 42, "Possession": 52, "Pressure Time (s)": 22, "Avg Recovery (s)": 1.5},
            {"MatchID": "1", "Team": "Blue", "Name": "C", "xG": 0.4, "xGA": 0.1, "Assists": 1, "Time_1st%": 20, "Time_2nd%": 28, "Possession": 52, "Pressure Time (s)": 22, "Avg Recovery (s)": 1.1},
        ])

        pair, trio = app.prepare_partnership_intelligence_tables(season)

        self.assertFalse(pair.empty)
        self.assertFalse(trio.empty)

        required_pair_columns = {
            "Partnership Index",
            "Value Lift",
            "Rotation Fit",
            "Handoff Quality",
            "Pressure Escape",
            "confidence_level",
            "ci_low",
            "ci_high",
            "sample_count",
            "expected_xgd_lift_per_match",
            "win_rate_lift_points",
            "best_context_badge",
            "risk_context_badge",
            "ChemistryScore_Shrunk",
        }
        self.assertTrue(required_pair_columns.issubset(set(pair.columns)))


if __name__ == "__main__":
    unittest.main()
