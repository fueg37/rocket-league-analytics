import unittest

import pandas as pd

from analytics.chemistry import (
    build_pairwise_feature_matrix,
    build_season_chemistry_tables,
    build_trio_feature_matrix,
)


class ChemistryTests(unittest.TestCase):
    def _streams(self):
        frames = pd.DataFrame([
            {"Frame": 1, "Team": "Blue", "Player": "A", "RotationRole": "Defense", "UnderPressure": 1},
            {"Frame": 1, "Team": "Blue", "Player": "B", "RotationRole": "Attack", "UnderPressure": 1},
            {"Frame": 1, "Team": "Blue", "Player": "C", "RotationRole": "Support", "UnderPressure": 1},
            {"Frame": 2, "Team": "Blue", "Player": "A", "RotationRole": "Defense", "UnderPressure": 0},
            {"Frame": 2, "Team": "Blue", "Player": "B", "RotationRole": "Attack", "UnderPressure": 0},
            {"Frame": 2, "Team": "Blue", "Player": "C", "RotationRole": "Support", "UnderPressure": 0},
        ])
        events = pd.DataFrame([
            {"Frame": 1, "Team": "Blue", "ExpectedValue": 0.4, "FromPlayer": "A", "ToPlayer": "B", "Success": 1, "PressureRelease": 1, "Players": ["A", "B"]},
            {"Frame": 2, "Team": "Blue", "ExpectedValue": 0.3, "FromPlayer": "B", "ToPlayer": "C", "Success": 1, "PressureRelease": 0, "Players": ["B", "C"]},
            {"Frame": 1, "Team": "Blue", "ExpectedValue": 0.25, "FromPlayer": "A", "ToPlayer": "C", "Success": 0, "PressureRelease": 1, "Players": ["A", "C"]},
            {"Frame": 1, "Team": "Blue", "ExpectedValue": 0.5, "Players": ["A", "B", "C"]},
        ])
        return frames, events

    def test_pairwise_contains_shrinkage_and_ci(self):
        frames, events = self._streams()
        out = build_pairwise_feature_matrix(frames, events)
        self.assertFalse(out.empty)
        required = {"ChemistryScore", "ChemistryScore_Shrunk", "CI_Low", "CI_High", "Reliability", "ExpectedValueGain_Shrunk"}
        self.assertTrue(required.issubset(set(out.columns)))
        self.assertTrue((out["CI_High"] >= out["CI_Low"]).all())

    def test_trio_builds(self):
        frames, events = self._streams()
        out = build_trio_feature_matrix(frames, events)
        self.assertFalse(out.empty)
        self.assertIn("ChemistryScore_Shrunk", out.columns)

    def test_build_from_season_records(self):
        season = pd.DataFrame([
            {"MatchID": "1", "Team": "Blue", "Name": "A", "xG": 0.8, "xGA": 0.2, "Assists": 1, "Time_1st%": 40, "Time_2nd%": 20, "Possession": 52, "Pressure Time (s)": 22, "Avg Recovery (s)": 1.2},
            {"MatchID": "1", "Team": "Blue", "Name": "B", "xG": 0.6, "xGA": 0.3, "Assists": 0, "Time_1st%": 25, "Time_2nd%": 42, "Possession": 52, "Pressure Time (s)": 22, "Avg Recovery (s)": 1.5},
            {"MatchID": "1", "Team": "Blue", "Name": "C", "xG": 0.4, "xGA": 0.1, "Assists": 1, "Time_1st%": 20, "Time_2nd%": 28, "Possession": 52, "Pressure Time (s)": 22, "Avg Recovery (s)": 1.1},
        ])
        pair, trio = build_season_chemistry_tables(season)
        self.assertFalse(pair.empty)
        self.assertFalse(trio.empty)


if __name__ == "__main__":
    unittest.main()
