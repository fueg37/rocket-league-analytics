import unittest

import pandas as pd

from analytics.chemistry import (
    add_chemistry_explanations,
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
        required = {
            "ChemistryScore",
            "ChemistryScore_Shrunk",
            "CI_Low",
            "CI_High",
            "Reliability",
            "ExpectedValueGain_Shrunk",
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

            "ExpectedValueGain_ContributionPct",
            "RotationalComplementarity_ContributionPct",
            "PossessionHandoffEfficiency_ContributionPct",
            "PressureReleaseReliability_ContributionPct",
            "context_score_leading",
            "context_score_tied",
            "context_score_trailing",
            "context_score_defensive_third",
            "context_score_offensive_third",
            "context_score_high_pressure",
            "primary_driver_label",
            "secondary_driver_label",
            "best_context_label",
            "risk_context_label",
            "primary_driver_explanation",
            "context_usage_explanation",
        }
        self.assertTrue(required.issubset(set(out.columns)))
        self.assertTrue((out["CI_High"] >= out["CI_Low"]).all())


    def test_contract_scaling_bounds_and_confidence_labels(self):
        frames, events = self._streams()
        out = build_pairwise_feature_matrix(frames, events)
        self.assertTrue(((out["Partnership Index"] >= 0) & (out["Partnership Index"] <= 100)).all())
        for col in ["Value Lift", "Rotation Fit", "Handoff Quality", "Pressure Escape"]:
            self.assertTrue(((out[col] >= 0) & (out[col] <= 100)).all())
        self.assertTrue(set(out["confidence_level"].unique()).issubset({"Low", "Medium", "High"}))
        self.assertTrue((out["ci_high"] >= out["ci_low"]).all())

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


    def test_contribution_percentages_sum_to_100(self):
        frames, events = self._streams()
        out = build_pairwise_feature_matrix(frames, events)
        pct_cols = [
            "ExpectedValueGain_ContributionPct",
            "RotationalComplementarity_ContributionPct",
            "PossessionHandoffEfficiency_ContributionPct",
            "PressureReleaseReliability_ContributionPct",
        ]
        total = out[pct_cols].sum(axis=1).round(6)
        self.assertTrue((total == 100.0).all())

    def test_low_confidence_explanations_downgrade_certainty(self):
        df = pd.DataFrame([
            {
                "primary_driver_label": "Pressure Release",
                "best_context_label": "Trailing game states",
                "sample_count": 2,
                "ci_low": 40.0,
                "ci_high": 80.0,
            }
        ])
        out = add_chemistry_explanations(df)
        self.assertIn("Shows signs", out.loc[0, "primary_driver_explanation"])
        self.assertIn("May be", out.loc[0, "context_usage_explanation"])


if __name__ == "__main__":
    unittest.main()
