import unittest

import pandas as pd

from charts.factory import coach_report_timeline_chart


class CoachReportTimelineChartTests(unittest.TestCase):
    def test_chart_renders_expected_trace_contract(self):
        win_prob_df = pd.DataFrame(
            {
                "Time": [0, 10, 20, 30],
                "WinProb": [48.0, 52.0, 61.0, 58.0],
            }
        )
        momentum_series = pd.Series([0.1, -0.05, 0.18, 0.0], index=[0, 10, 20, 30])
        coach_report_df = pd.DataFrame(
            {
                "Time": [8.0, 22.0],
                "RecommendedAction": ["challenge_now", "rotate_backpost"],
                "Role": ["first man", "third man"],
                "ExpectedSwingP10": [0.01, -0.03],
                "ExpectedSwingP90": [0.12, 0.04],
                "MissedSwing": [0.08, -0.05],
                "Confidence": [0.77, 0.64],
                "ClipKey": ["frame:240|window:180-300", "frame:660|window:600-720"],
            }
        )

        fig = coach_report_timeline_chart(win_prob_df, momentum_series, coach_report_df)

        self.assertEqual(len(fig.data), 3)
        self.assertEqual(fig.data[0].name, "Win probability")
        self.assertEqual(fig.data[1].name, "Momentum")
        self.assertEqual(fig.data[2].name, "Missed opportunities")
        self.assertEqual(len(fig.data[2].x), len(coach_report_df))
        self.assertEqual(len(fig.data[2].customdata[0]), 6)


    def test_chart_accepts_scalar_interval_fallback(self):
        win_prob_df = pd.DataFrame({"Time": [0, 10], "WinProb": [50.0, 51.0]})
        momentum_series = pd.Series([0.02, -0.01], index=[0, 10])
        coach_report_df = pd.DataFrame(
            {
                "Time": [5.0, 9.0],
                "RecommendedAction": ["challenge_now", "shadow_defend"],
                "Role": ["first man", "second man"],
                "ExpectedSwingP10": 0.01,
                "ExpectedSwingP90": 0.08,
                "MissedSwing": [0.05, -0.01],
                "Confidence": [0.8, 0.6],
            }
        )

        fig = coach_report_timeline_chart(win_prob_df, momentum_series, coach_report_df)
        self.assertEqual(fig.data[2].customdata[0][2], "[0.01, 0.08]")
        self.assertEqual(fig.data[2].customdata[1][2], "[0.01, 0.08]")

    def test_chart_handles_missing_interval_columns_without_crashing(self):
        win_prob_df = pd.DataFrame({"Time": [0, 10], "WinProb": [49.0, 52.0]})
        momentum_series = pd.Series([0.0, 0.04], index=[0, 10])
        coach_report_df = pd.DataFrame(
            {
                "Time": [6.0],
                "RecommendedAction": ["rotate_backpost"],
                "Role": ["third man"],
                "MissedSwing": [0.03],
                "Confidence": [0.72],
            }
        )

        fig = coach_report_timeline_chart(win_prob_df, momentum_series, coach_report_df)
        self.assertEqual(fig.data[2].name, "Missed opportunities")
        self.assertEqual(fig.data[2].customdata[0][2], "n/a")


if __name__ == "__main__":
    unittest.main()
