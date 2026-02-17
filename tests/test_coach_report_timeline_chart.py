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


if __name__ == "__main__":
    unittest.main()
