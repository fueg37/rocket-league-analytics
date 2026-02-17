import unittest

import numpy as np
import pandas as pd

from constants import SPEED_CANONICAL_UNIT, SPEED_DISPLAY_UNIT_DEFAULT, UU_PER_SEC_TO_MPH
from charts.factory import goal_mouth_scatter
from analytics.shot_quality import (
    COL_TARGET_X,
    COL_TARGET_Z,
    COL_XG,
    COL_XGOT,
    SHOT_COL_PLAYER,
    SHOT_COL_RESULT,
    SHOT_COL_TEAM,
)
from utils import format_speed, uu_per_sec_to_mph


class SpeedUnitsTests(unittest.TestCase):
    def test_uu_per_sec_to_mph(self):
        self.assertAlmostEqual(uu_per_sec_to_mph(2200), 2200 * UU_PER_SEC_TO_MPH)
        self.assertEqual(round(uu_per_sec_to_mph(2200), 1), 49.2)

    def test_format_speed_defaults_to_mph(self):
        self.assertEqual(format_speed(2200), "49.2 mph")

    def test_format_speed_supports_canonical_unit(self):
        self.assertEqual(format_speed(2200, unit=SPEED_CANONICAL_UNIT, precision=0), "2200 uu/s")

    def test_format_speed_handles_invalid_values(self):
        self.assertEqual(format_speed(None), "N/A")
        self.assertEqual(format_speed(float('nan')), "N/A")
        self.assertEqual(format_speed(np.nan), "N/A")
        self.assertEqual(format_speed("fast"), "N/A")
        self.assertEqual(format_speed(True), "N/A")

    def test_format_speed_precision_behavior(self):
        self.assertEqual(format_speed(2200, precision=0), "49 mph")
        self.assertEqual(format_speed(2200, precision=1), "49.2 mph")
        self.assertEqual(format_speed(2200, precision=2), "49.21 mph")

    def test_format_speed_handles_unknown_unit(self):
        self.assertEqual(format_speed(1200, unit="km/h"), "N/A")

    def test_format_speed_falls_back_when_precision_invalid(self):
        self.assertEqual(format_speed(2200, precision="x"), "49.2 mph")
        self.assertEqual(SPEED_DISPLAY_UNIT_DEFAULT, "mph")

    def test_goal_mouth_scatter_speed_display_uses_mph_only(self):
        shots = pd.DataFrame(
            [
                {
                    COL_TARGET_X: 50,
                    COL_TARGET_Z: 200,
                    SHOT_COL_PLAYER: "Blue1",
                    SHOT_COL_RESULT: "Shot",
                    SHOT_COL_TEAM: "Blue",
                    COL_XG: 0.2,
                    COL_XGOT: 0.3,
                    "Speed": 2200,
                }
            ]
        )

        fig = goal_mouth_scatter(shots, include_xgot=False, on_target_only=False)
        speed_display = fig.data[0].customdata[0][6]

        self.assertIn("mph", speed_display)
        self.assertNotIn("uu/s", speed_display)


if __name__ == "__main__":
    unittest.main()
