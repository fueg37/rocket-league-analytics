import unittest

import numpy as np

from constants import SPEED_CANONICAL_UNIT, SPEED_DISPLAY_UNIT_DEFAULT, UU_PER_SEC_TO_MPH
from utils import format_speed, uu_per_sec_to_mph


class SpeedUnitsTests(unittest.TestCase):
    def test_uu_per_sec_to_mph(self):
        self.assertAlmostEqual(uu_per_sec_to_mph(2200), 2200 * UU_PER_SEC_TO_MPH)

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

    def test_format_speed_handles_unknown_unit(self):
        self.assertEqual(format_speed(1200, unit="km/h"), "N/A")

    def test_format_speed_falls_back_when_precision_invalid(self):
        self.assertEqual(format_speed(2200, precision="x"), "49.2 mph")
        self.assertEqual(SPEED_DISPLAY_UNIT_DEFAULT, "mph")


if __name__ == "__main__":
    unittest.main()
