import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from analytics.save_metrics import (
    SaveFeatures,
    aggregate_save_summary,
    build_save_touch_index,
    resolve_saver_for_shot,
    score_save,
)
from analytics.shot_quality import SHOT_COL_FRAME, SHOT_COL_PLAYER, SHOT_COL_RESULT, SHOT_COL_TEAM, SHOT_COL_X, SHOT_COL_Y


class SaveMetricsTests(unittest.TestCase):
    def _proto(self):
        players = [
            SimpleNamespace(name="Blue1", is_orange=False),
            SimpleNamespace(name="Orange1", is_orange=True),
        ]
        return SimpleNamespace(players=players, game_stats=SimpleNamespace(hits=[]))

    def test_score_bounds(self):
        low = SaveFeatures(shot_speed=800, dist_to_goal=4000, angle_off_center=0.0, shot_z=50, saver_dist=200)
        high = SaveFeatures(shot_speed=2600, dist_to_goal=1200, angle_off_center=1.0, shot_z=700, saver_dist=1500)
        low_score = score_save(low)
        high_score = score_save(high)
        self.assertTrue(0.0 <= low_score.save_difficulty_index <= 1.0)
        self.assertTrue(0.0 <= high_score.save_difficulty_index <= 1.0)
        self.assertGreater(high_score.save_difficulty_index, low_score.save_difficulty_index)

    def test_touch_index_and_resolver_prefers_explicit_save(self):
        hit = SimpleNamespace(frame_number=300, player_id=SimpleNamespace(id="2"), is_save=True)
        proto = SimpleNamespace(game_stats=SimpleNamespace(hits=[hit]))
        player_map = {"2": "Orange1"}
        pid_team = {"2": "Orange"}
        idx = build_save_touch_index(proto, player_map, pid_team)

        player_pos = {
            "Orange1": {"team": "Orange", "frames": np.array([300]), "x": np.array([100]), "y": np.array([100])}
        }
        saver, _, source, confidence = resolve_saver_for_shot(
            frame=301,
            defending_team="Orange",
            shot_x=0,
            shot_y=0,
            player_pos=player_pos,
            save_touch_index=idx,
        )
        self.assertEqual(saver, "Orange1")
        self.assertEqual(source, "explicit_save_touch")
        self.assertEqual(confidence, 1.0)

    def test_aggregate_summary_and_aliases(self):
        proto = self._proto()
        events = pd.DataFrame([
            {
                "Saver": "Blue1",
                "Team": "Blue",
                "Frame": 100,
                "Time": 3.3,
                "Shooter": "Orange1",
                "AttributionSource": "explicit_save_touch",
                "AttributionConfidence": 1.0,
                "ShotSpeed": 2000,
                "DistToGoal": 1200,
                "AngleOffCenter": 0.7,
                "ShotHeight": 300,
                "SaverDist": 600,
                "SaveDifficultyIndex": 0.6,
                "ExpectedSaveProb": 0.4,
                "SaveImpact": 0.6,
            }
        ])
        summary = aggregate_save_summary(proto, events)
        row = summary[summary["Name"] == "Blue1"].iloc[0]
        self.assertEqual(row["SaveEvents"], 1)
        self.assertAlmostEqual(row["Total_SaveImpact"], 0.6)
        self.assertEqual(row["Saves_Nearby"], 1)
        self.assertAlmostEqual(row["Total_xS"], row["Total_SaveDifficulty"])


if __name__ == "__main__":
    unittest.main()
