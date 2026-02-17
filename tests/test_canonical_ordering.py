import pandas as pd

from constants import GAME_STATE_ORDER, KICKOFF_SPAWN_ORDER
from utils import apply_categorical_order, stable_sort


def test_kickoff_spawn_order_is_canonical():
    kickoff_df = pd.DataFrame(
        {
            "Player": ["B", "A", "C"],
            "Spawn": ["Diagonal", "Center", "Off-Center"],
            "Time to Hit": [1.4, 1.1, 1.2],
        }
    )

    ordered = apply_categorical_order(kickoff_df, "Spawn", KICKOFF_SPAWN_ORDER)
    ordered = stable_sort(ordered, by=["Spawn", "Player"], ascending=[True, True])

    assert ordered["Spawn"].astype(str).tolist() == ["Center", "Off-Center", "Diagonal"]


def test_situational_state_order_is_canonical():
    situational_df = pd.DataFrame(
        {
            "State": ["Trailing", "Tied", "Leading"],
            "Goals": [3, 2, 4],
        }
    )

    ordered = apply_categorical_order(situational_df, "State", GAME_STATE_ORDER)
    ordered = stable_sort(ordered, by=["State"], ascending=[True])

    assert ordered["State"].astype(str).tolist() == ["Leading", "Tied", "Trailing"]
