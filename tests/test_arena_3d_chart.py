import pandas as pd

from charts.factory import arena_3d_shot_chart
from analytics.shot_quality import (
    SHOT_COL_PLAYER,
    SHOT_COL_RESULT,
    SHOT_COL_TEAM,
    SHOT_COL_X,
    SHOT_COL_Y,
    COL_SHOT_Z,
    COL_TARGET_X,
    COL_TARGET_Z,
    COL_XG,
    COL_XGOT,
)


def test_arena_3d_shot_chart_smoke_builds_3d_trace():
    df = pd.DataFrame(
        [
            {
                SHOT_COL_PLAYER: "P1",
                SHOT_COL_TEAM: "Blue",
                SHOT_COL_RESULT: "Shot",
                SHOT_COL_X: 120.0,
                SHOT_COL_Y: -2200.0,
                COL_SHOT_Z: 180.0,
                COL_TARGET_X: 150.0,
                COL_TARGET_Z: 320.0,
                COL_XG: 0.24,
                COL_XGOT: 0.38,
                "Frame": 100,
            }
        ]
    )

    fig = arena_3d_shot_chart(df, enable_3d=True, show_trajectories=True)

    assert len(fig.data) >= 3
    assert any(getattr(t, "type", "") == "scatter3d" for t in fig.data)


def test_arena_3d_shot_chart_supports_2d_fallback():
    df = pd.DataFrame(
        [
            {
                SHOT_COL_PLAYER: "P1",
                SHOT_COL_TEAM: "Blue",
                SHOT_COL_RESULT: "Goal",
                SHOT_COL_X: 0.0,
                SHOT_COL_Y: -3000.0,
                COL_SHOT_Z: 90.0,
                COL_TARGET_X: 50.0,
                COL_TARGET_Z: 200.0,
                COL_XG: 0.1,
                COL_XGOT: 0.2,
                "Frame": 50,
            }
        ]
    )

    fig = arena_3d_shot_chart(df, enable_3d=False)

    assert any(getattr(t, "type", "") == "scatter" for t in fig.data)
