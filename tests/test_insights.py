import pandas as pd

from analytics.insights import build_key_insight
from analytics.shot_quality import COL_XG


def test_build_key_insight_prefers_top_scoring_narrative_with_deterministic_tie_break():
    df = pd.DataFrame([
        {"Goals": 4, "Shots": 12, "Saves": 5, "Shots Faced": 9, "xGA": 1.3},
        {"Goals": 0, "Shots": 0, "Saves": 3, "Shots Faced": 5, "xGA": 0.8},
    ])
    shot_df = pd.DataFrame({COL_XG: [0.2, 0.3, 0.4, 0.5]})
    momentum = pd.Series([0, 1, 1, 2, 3, 4, 5, 6], index=[0, 10, 20, 30, 40, 50, 60, 70])
    coach_report_df = pd.DataFrame({"MissedSwing": [-0.3, -0.2, 0.1]})

    insight = build_key_insight(df, shot_df, momentum, coach_report_df)

    assert insight["headline"] == "Clinical finishing outpaced shot quality"
    assert insight["confidence_source_tag"] == "high • xG_delta"


def test_build_key_insight_falls_back_once_when_metrics_missing():
    insight = build_key_insight(pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame())

    assert insight["headline"] == "Insufficient data for a targeted key insight"
    assert insight["confidence_source_tag"] == "low • fallback_missing_metrics"
