"""Season-level possession value reporting adapters."""

from __future__ import annotations

import pandas as pd


def build_player_value_reports(actions_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate action-level transition gains into season player profiles."""
    if actions_df is None or actions_df.empty:
        return pd.DataFrame(
            columns=[
                "Name",
                "Team",
                "ProgressiveValue",
                "DefensiveValuePrevention",
                "NeutralizationValue",
                "HighLeverageValue",
                "Total_VAEP",
                "Avg_VAEP",
            ]
        )

    df = actions_df.copy()
    if "ValueDelta_3s" not in df.columns:
        df["ValueDelta_3s"] = 0.0
    if "ContextWeight" not in df.columns:
        df["ContextWeight"] = 1.0

    event_type = df.get("EventType", pd.Series(["touch"] * len(df), index=df.index)).astype(str).str.lower()
    defensive_mask = event_type.str.contains("save|clear|block|challenge")
    neutral_mask = event_type.str.contains("50|challenge|neutral")
    progressive_mask = event_type.str.contains("touch|pass|shot|dribble|carry")

    df["ProgressiveValue"] = df["ValueDelta_3s"].where(progressive_mask, 0.0)
    df["DefensiveValuePrevention"] = (-df["ValueDelta_3s"]).where(defensive_mask & (df["ValueDelta_3s"] < 0), 0.0)
    df["NeutralizationValue"] = df["ValueDelta_3s"].abs().where(neutral_mask, 0.0)
    df["HighLeverageValue"] = df["ValueDelta_3s"] * pd.to_numeric(df["ContextWeight"], errors="coerce").fillna(1.0)

    out = (
        df.groupby(["Player", "Team"], as_index=False)
        .agg(
            ProgressiveValue=("ProgressiveValue", "sum"),
            DefensiveValuePrevention=("DefensiveValuePrevention", "sum"),
            NeutralizationValue=("NeutralizationValue", "sum"),
            HighLeverageValue=("HighLeverageValue", "sum"),
            Total_VAEP=("ValueDelta_3s", "sum"),
            Avg_VAEP=("ValueDelta_3s", "mean"),
        )
        .rename(columns={"Player": "Name"})
    )

    for col in [
        "ProgressiveValue",
        "DefensiveValuePrevention",
        "NeutralizationValue",
        "HighLeverageValue",
        "Total_VAEP",
        "Avg_VAEP",
    ]:
        out[col] = out[col].round(4)
    return out
