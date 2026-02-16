"""Deterministic chart sorting rules."""

from __future__ import annotations

import pandas as pd


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower()


def sort_rank_desc(
    df: pd.DataFrame,
    metric_col: str,
    *,
    player_col: str = "Name",
    team_col: str = "Team",
) -> pd.DataFrame:
    """Sort a ranking table by descending metric with stable name/team tie-breakers."""
    rank_df = df.copy()
    rank_df[metric_col] = pd.to_numeric(rank_df[metric_col], errors="coerce").fillna(0)

    sort_cols = [metric_col]
    ascending = [False]

    if player_col in rank_df.columns:
        rank_df["_player_key"] = _normalize_text(rank_df[player_col])
        sort_cols.append("_player_key")
        ascending.append(True)
    if team_col in rank_df.columns:
        rank_df["_team_key"] = _normalize_text(rank_df[team_col])
        sort_cols.append("_team_key")
        ascending.append(True)

    sorted_df = rank_df.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)
    return sorted_df.drop(columns=[c for c in ["_player_key", "_team_key"] if c in sorted_df.columns])


def sort_time_asc(
    df: pd.DataFrame,
    time_col: str,
    *,
    player_col: str = "Name",
    team_col: str = "Team",
) -> pd.DataFrame:
    """Sort timeline data by ascending time with deterministic player/team tie-breakers."""
    time_df = df.copy()
    time_df[time_col] = pd.to_numeric(time_df[time_col], errors="coerce")
    time_df = time_df.dropna(subset=[time_col])

    sort_cols = [time_col]
    ascending = [True]

    if player_col in time_df.columns:
        time_df["_player_key"] = _normalize_text(time_df[player_col])
        sort_cols.append("_player_key")
        ascending.append(True)
    if team_col in time_df.columns:
        time_df["_team_key"] = _normalize_text(time_df[team_col])
        sort_cols.append("_team_key")
        ascending.append(True)

    sorted_df = time_df.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)
    return sorted_df.drop(columns=[c for c in ["_player_key", "_team_key"] if c in sorted_df.columns])
