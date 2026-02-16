from __future__ import annotations

import pandas as pd

from analytics.schema import SCHEMA_VERSION


EVENT_V3_DEFAULTS = {
    "outcome_type": "",
    "is_on_target": False,
    "is_big_chance": False,
    "speed": 0.0,
    "pressure_context": "unknown",
    "nearest_defender_distance": pd.NA,
    "shot_angle": pd.NA,
    "shooter_boost": pd.NA,
    "distance_to_goal": pd.NA,
    "xg": 0.0,
    "xa": 0.0,
}


def _apply_schema_version(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "schema_version" not in out.columns:
        out["schema_version"] = 1
    out["schema_version"] = pd.to_numeric(out["schema_version"], errors="coerce").fillna(1).astype(int)
    return out


def _migrate_stats_v1_to_v2(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "MatchID" in out.columns:
        out["MatchID"] = out["MatchID"].astype(str)
    if "Timestamp" in out.columns:
        out["Timestamp"] = out["Timestamp"].fillna("")
    out["schema_version"] = max(2, int(out["schema_version"].max()))
    return out


def _migrate_kickoff_v1_to_v2(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "MatchID" in out.columns:
        out["MatchID"] = out["MatchID"].astype(str)
    out["schema_version"] = max(2, int(out["schema_version"].max()))
    return out


def _migrate_event_v2_to_v3(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, default in EVENT_V3_DEFAULTS.items():
        if col not in out.columns:
            out[col] = default
        else:
            out[col] = out[col].fillna(default)
    if "metric_value" in out.columns and "xg" in out.columns:
        metric = pd.to_numeric(out["metric_value"], errors="coerce")
        xg = pd.to_numeric(out["xg"], errors="coerce")
        out["xg"] = xg.fillna(metric).fillna(0.0)
    out["schema_version"] = SCHEMA_VERSION
    return out


def migrate_dataframe(df: pd.DataFrame, table_kind: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else _apply_schema_version(df)

    out = _apply_schema_version(df)
    current_min = int(out["schema_version"].min())
    if current_min < 2:
        if table_kind == "stats":
            out = _migrate_stats_v1_to_v2(out)
        elif table_kind == "kickoff":
            out = _migrate_kickoff_v1_to_v2(out)

    if table_kind == "event" and int(out["schema_version"].min()) < 3:
        out = _migrate_event_v2_to_v3(out)

    if "schema_version" in out.columns:
        out["schema_version"] = SCHEMA_VERSION
    return out
