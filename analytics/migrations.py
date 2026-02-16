from __future__ import annotations

from typing import Callable

import pandas as pd

from analytics.schema import SCHEMA_VERSION


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
    out["schema_version"] = SCHEMA_VERSION
    return out


def _migrate_kickoff_v1_to_v2(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "MatchID" in out.columns:
        out["MatchID"] = out["MatchID"].astype(str)
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
    return out
