from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

import pandas as pd

SCHEMA_VERSION = 4


class EventType(str, Enum):
    SHOT_TAKEN = "shot_taken"
    SHOT_ON_TARGET = "shot_on_target"
    GOAL = "goal"
    SAVE = "save"
    BLOCK = "block"
    CLEAR = "clear"
    CHALLENGE_WIN = "challenge_win"
    CHALLENGE_LOSS = "challenge_loss"
    PASS_COMPLETED = "pass_completed"
    TOUCH = "touch"
    BOOST_PICKUP = "boost_pickup"
    KICKOFF = "kickoff"


SHOT_LIFECYCLE_EVENT_TYPES = frozenset({
    EventType.SHOT_TAKEN,
    EventType.SHOT_ON_TARGET,
    EventType.GOAL,
    EventType.SAVE,
    EventType.BLOCK,
    EventType.CLEAR,
    EventType.CHALLENGE_WIN,
    EventType.CHALLENGE_LOSS,
    EventType.PASS_COMPLETED,
})


@dataclass(frozen=True)
class TableContract:
    name: str
    columns: Mapping[str, str]

    def empty(self) -> pd.DataFrame:
        return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in self.columns.items()})


MATCH_CONTRACT = TableContract(
    name="match",
    columns={
        "schema_version": "int64",
        "match_id": "string",
        "file_name": "string",
        "map_name": "string",
        "playlist": "string",
        "num_frames": "int64",
        "duration_seconds": "float64",
        "overtime_seconds": "float64",
    },
)

FRAME_STATE_CONTRACT = TableContract(
    name="frame_state",
    columns={
        "schema_version": "int64",
        "match_id": "string",
        "frame": "int64",
        "time_seconds": "float64",
        "ball_x": "float64",
        "ball_y": "float64",
        "ball_z": "float64",
        "ball_vx": "float64",
        "ball_vy": "float64",
        "ball_vz": "float64",
        "score_blue": "int64",
        "score_orange": "int64",
        "possessing_player": "string",
        "possessing_team": "string",
    },
)

EVENT_CONTRACT = TableContract(
    name="event",
    columns={
        "schema_version": "int64",
        "match_id": "string",
        "event_id": "string",
        "frame": "int64",
        "time_seconds": "float64",
        "event_type": "string",
        "player_id": "string",
        "player_name": "string",
        "team": "string",
        "x": "float64",
        "y": "float64",
        "z": "float64",
        "metric_value": "float64",
        "is_key_play": "bool",
        "outcome_type": "string",
        "is_on_target": "bool",
        "is_big_chance": "bool",
        "speed": "float64",
        "pressure_context": "string",
        "nearest_defender_distance": "float64",
        "shot_angle": "float64",
        "shooter_boost": "float64",
        "distance_to_goal": "float64",
        "xg_pre": "float64",
        "xg_post": "float64",
        "xg_model_version": "string",
        "xg_calibration_version": "string",
        "xg": "float64",
        "xa": "float64",
    },
)

POSSESSION_SEGMENT_CONTRACT = TableContract(
    name="possession_segment",
    columns={
        "schema_version": "int64",
        "match_id": "string",
        "segment_id": "string",
        "start_frame": "int64",
        "end_frame": "int64",
        "start_time": "float64",
        "end_time": "float64",
        "duration_seconds": "float64",
        "player_name": "string",
        "team": "string",
    },
)


@dataclass
class AnalyticsTables:
    match: pd.DataFrame
    frame_state: pd.DataFrame
    event: pd.DataFrame
    possession_segment: pd.DataFrame


def _coerce_table(df: pd.DataFrame, contract: TableContract) -> pd.DataFrame:
    if df is None or df.empty:
        return contract.empty()
    out = df.copy()
    for col, dtype in contract.columns.items():
        if col not in out.columns:
            out[col] = pd.NA
        try:
            out[col] = out[col].astype(dtype)
        except (TypeError, ValueError):
            pass
    return out[list(contract.columns.keys())]


def normalize_tables(tables: AnalyticsTables) -> AnalyticsTables:
    return AnalyticsTables(
        match=_coerce_table(tables.match, MATCH_CONTRACT),
        frame_state=_coerce_table(tables.frame_state, FRAME_STATE_CONTRACT),
        event=_coerce_table(tables.event, EVENT_CONTRACT),
        possession_segment=_coerce_table(tables.possession_segment, POSSESSION_SEGMENT_CONTRACT),
    )


def with_schema_version(df: pd.DataFrame, schema_version: int = SCHEMA_VERSION) -> pd.DataFrame:
    out = df.copy()
    out["schema_version"] = int(schema_version)
    return out


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out
