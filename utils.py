"""Rocket League Analytics — shared utility functions.

Eliminates the 14+ duplicated player/team mapping patterns scattered across app.py.
"""
import math
from numbers import Real

import numpy as np

from constants import (
    MAX_BALL_SPEED_UU_PER_SEC,
    REPLAY_FPS,
    SPEED_CANONICAL_UNIT,
    SPEED_DISPLAY_UNIT_DEFAULT,
    UU_PER_SEC_TO_MPH,
)


# ── Player / Team Mappings ──────────────────────────────────────────────

def build_pid_team_map(proto):
    """Player ID (str) → team name.

    Replaces: {str(p.id.id): "Orange" if p.is_orange else "Blue" for p in proto.players}
    """
    return {str(p.id.id): ("Orange" if p.is_orange else "Blue") for p in proto.players}


def build_pid_name_map(proto):
    """Player ID (str) → display name.

    Replaces: {str(p.id.id): p.name for p in proto.players}
    """
    return {str(p.id.id): p.name for p in proto.players}


def build_player_team_map(proto):
    """Player display name → team name.

    Replaces: {p.name: "Orange" if p.is_orange else "Blue" for p in proto.players}
    """
    return {p.name: ("Orange" if p.is_orange else "Blue") for p in proto.players}


def get_team_players(proto, team):
    """Return list of player names belonging to *team* ("Blue" or "Orange")."""
    return [p.name for p in proto.players
            if ("Orange" if p.is_orange else "Blue") == team]


def build_player_positions(proto, game_df):
    """Pre-compute per-player position numpy arrays for fast frame lookups.

    Returns {name: {'team': str, 'x': ndarray, 'y': ndarray, 'frames': ndarray}}
    for every player whose position data is available in game_df.
    """
    positions = {}
    for p in proto.players:
        if p.name in game_df:
            pdf = game_df[p.name]
            if 'pos_x' in pdf.columns:
                positions[p.name] = {
                    'team': "Orange" if p.is_orange else "Blue",
                    'x': pdf['pos_x'].values,
                    'y': pdf['pos_y'].values,
                    'frames': pdf.index.values,
                }
    return positions


# ── Frame / Time Helpers ────────────────────────────────────────────────

def frame_to_seconds(frame, fps=REPLAY_FPS):
    """Convert a frame index to seconds."""
    return frame / float(fps)


def seconds_to_frame(seconds, fps=REPLAY_FPS):
    """Convert seconds to the nearest frame index."""
    return int(seconds * fps)


def fmt_time(seconds):
    """Format a time in seconds as *M:SS*."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


# ── Speed Unit Helpers ───────────────────────────────────────────────────

def uu_per_sec_to_mph(value: float) -> float:
    """Convert speed from Unreal Units/second (uu/s) to miles/hour (mph)."""
    return float(value) * UU_PER_SEC_TO_MPH


def normalize_speed_uu_per_sec(value: float) -> float:
    """Normalize raw telemetry speed into canonical uu/s.

    Some replay feeds can intermittently emit speed magnitudes in deci-uu/s
    (10x canonical). We auto-correct only when the value is implausible for
    Rocket League ball physics and a /10 scale lands inside the expected band.
    """
    speed = float(value)
    if speed < 0:
        return 0.0

    plausible_max = MAX_BALL_SPEED_UU_PER_SEC * 1.15
    if speed > plausible_max and (speed / 10.0) <= plausible_max:
        return speed / 10.0
    return speed


def format_speed(value, unit=SPEED_DISPLAY_UNIT_DEFAULT, precision=1, na="N/A") -> str:
    """Format a speed value with defensive handling and unit conversion.

    Input values are always interpreted as uu/s. Conversion is only applied for
    display units (currently mph and uu/s).
    """
    if value is None or isinstance(value, bool):
        return na

    if not isinstance(value, Real):
        return na

    numeric_value = float(value)
    if math.isnan(numeric_value):
        return na

    normalized_unit = str(unit or SPEED_DISPLAY_UNIT_DEFAULT).strip().lower()
    if normalized_unit == "mph":
        converted_value = uu_per_sec_to_mph(numeric_value)
        display_unit = "mph"
    elif normalized_unit in {"uu/s", "uups"}:
        converted_value = numeric_value
        display_unit = SPEED_CANONICAL_UNIT
    else:
        return na

    try:
        decimals = max(0, int(precision))
    except (TypeError, ValueError):
        decimals = 1

    return f"{converted_value:.{decimals}f} {display_unit}"
