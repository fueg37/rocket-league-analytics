"""Shot quality computations and schema constants.

This module centralizes shot-feature derivation and expected scoring metrics so
xG and xGOT can be treated as co-equal first-class values throughout the app.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping


# Canonical metric columns
COL_XG = "xG"
COL_XGOT = "xGOT"
COL_ON_TARGET = "OnTarget"
COL_TARGET_X = "TargetX"
COL_TARGET_Z = "TargetZ"

# Shared shot-event schema columns
SHOT_COL_PLAYER = "Player"
SHOT_COL_TEAM = "Team"
SHOT_COL_FRAME = "Frame"
SHOT_COL_RESULT = "Result"
SHOT_COL_BIG_CHANCE = "BigChance"
SHOT_COL_X = "X"
SHOT_COL_Y = "Y"
SHOT_COL_SPEED = "Speed"

SHOT_EVENT_COLUMNS = (
    SHOT_COL_PLAYER,
    SHOT_COL_TEAM,
    SHOT_COL_FRAME,
    COL_XG,
    COL_XGOT,
    COL_ON_TARGET,
    COL_TARGET_X,
    COL_TARGET_Z,
    SHOT_COL_RESULT,
    SHOT_COL_BIG_CHANCE,
    SHOT_COL_X,
    SHOT_COL_Y,
    SHOT_COL_SPEED,
)


def project_to_goal_plane(
    shot_x: float,
    shot_y: float,
    shot_z: float,
    vel_x: float,
    vel_y: float,
    vel_z: float,
    target_y: float,
) -> tuple[float | None, float | None]:
    """Project ball trajectory to the goal plane (Y=target_y).

    Returns (target_x, target_z). If projection cannot be computed because the
    shot has no y-velocity, returns (None, None).
    """
    if abs(vel_y) < 1e-6:
        return None, None

    t = (target_y - shot_y) / vel_y
    target_x = shot_x + vel_x * t
    target_z = shot_z + vel_z * t
    return float(target_x), float(target_z)


def compute_shot_features(
    shot_x: float,
    shot_y: float,
    shot_z: float,
    vel_x: float,
    vel_y: float,
    vel_z: float,
    target_y: float,
    *,
    goal_half_w: float = 893.0,
    goal_height: float = 642.0,
) -> dict[str, float | bool | None]:
    """Derive reusable geometry/trajectory features for shot quality models."""
    dx = float(shot_x)
    dy = float(target_y - shot_y)
    dist = math.sqrt(dx * dx + dy * dy)

    vec_l = (-goal_half_w - shot_x, target_y - shot_y)
    vec_r = (goal_half_w - shot_x, target_y - shot_y)
    norm_l = math.sqrt(vec_l[0] ** 2 + vec_l[1] ** 2)
    norm_r = math.sqrt(vec_r[0] ** 2 + vec_r[1] ** 2)

    if norm_l <= 0 or norm_r <= 0:
        angle = 0.0
    else:
        dot = (vec_l[0] * vec_r[0] + vec_l[1] * vec_r[1]) / (norm_l * norm_r)
        angle = math.acos(max(-1.0, min(1.0, dot)))

    speed = math.sqrt(vel_x * vel_x + vel_y * vel_y + vel_z * vel_z)
    target_x, target_z = project_to_goal_plane(
        shot_x, shot_y, shot_z, vel_x, vel_y, vel_z, target_y
    )
    on_target = bool(
        target_x is not None
        and target_z is not None
        and abs(target_x) <= goal_half_w
        and 0 <= target_z <= goal_height
    )

    return {
        "distance_to_goal": dist,
        "open_angle": angle,
        "speed": speed,
        "shot_z": float(shot_z),
        COL_TARGET_X: target_x,
        COL_TARGET_Z: target_z,
        COL_ON_TARGET: on_target,
    }


def calculate_xg_probability(
    shot_x: float,
    shot_y: float,
    shot_z: float,
    vel_x: float,
    vel_y: float,
    vel_z: float,
    target_y: float,
) -> float:
    """Compute pre-shot expected-goal probability (xG)."""
    features = compute_shot_features(
        shot_x, shot_y, shot_z, vel_x, vel_y, vel_z, target_y
    )

    base_xg = (features["open_angle"] * 0.85) * math.exp(
        -0.00045 * features["distance_to_goal"]
    )
    speed_factor = 1.0 + (features["speed"] - 1400) / 2000.0
    speed_factor = max(0.5, min(speed_factor, 1.5))
    height_factor = 1.15 if features["shot_z"] > 150 else 1.0
    xg = base_xg * speed_factor * height_factor
    return min(max(xg, 0.01), 0.99)


def calculate_xgot_probability(
    shot_x: float,
    shot_y: float,
    shot_z: float,
    vel_x: float,
    vel_y: float,
    vel_z: float,
    target_y: float,
) -> float:
    """Compute post-shot expected-goal-on-target probability (xGOT)."""
    features = compute_shot_features(
        shot_x, shot_y, shot_z, vel_x, vel_y, vel_z, target_y
    )
    xg = calculate_xg_probability(
        shot_x, shot_y, shot_z, vel_x, vel_y, vel_z, target_y
    )

    if not features[COL_ON_TARGET]:
        return round(max(0.01, xg * 0.35), 4)

    target_x = float(features[COL_TARGET_X])
    target_z = float(features[COL_TARGET_Z])
    horizontal_edge = abs(target_x) / 893.0
    vertical_edge = abs(target_z - 321.0) / 321.0
    placement_bonus = min(0.35, 0.22 * horizontal_edge + 0.18 * vertical_edge)
    pace_bonus = min(0.2, max(0.0, (features["speed"] - 1500.0) / 3200.0))
    xgot = xg + placement_bonus + pace_bonus
    return min(max(xgot, 0.01), 0.99)


def validate_shot_metric_columns(
    columns: Iterable[str], required: Iterable[str] | None = None
) -> tuple[bool, list[str]]:
    """Validate required shot metric columns before chart rendering."""
    col_set = set(columns)
    expected = list(required) if required is not None else [
        SHOT_COL_PLAYER,
        SHOT_COL_TEAM,
        SHOT_COL_FRAME,
        COL_XG,
        COL_XGOT,
        COL_ON_TARGET,
        COL_TARGET_X,
        COL_TARGET_Z,
        SHOT_COL_RESULT,
        SHOT_COL_X,
        SHOT_COL_Y,
    ]
    missing = [c for c in expected if c not in col_set]
    return len(missing) == 0, missing


def metric_value(row: Mapping[str, float], metric_name: str, default: float = 0.0) -> float:
    """Small helper to retrieve a metric value from a row mapping."""
    value = row.get(metric_name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
