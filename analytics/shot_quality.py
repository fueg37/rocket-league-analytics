"""Shot quality computations and schema constants.

This module centralizes shot-feature derivation and expected scoring metrics so
xG and xGOT can be treated as co-equal first-class values throughout the app.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

from analytics.contracts import flatten_metric_contract, metric_contract
from analytics.stats_uncertainty import bootstrap_mean_interval, deterministic_seed, reliability_from_sample_size


# Canonical metric columns
COL_XG = "xG"
COL_XGOT = "xGOT"
COL_ON_TARGET = "OnTarget"
COL_TARGET_X = "TargetX"
COL_TARGET_Z = "TargetZ"
COL_SHOT_Z = "ShotZ"
COL_GOALKEEPER_DIST = "GoalkeeperDist"
COL_SHOT_ANGLE = "ShotAngle"
COL_DIST_TO_GOAL = "DistToGoal"

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
    "xG_CI_Low",
    "xG_CI_High",
    "xG_SampleSize",
    "xG_Reliability",
    COL_XGOT,
    "xGOT_CI_Low",
    "xGOT_CI_High",
    "xGOT_SampleSize",
    "xGOT_Reliability",
    COL_ON_TARGET,
    COL_TARGET_X,
    COL_TARGET_Z,
    COL_SHOT_Z,
    COL_GOALKEEPER_DIST,
    COL_SHOT_ANGLE,
    COL_DIST_TO_GOAL,
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
    target_x, target_z = project_to_goal_plane(shot_x, shot_y, shot_z, vel_x, vel_y, vel_z, target_y)
    crosses_goal_plane = False
    if abs(vel_y) >= 1e-6:
        t_goal = (target_y - shot_y) / vel_y
        crosses_goal_plane = t_goal >= 0
    on_target = bool(
        crosses_goal_plane
        and target_x is not None
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



def calculate_xg_contract(
    shot_x: float,
    shot_y: float,
    shot_z: float,
    vel_x: float,
    vel_y: float,
    vel_z: float,
    target_y: float,
):
    """Return xG in the shared uncertainty-aware scalar contract."""
    xg = calculate_xg_probability(shot_x, shot_y, shot_z, vel_x, vel_y, vel_z, target_y)
    spread = max(0.01, min(0.12, 0.08 * (1.0 - xg) + 0.02))
    contract = metric_contract(
        xg,
        ci_low=max(0.0, xg - spread),
        ci_high=min(1.0, xg + spread),
        sample_size=1,
        reliability="low",
    )
    return flatten_metric_contract("xG", contract)


def calculate_xgot_contract(
    shot_x: float,
    shot_y: float,
    shot_z: float,
    vel_x: float,
    vel_y: float,
    vel_z: float,
    target_y: float,
):
    """Return xGOT in the shared uncertainty-aware scalar contract."""
    xgot = calculate_xgot_probability(shot_x, shot_y, shot_z, vel_x, vel_y, vel_z, target_y)
    spread = max(0.015, min(0.16, 0.10 * (1.0 - xgot) + 0.03))
    contract = metric_contract(
        xgot,
        ci_low=max(0.0, xgot - spread),
        ci_high=min(1.0, xgot + spread),
        sample_size=1,
        reliability="low",
    )
    return flatten_metric_contract("xGOT", contract)


def aggregate_metric_contract(values: Iterable[float], metric_name: str) -> dict[str, float | int | str]:
    """Aggregate a scalar metric into mean + bootstrap interval contract columns."""
    vals = [float(v) for v in values]
    sample_size = len(vals)
    if sample_size == 0:
        return flatten_metric_contract(metric_name, metric_contract(0.0, sample_size=0, reliability="low"))
    seed = deterministic_seed(metric_name, sample_size, round(sum(vals), 6))
    mean, ci_low, ci_high = bootstrap_mean_interval(vals, seed=seed)
    return flatten_metric_contract(
        metric_name,
        metric_contract(
            mean,
            ci_low=ci_low,
            ci_high=ci_high,
            sample_size=sample_size,
            reliability=reliability_from_sample_size(sample_size),
        ),
    )

def validate_shot_metric_columns(
    columns: Iterable[str], required: Iterable[str] | None = None
) -> tuple[bool, list[str]]:
    """Validate required shot metric columns before chart rendering.

    Contract fields are optional for backward compatibility; callers may pass
    an explicit `required` list when strict uncertainty columns are needed.
    """
    col_set = set(columns)
    expected = list(required) if required is not None else [
        SHOT_COL_PLAYER,
        SHOT_COL_TEAM,
        SHOT_COL_FRAME,
        COL_XG,
        "xG_CI_Low",
        "xG_CI_High",
        "xG_SampleSize",
        "xG_Reliability",
        COL_XGOT,
        "xGOT_CI_Low",
        "xGOT_CI_High",
        "xGOT_SampleSize",
        "xGOT_Reliability",
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
