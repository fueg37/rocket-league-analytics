from __future__ import annotations

from typing import Dict

import numpy as np

from constants import GOAL_HALF_W

FEATURE_VERSION = "2026.02"

PRE_SHOT_FEATURE_COLUMNS = [
    "distance_to_goal",
    "shot_angle",
    "ball_height",
    "shooter_speed",
    "pressure_score",
    "nearest_defender_distance",
    "shooter_boost",
    "buildup_seconds",
    "touches_in_chain",
    "chain_duration",
    "chain_avg_ball_speed",
    "chain_final_third_entries",
    "chain_turnovers_forced",
]

POST_SHOT_FEATURE_COLUMNS = [
    "shot_speed",
    "shot_vx_norm",
    "shot_vy_norm",
    "shot_vz_norm",
    "target_placement_x",
    "shot_height",
    "keeper_distance",
    "keeper_line_offset",
]


def _safe_norm(vec: np.ndarray) -> float:
    n = float(np.linalg.norm(vec))
    return n if n > 1e-6 else 0.0


def classify_pressure_context(nearest_defender_distance: float) -> str:
    if np.isnan(nearest_defender_distance):
        return "unknown"
    if nearest_defender_distance <= 500:
        return "high"
    if nearest_defender_distance <= 1200:
        return "medium"
    return "low"


def pressure_score_from_distance(nearest_defender_distance: float) -> float:
    if np.isnan(nearest_defender_distance):
        return 0.5
    return float(np.clip(1.0 - (nearest_defender_distance / 2500.0), 0.0, 1.0))


def build_pre_shot_features(
    shot_x: float,
    shot_y: float,
    shot_z: float,
    shooter_speed: float,
    nearest_defender_distance: float,
    shooter_boost: float,
    buildup_seconds: float,
    team: str,
    touches_in_chain: float = 0.0,
    chain_duration: float = 0.0,
    chain_avg_ball_speed: float = 0.0,
    chain_final_third_entries: float = 0.0,
    chain_turnovers_forced: float = 0.0,
) -> Dict[str, float]:
    target_y = 5120.0 if team == "Blue" else -5120.0
    vec_l = np.array([-893.0 - shot_x, target_y - shot_y], dtype=float)
    vec_r = np.array([893.0 - shot_x, target_y - shot_y], dtype=float)
    norm_l, norm_r = _safe_norm(vec_l), _safe_norm(vec_r)
    if norm_l == 0.0 or norm_r == 0.0:
        shot_angle = 0.0
    else:
        dot = float(np.dot(vec_l / norm_l, vec_r / norm_r))
        shot_angle = float(np.arccos(np.clip(dot, -1.0, 1.0)))

    return {
        "distance_to_goal": float(np.hypot(shot_x, target_y - shot_y)),
        "shot_angle": shot_angle,
        "ball_height": float(max(0.0, shot_z)),
        "shooter_speed": float(max(0.0, shooter_speed)),
        "pressure_score": pressure_score_from_distance(nearest_defender_distance),
        "nearest_defender_distance": float(nearest_defender_distance),
        "shooter_boost": float(np.clip(shooter_boost if not np.isnan(shooter_boost) else 0.0, 0.0, 100.0)),
        "buildup_seconds": float(max(0.0, buildup_seconds)),
        "touches_in_chain": float(max(0.0, touches_in_chain if not np.isnan(touches_in_chain) else 0.0)),
        "chain_duration": float(max(0.0, chain_duration if not np.isnan(chain_duration) else 0.0)),
        "chain_avg_ball_speed": float(max(0.0, chain_avg_ball_speed if not np.isnan(chain_avg_ball_speed) else 0.0)),
        "chain_final_third_entries": float(max(0.0, chain_final_third_entries if not np.isnan(chain_final_third_entries) else 0.0)),
        "chain_turnovers_forced": float(max(0.0, chain_turnovers_forced if not np.isnan(chain_turnovers_forced) else 0.0)),
    }


def build_post_shot_features(
    shot_x: float,
    shot_y: float,
    shot_z: float,
    vel_x: float,
    vel_y: float,
    vel_z: float,
    keeper_distance: float,
    keeper_line_offset: float,
    team: str,
) -> Dict[str, float]:
    target_y = 5120.0 if team == "Blue" else -5120.0
    shot_speed = float(np.linalg.norm([vel_x, vel_y, vel_z]))
    if abs(vel_y) > 1e-6:
        t_goal = (target_y - shot_y) / vel_y
    else:
        t_goal = 0.0
    projected_x = float(shot_x + vel_x * max(t_goal, 0.0))
    placement_x = float(np.clip(abs(projected_x) / GOAL_HALF_W, 0.0, 2.0))
    if shot_speed > 1e-6:
        vx_n, vy_n, vz_n = vel_x / shot_speed, vel_y / shot_speed, vel_z / shot_speed
    else:
        vx_n, vy_n, vz_n = 0.0, 0.0, 0.0

    return {
        "shot_speed": shot_speed,
        "shot_vx_norm": float(vx_n),
        "shot_vy_norm": float(vy_n),
        "shot_vz_norm": float(vz_n),
        "target_placement_x": placement_x,
        "shot_height": float(max(0.0, shot_z)),
        "keeper_distance": float(keeper_distance),
        "keeper_line_offset": float(keeper_line_offset),
    }
