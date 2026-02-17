"""Save analytics foundations.

This module provides a canonical save metric contract and helpers for:
- save-event attribution
- save feature extraction
- save scoring (heuristic + calibrated-ready contract)
- player-level aggregation with backward-compatible aliases
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from analytics.contracts import flatten_metric_contract, metric_contract
from analytics.stats_uncertainty import bootstrap_mean_interval, deterministic_seed, reliability_from_sample_size
from constants import FIELD_HALF_Y, REPLAY_FPS
from analytics.shot_quality import (
    COL_DIST_TO_GOAL,
    COL_GOALKEEPER_DIST,
    COL_SHOT_ANGLE,
    COL_SHOT_Z,
    SHOT_COL_FRAME,
    SHOT_COL_PLAYER,
    SHOT_COL_RESULT,
    SHOT_COL_TEAM,
    SHOT_COL_X,
    SHOT_COL_Y,
)


SAVE_METRIC_MODEL_VERSION = "heuristic-v2"

CANONICAL_EVENT_COLUMNS = [
    "Saver",
    "Team",
    "Frame",
    "Time",
    "Shooter",
    "AttributionSource",
    "AttributionConfidence",
    "ShotSpeed",
    "DistToGoal",
    "AngleOffCenter",
    "ShotHeight",
    "SaverDist",
    "SaveDifficultyIndex",
    "SaveDifficultyIndex_CI_Low",
    "SaveDifficultyIndex_CI_High",
    "SaveDifficultyIndex_SampleSize",
    "SaveDifficultyIndex_Reliability",
    "ExpectedSaveProb",
    "ExpectedSaveProb_CI_Low",
    "ExpectedSaveProb_CI_High",
    "ExpectedSaveProb_SampleSize",
    "ExpectedSaveProb_Reliability",
    "SaveImpact",
    "SaveImpact_CI_Low",
    "SaveImpact_CI_High",
    "SaveImpact_SampleSize",
    "SaveImpact_Reliability",
]

CANONICAL_SUMMARY_COLUMNS = [
    "Name",
    "Team",
    "SaveEvents",
    "Total_SaveDifficulty",
    "Avg_SaveDifficulty",
    "Total_ExpectedSaves",
    "Actual_Saves",
    "Total_SaveImpact",
    "Avg_SaveImpact",
    "Avg_SaveImpact_CI_Low",
    "Avg_SaveImpact_CI_High",
    "Avg_SaveImpact_SampleSize",
    "Avg_SaveImpact_Reliability",
    "HighDifficultySaves",
]


@dataclass(frozen=True)
class SaveFeatures:
    """Features used to estimate save difficulty/probability."""

    shot_speed: float
    dist_to_goal: float
    angle_off_center: float
    shot_z: float
    saver_dist: float


@dataclass(frozen=True)
class SaveScore:
    """Outputs from the scoring backend."""

    save_difficulty_index: float
    expected_save_prob: float
    save_impact: float
    model_version: str = SAVE_METRIC_MODEL_VERSION


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def build_save_touch_index(proto, player_map: Dict[str, str], pid_team: Dict[str, str]):
    """Index explicit save touches by frame and defending team."""
    touch_index: Dict[Tuple[int, str], list] = {}
    hits = getattr(getattr(proto, "game_stats", None), "hits", [])
    for hit in hits:
        if not getattr(hit, "player_id", None):
            continue
        if not bool(getattr(hit, "is_save", False)):
            continue
        pid = str(hit.player_id.id)
        saver = player_map.get(pid)
        team = pid_team.get(pid)
        if not saver or not team:
            continue
        key = (int(hit.frame_number), team)
        touch_index.setdefault(key, []).append(saver)
    return touch_index


def find_nearest_defender(frame: int, defending_team: str, shot_x: float, shot_y: float, player_pos: Dict):
    nearest_defender: Optional[str] = None
    nearest_dist = np.inf
    for pname, pd_info in player_pos.items():
        if pd_info.get("team") != defending_team:
            continue
        frames = pd_info.get("frames")
        xs = pd_info.get("x")
        ys = pd_info.get("y")
        if frames is None or xs is None or ys is None or len(xs) == 0:
            continue
        pi = min(np.searchsorted(frames, frame), len(xs) - 1)
        if pi < 0:
            continue
        px, py = xs[pi], ys[pi]
        dist = float(np.sqrt((px - shot_x) ** 2 + (py - shot_y) ** 2))
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_defender = pname
    return nearest_defender, float(nearest_dist) if np.isfinite(nearest_dist) else np.nan


def resolve_saver_for_shot(
    frame: int,
    defending_team: str,
    shot_x: float,
    shot_y: float,
    player_pos: Dict,
    save_touch_index: Dict[Tuple[int, str], list],
    search_window_frames: int = 12,
):
    """Resolve saver with explicit save-touch attribution first, nearest fallback second."""
    for delta in range(search_window_frames + 1):
        for candidate_frame in (frame - delta, frame + delta):
            savers = save_touch_index.get((candidate_frame, defending_team), [])
            if savers:
                return savers[0], np.nan, "explicit_save_touch", 1.0

    nearest_defender, nearest_dist = find_nearest_defender(frame, defending_team, shot_x, shot_y, player_pos)
    if nearest_defender:
        return nearest_defender, nearest_dist, "nearest_defender_fallback", 0.5
    return None, np.nan, "unresolved", 0.0


def build_save_features(shot: pd.Series, target_y: float) -> SaveFeatures:
    """Build canonical features for save scoring."""
    shot_x = float(shot[SHOT_COL_X])
    shot_y = float(shot[SHOT_COL_Y])
    shot_speed = float(shot.get("Speed", 0.0) or 0.0)

    dist_to_goal = shot.get(COL_DIST_TO_GOAL, np.nan)
    if pd.isna(dist_to_goal):
        dist_to_goal = np.sqrt(shot_x ** 2 + (target_y - shot_y) ** 2)

    angle_off_center = shot.get(COL_SHOT_ANGLE, np.nan)
    if pd.isna(angle_off_center):
        angle_off_center = np.arctan2(abs(shot_x), abs(target_y - shot_y))

    shot_z = shot.get(COL_SHOT_Z, np.nan)
    if pd.isna(shot_z):
        shot_z = 0.0

    saver_dist = shot.get(COL_GOALKEEPER_DIST, np.nan)

    return SaveFeatures(
        shot_speed=float(shot_speed),
        dist_to_goal=float(dist_to_goal),
        angle_off_center=float(angle_off_center),
        shot_z=float(max(0.0, shot_z)),
        saver_dist=float(saver_dist) if not pd.isna(saver_dist) else np.nan,
    )


def score_save_heuristic(features: SaveFeatures) -> SaveScore:
    """Heuristic scoring backend with explicit semantic separation.

    - SaveDifficultyIndex (SDI): bounded [0, 1], rule-based index
    - ExpectedSaveProb: derived from SDI as (1 - SDI)
    - SaveImpact for successful save events: 1 - ExpectedSaveProb = SDI
    """
    speed_norm = min(features.shot_speed / 4000.0, 1.5)
    speed_factor = 0.30 * (speed_norm ** 0.8)

    reaction_time = features.dist_to_goal / max(features.shot_speed, 500.0)
    reaction_factor = max(0.0, 0.30 * (1.0 - min(reaction_time / 1.0, 1.0)))

    angle_factor = 0.20 * min(abs(features.angle_off_center) / (np.pi / 2), 1.0)
    height_factor = 0.15 * min(max(features.shot_z - 200, 0) / 600.0, 1.0)

    saver_dist = 0.0 if pd.isna(features.saver_dist) else features.saver_dist
    saver_factor = 0.10 * min(saver_dist / 2000.0, 1.0)

    sdi = clamp01(speed_factor + reaction_factor + angle_factor + height_factor + saver_factor)
    expected_save_prob = clamp01(1.0 - sdi)
    save_impact = clamp01(1.0 - expected_save_prob)

    return SaveScore(
        save_difficulty_index=round(sdi, 3),
        expected_save_prob=round(expected_save_prob, 3),
        save_impact=round(save_impact, 3),
    )


def score_save(features: SaveFeatures, mode: str = "heuristic") -> SaveScore:
    """Scoring entrypoint with backend selection."""
    if mode == "heuristic":
        return score_save_heuristic(features)
    raise ValueError(f"Unsupported save scoring mode: {mode}")


def build_save_events(
    proto,
    shot_df: pd.DataFrame,
    player_pos: Dict,
    player_map: Dict[str, str],
    pid_team: Dict[str, str],
    scoring_mode: str = "heuristic",
):
    """Build event-level save analytics rows."""
    if shot_df.empty:
        return pd.DataFrame(columns=CANONICAL_EVENT_COLUMNS)

    saved_shots = shot_df[shot_df[SHOT_COL_RESULT] == "Shot"].copy()
    if saved_shots.empty:
        return pd.DataFrame(columns=CANONICAL_EVENT_COLUMNS)

    save_touch_index = build_save_touch_index(proto, player_map, pid_team)
    events = []

    for _, shot in saved_shots.iterrows():
        frame = int(shot[SHOT_COL_FRAME])
        shot_team = shot[SHOT_COL_TEAM]
        defending_team = "Orange" if shot_team == "Blue" else "Blue"
        target_y = FIELD_HALF_Y if shot_team == "Blue" else -FIELD_HALF_Y

        saver, nearest_dist, source, confidence = resolve_saver_for_shot(
            frame=frame,
            defending_team=defending_team,
            shot_x=float(shot[SHOT_COL_X]),
            shot_y=float(shot[SHOT_COL_Y]),
            player_pos=player_pos,
            save_touch_index=save_touch_index,
        )
        if not saver:
            continue

        features = build_save_features(shot, target_y=target_y)
        saver_dist = features.saver_dist
        if pd.isna(saver_dist):
            saver_dist = nearest_dist
            features = SaveFeatures(
                shot_speed=features.shot_speed,
                dist_to_goal=features.dist_to_goal,
                angle_off_center=features.angle_off_center,
                shot_z=features.shot_z,
                saver_dist=saver_dist,
            )

        score = score_save(features, mode=scoring_mode)
        sdi_contract = flatten_metric_contract("SaveDifficultyIndex", metric_contract(score.save_difficulty_index, ci_low=max(0.0, score.save_difficulty_index - 0.08), ci_high=min(1.0, score.save_difficulty_index + 0.08), sample_size=1, reliability="low"))
        exp_contract = flatten_metric_contract("ExpectedSaveProb", metric_contract(score.expected_save_prob, ci_low=max(0.0, score.expected_save_prob - 0.08), ci_high=min(1.0, score.expected_save_prob + 0.08), sample_size=1, reliability="low"))
        impact_contract = flatten_metric_contract("SaveImpact", metric_contract(score.save_impact, ci_low=max(0.0, score.save_impact - 0.08), ci_high=min(1.0, score.save_impact + 0.08), sample_size=1, reliability="low"))
        event_row = {
                "Saver": saver,
                "Team": defending_team,
                "Frame": frame,
                "Time": round(frame / REPLAY_FPS, 1),
                "Shooter": shot[SHOT_COL_PLAYER],
                "AttributionSource": source,
                "AttributionConfidence": round(confidence, 2),
                "ShotSpeed": int(round(features.shot_speed)),
                "DistToGoal": int(round(features.dist_to_goal)),
                "AngleOffCenter": round(float(features.angle_off_center), 3),
                "ShotHeight": int(round(features.shot_z)),
                "SaverDist": int(round(saver_dist)) if not pd.isna(saver_dist) else 0,
                "SaveDifficultyIndex": score.save_difficulty_index,
                "ExpectedSaveProb": score.expected_save_prob,
                "SaveImpact": score.save_impact,
            }
        event_row.update(sdi_contract)
        event_row.update(exp_contract)
        event_row.update(impact_contract)
        events.append(event_row)

    return pd.DataFrame(events, columns=CANONICAL_EVENT_COLUMNS)


def aggregate_save_summary(proto, save_events_df: pd.DataFrame, high_difficulty_quantile: float = 0.80):
    """Aggregate canonical save summary and include legacy aliases."""
    threshold = 1.0
    if not save_events_df.empty:
        threshold = float(save_events_df["SaveDifficultyIndex"].quantile(high_difficulty_quantile))

    rows = []
    for p in proto.players:
        name = p.name
        team = "Orange" if p.is_orange else "Blue"
        p_saves = save_events_df[save_events_df["Saver"] == name] if not save_events_df.empty else pd.DataFrame()

        save_events = int(len(p_saves))
        total_difficulty = round(float(p_saves["SaveDifficultyIndex"].sum()), 2) if save_events else 0.0
        avg_difficulty = round(float(p_saves["SaveDifficultyIndex"].mean()), 3) if save_events else 0.0
        total_expected = round(float(p_saves["ExpectedSaveProb"].sum()), 2) if save_events else 0.0
        total_impact = round(float(p_saves["SaveImpact"].sum()), 2) if save_events else 0.0
        avg_impact = round(float(p_saves["SaveImpact"].mean()), 3) if save_events else 0.0
        if save_events:
            mean_impact, impact_ci_low, impact_ci_high = bootstrap_mean_interval(
                p_saves["SaveImpact"].tolist(),
                seed=deterministic_seed(name, team, save_events, "save_impact"),
            )
        else:
            mean_impact, impact_ci_low, impact_ci_high = 0.0, 0.0, 0.0
        reliability = reliability_from_sample_size(save_events)
        high_difficulty_saves = int((p_saves["SaveDifficultyIndex"] >= threshold).sum()) if save_events else 0

        rows.append(
            {
                "Name": name,
                "Team": team,
                "SaveEvents": save_events,
                "Total_SaveDifficulty": total_difficulty,
                "Avg_SaveDifficulty": avg_difficulty,
                "Total_ExpectedSaves": total_expected,
                "Actual_Saves": save_events,
                "Total_SaveImpact": total_impact,
                "Avg_SaveImpact": round(mean_impact, 3) if save_events else avg_impact,
                "Avg_SaveImpact_CI_Low": round(impact_ci_low, 3),
                "Avg_SaveImpact_CI_High": round(impact_ci_high, 3),
                "Avg_SaveImpact_SampleSize": save_events,
                "Avg_SaveImpact_Reliability": reliability,
                "HighDifficultySaves": high_difficulty_saves,
                # Legacy aliases for compatibility
                "Saves_Nearby": save_events,
                "Total_xS": total_difficulty,
                "Avg_xS": avg_difficulty,
                "Hard_Saves": high_difficulty_saves,
            }
        )

    return pd.DataFrame(rows)


def calculate_save_analytics(proto, shot_df, player_pos, player_map, pid_team, scoring_mode: str = "heuristic"):
    """Return event + summary save analytics dataframes."""
    events = build_save_events(proto, shot_df, player_pos, player_map, pid_team, scoring_mode=scoring_mode)
    summary = aggregate_save_summary(proto, events)
    return events, summary
