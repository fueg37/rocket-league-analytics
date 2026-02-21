"""Canonical possession-value domain model and transition-value estimators.

This module centralizes state encoding and action valuation so callers can rely on
one consistent schema instead of bespoke VAEP heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from constants import REPLAY_FPS

STATE_INDEX_COLUMNS = ["MatchID", "Frame"]
TEAM_BELIEF_COLUMNS = ["BluePossessionBelief", "OrangePossessionBelief"]
STATE_FEATURE_COLUMNS = [
    "BallPosX",
    "BallPosY",
    "BallPosZ",
    "BallVelX",
    "BallVelY",
    "BallVelZ",
    "BallSpeedUUPerSec",
    "NearestBlueDist",
    "NearestOrangeDist",
    "TeamBoostAvgBlue",
    "TeamBoostAvgOrange",
    "PressureBlue",
    "PressureOrange",
    "BlueAttacking",
    "OrangeAttacking",
    "SecondsRemaining",
]
CANONICAL_STATE_COLUMNS = STATE_INDEX_COLUMNS + TEAM_BELIEF_COLUMNS + STATE_FEATURE_COLUMNS


@dataclass(slots=True)
class TransitionValueEstimator:
    """Small linear transition-value estimator with multi-horizon outputs."""

    intercepts: dict[str, float]
    coefficients: dict[str, np.ndarray]
    feature_columns: list[str]
    horizons: tuple[str, ...] = ("3s", "10s")

    def predict_state_value(self, state: Mapping[str, float] | pd.Series | pd.DataFrame) -> dict[str, float] | pd.DataFrame:
        """Predict expected goal differential for configured horizons."""
        if isinstance(state, pd.DataFrame):
            features = _state_features_matrix(state, self.feature_columns)
            out = {
                f"ExpectedGoalDiff_{h}": self.intercepts[h] + (features @ self.coefficients[h])
                for h in self.horizons
            }
            return pd.DataFrame(out, index=state.index)

        state_series = pd.Series(state)
        features = np.array([float(state_series.get(col, 0.0) or 0.0) for col in self.feature_columns], dtype=float)
        return {h: float(self.intercepts[h] + np.dot(features, self.coefficients[h])) for h in self.horizons}


def _state_features_matrix(states: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    frame = states.copy()
    for col in feature_columns:
        if col not in frame.columns:
            frame[col] = 0.0
    return frame[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)


def fit_value_model(states: pd.DataFrame, outcomes: pd.DataFrame | Mapping[str, np.ndarray]) -> TransitionValueEstimator:
    """Fit a transition-value model from canonical states and observed outcomes."""
    feature_columns = [
        "BluePossessionBelief",
        "OrangePossessionBelief",
        "BallPosY",
        "BallVelY",
        "NearestBlueDist",
        "NearestOrangeDist",
        "TeamBoostAvgBlue",
        "TeamBoostAvgOrange",
        "PressureBlue",
        "PressureOrange",
        "BlueAttacking",
        "OrangeAttacking",
    ]
    horizons = ("3s", "10s")
    y_map: dict[str, np.ndarray] = {}

    for horizon in horizons:
        key = f"ExpectedGoalDiff_{horizon}"
        if isinstance(outcomes, pd.DataFrame):
            y = pd.to_numeric(outcomes.get(key, pd.Series([0.0] * len(states))), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        else:
            y = np.asarray(outcomes.get(key, np.zeros(len(states), dtype=float)), dtype=float)
        y_map[horizon] = y

    X = _state_features_matrix(states, feature_columns)
    X_aug = np.column_stack([np.ones(len(X)), X])
    intercepts: dict[str, float] = {}
    coefficients: dict[str, np.ndarray] = {}
    for horizon, y in y_map.items():
        beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        intercepts[horizon] = float(beta[0])
        coefficients[horizon] = beta[1:].astype(float)
    return TransitionValueEstimator(intercepts=intercepts, coefficients=coefficients, feature_columns=feature_columns, horizons=horizons)


def predict_state_value(state: Mapping[str, float] | pd.Series, model: TransitionValueEstimator | None = None) -> dict[str, float]:
    """Predict expected goal differential for one state over canonical horizons."""
    estimator = model or _default_estimator()
    return estimator.predict_state_value(state)


def _default_estimator() -> TransitionValueEstimator:
    feature_columns = [
        "BluePossessionBelief",
        "OrangePossessionBelief",
        "BallPosY",
        "BallVelY",
        "NearestBlueDist",
        "NearestOrangeDist",
        "TeamBoostAvgBlue",
        "TeamBoostAvgOrange",
        "PressureBlue",
        "PressureOrange",
        "BlueAttacking",
        "OrangeAttacking",
    ]
    intercepts = {"3s": 0.0, "10s": 0.0}
    coefficients = {
        "3s": np.array([0.35, -0.35, 0.00008, 0.00012, -0.00018, 0.00018, 0.0015, -0.0015, 0.08, -0.08, 0.05, -0.05]),
        "10s": np.array([0.55, -0.55, 0.00011, 0.00008, -0.00012, 0.00012, 0.0020, -0.0020, 0.06, -0.06, 0.08, -0.08]),
    }
    return TransitionValueEstimator(intercepts=intercepts, coefficients=coefficients, feature_columns=feature_columns)


def encode_replay_states(match_id: str, game_df: pd.DataFrame, player_pos: Mapping[str, dict], pid_team: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Encode replay frames into the canonical possession-value state schema."""
    ball_df = game_df.get("ball")
    if ball_df is None or ball_df.empty:
        return pd.DataFrame(columns=CANONICAL_STATE_COLUMNS)

    ball_frames = ball_df.index.to_numpy(dtype=int)

    def _numeric_column(df: pd.DataFrame, col: str, length: int) -> np.ndarray:
        if col not in df.columns:
            return np.zeros(length, dtype=float)
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    bx = _numeric_column(ball_df, "pos_x", len(ball_frames))
    by = _numeric_column(ball_df, "pos_y", len(ball_frames))
    bz = _numeric_column(ball_df, "pos_z", len(ball_frames))
    bvx = _numeric_column(ball_df, "vel_x", len(ball_frames))
    bvy = _numeric_column(ball_df, "vel_y", len(ball_frames))
    bvz = _numeric_column(ball_df, "vel_z", len(ball_frames))

    aligned_dists: list[np.ndarray] = []
    dist_is_blue: list[bool] = []
    aligned_boosts: list[np.ndarray] = []
    boost_is_blue: list[bool] = []

    for name, pdata in player_pos.items():
        team = pdata.get("team")
        player_frames = pdata.get("frames")
        player_x = pdata.get("x")
        player_y = pdata.get("y")

        if player_frames is not None and player_x is not None and player_y is not None and len(player_frames) > 0:
            player_frames_arr = np.asarray(player_frames, dtype=int)
            idx_per_frame = np.searchsorted(player_frames_arr, ball_frames, side="left").clip(0, len(player_frames_arr) - 1)

            x_arr = np.asarray(player_x, dtype=float)
            y_arr = np.asarray(player_y, dtype=float)
            valid_len = min(len(x_arr), len(y_arr), len(player_frames_arr))
            if valid_len > 0:
                aligned_idx = idx_per_frame.clip(0, valid_len - 1)
                dists = np.sqrt((x_arr[:valid_len][aligned_idx] - bx) ** 2 + (y_arr[:valid_len][aligned_idx] - by) ** 2)
                aligned_dists.append(dists)
                dist_is_blue.append(team == "Blue")

        player_df = game_df.get(name)
        if player_df is not None and "boost" in player_df.columns and not player_df.empty:
            boost_frames = player_df.index.to_numpy(dtype=int)
            boost_idx = np.searchsorted(boost_frames, ball_frames, side="left").clip(0, len(boost_frames) - 1)
            boost_vals = pd.to_numeric(player_df["boost"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            aligned_boosts.append(boost_vals[boost_idx])
            boost_is_blue.append(pdata.get("team", "Blue") == "Blue")

    if aligned_dists:
        dist_stack = np.vstack(aligned_dists)
        blue_mask = np.asarray(dist_is_blue, dtype=bool)
        orange_mask = ~blue_mask
        nearest_blue = np.min(np.where(blue_mask[:, None], dist_stack, 10000.0), axis=0)
        nearest_orange = np.min(np.where(orange_mask[:, None], dist_stack, 10000.0), axis=0)
    else:
        nearest_blue = np.full(len(ball_frames), 10000.0)
        nearest_orange = np.full(len(ball_frames), 10000.0)

    if aligned_boosts:
        boost_stack = np.vstack(aligned_boosts)
        blue_boost_mask = np.asarray(boost_is_blue, dtype=bool)
        orange_boost_mask = ~blue_boost_mask
        blue_boost_values = np.where(blue_boost_mask[:, None], boost_stack, np.nan)
        orange_boost_values = np.where(orange_boost_mask[:, None], boost_stack, np.nan)
        boost_blue = np.nan_to_num(np.nanmean(blue_boost_values, axis=0), nan=0.0)
        boost_orange = np.nan_to_num(np.nanmean(orange_boost_values, axis=0), nan=0.0)
    else:
        boost_blue = np.zeros(len(ball_frames), dtype=float)
        boost_orange = np.zeros(len(ball_frames), dtype=float)

    blue_belief = 1.0 / (1.0 + np.exp(-((nearest_orange - nearest_blue) / 450.0 + np.tanh(by / 3500.0) * 0.35)))
    orange_belief = 1.0 - blue_belief
    pressure_blue = np.maximum(0.0, 1.0 - (nearest_blue / 2000.0))
    pressure_orange = np.maximum(0.0, 1.0 - (nearest_orange / 2000.0))

    max_frame = int(game_df.index.max()) if len(game_df.index) else 0
    states = pd.DataFrame(
        {
            "MatchID": np.full(len(ball_frames), match_id, dtype=object),
            "Frame": ball_frames,
            "BluePossessionBelief": blue_belief,
            "OrangePossessionBelief": orange_belief,
            "BallPosX": bx,
            "BallPosY": by,
            "BallPosZ": bz,
            "BallVelX": bvx,
            "BallVelY": bvy,
            "BallVelZ": bvz,
            "BallSpeedUUPerSec": np.sqrt(bvx**2 + bvy**2 + bvz**2),
            "NearestBlueDist": nearest_blue,
            "NearestOrangeDist": nearest_orange,
            "TeamBoostAvgBlue": boost_blue,
            "TeamBoostAvgOrange": boost_orange,
            "PressureBlue": pressure_blue,
            "PressureOrange": pressure_orange,
            "BlueAttacking": (by > 0.0).astype(float),
            "OrangeAttacking": (by < 0.0).astype(float),
            "SecondsRemaining": np.maximum(0.0, (max_frame - ball_frames) / float(REPLAY_FPS)),
        },
        columns=CANONICAL_STATE_COLUMNS,
    )
    return states


def _nearest_team_distances(frame: int, bx: float, by: float, player_pos: Mapping[str, dict]) -> tuple[float, float]:
    blue_min, orange_min = 10000.0, 10000.0
    for pdata in player_pos.values():
        frames = pdata.get("frames")
        xs = pdata.get("x")
        ys = pdata.get("y")
        if frames is None or xs is None or ys is None or len(frames) == 0:
            continue
        idx = min(np.searchsorted(frames, frame), len(xs) - 1)
        dist = float(np.sqrt((float(xs[idx]) - bx) ** 2 + (float(ys[idx]) - by) ** 2))
        if pdata.get("team") == "Blue":
            blue_min = min(blue_min, dist)
        else:
            orange_min = min(orange_min, dist)
    return blue_min, orange_min


def _team_boost_context(frame: int, game_df: pd.DataFrame, player_pos: Mapping[str, dict]) -> tuple[float, float]:
    boost_vals = {"Blue": [], "Orange": []}
    for name, pdata in player_pos.items():
        player_df = game_df.get(name)
        if player_df is None or "boost" not in player_df.columns:
            continue
        frames = player_df.index.values
        idx = min(np.searchsorted(frames, frame), len(frames) - 1)
        boost_vals[pdata.get("team", "Blue")].append(float(player_df["boost"].iloc[idx]))

    return (
        float(np.mean(boost_vals["Blue"])) if boost_vals["Blue"] else 0.0,
        float(np.mean(boost_vals["Orange"])) if boost_vals["Orange"] else 0.0,
    )


def _poss_belief(nearest_blue: float, nearest_orange: float, ball_y: float, team: str) -> float:
    delta = nearest_orange - nearest_blue
    terrain = np.tanh(ball_y / 3500.0)
    blue_poss = float(1.0 / (1.0 + np.exp(-(delta / 450.0 + terrain * 0.35))))
    return blue_poss if team == "Blue" else 1.0 - blue_poss


def compute_action_value_deltas(events: pd.DataFrame, states: pd.DataFrame, model: TransitionValueEstimator | None = None) -> pd.DataFrame:
    """Assign value added/lost per touch/challenge from state transition deltas."""
    if events is None or events.empty:
        return pd.DataFrame(columns=["MatchID", "Frame", "Player", "Team", "EventType", "ValueDelta_3s", "ValueDelta_10s", "VAEP"])
    if states is None or states.empty:
        out = events.copy()
        for col in ["ValueDelta_3s", "ValueDelta_10s", "VAEP"]:
            out[col] = 0.0
        return out

    estimator = model or _default_estimator()
    state_lookup = states.reset_index(drop=True)
    state_cache = _build_state_cache(state_lookup)
    valued = events.copy()
    value_3s = []
    value_10s = []

    for row in valued.itertuples(index=False):
        match_id = getattr(row, "MatchID", None)
        frame = int(getattr(row, "Frame", 0) or 0)
        team = getattr(row, "Team", "Blue") or "Blue"
        post_frame = int(getattr(row, "PostFrame", frame + 1) or (frame + 1))

        pre_state = _get_state(state_lookup, state_cache, match_id, frame)
        post_state = _get_state(state_lookup, state_cache, match_id, post_frame)
        pre_pred = estimator.predict_state_value(pre_state)
        post_pred = estimator.predict_state_value(post_state)

        direction = 1.0 if team == "Blue" else -1.0
        value_3s.append(direction * (post_pred["3s"] - pre_pred["3s"]))
        value_10s.append(direction * (post_pred["10s"] - pre_pred["10s"]))

    valued["ValueDelta_3s"] = np.round(value_3s, 4)
    valued["ValueDelta_10s"] = np.round(value_10s, 4)
    valued["VAEP"] = valued["ValueDelta_3s"]
    return valued


def _build_state_cache(states: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if states.empty:
        return {}

    match_ids = states["MatchID"].to_numpy(dtype=object)
    frames = pd.to_numeric(states["Frame"], errors="coerce").fillna(0).to_numpy(dtype=int)
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for match_id in pd.unique(match_ids):
        row_positions = np.flatnonzero(match_ids == match_id)
        if row_positions.size == 0:
            continue
        order = np.argsort(frames[row_positions], kind="mergesort")
        sorted_positions = row_positions[order]
        cache[match_id] = (frames[sorted_positions], sorted_positions)

    return cache


def _get_state(
    state_lookup: pd.DataFrame,
    state_cache: Mapping[str, tuple[np.ndarray, np.ndarray]],
    match_id: str,
    frame: int,
) -> pd.Series:
    cached = state_cache.get(match_id)
    if cached is None:
        return pd.Series(dtype=float)

    frames, row_positions = cached
    insertion_idx = int(np.searchsorted(frames, frame, side="left"))

    if insertion_idx <= 0:
        nearest_idx = 0
    elif insertion_idx >= len(frames):
        nearest_idx = len(frames) - 1
    else:
        prev_diff = abs(frame - int(frames[insertion_idx - 1]))
        next_diff = abs(int(frames[insertion_idx]) - frame)
        nearest_idx = insertion_idx - 1 if prev_diff <= next_diff else insertion_idx

    return state_lookup.iloc[int(row_positions[nearest_idx])]
