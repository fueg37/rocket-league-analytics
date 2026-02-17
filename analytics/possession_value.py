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

    max_frame = int(game_df.index.max()) if len(game_df.index) else 0
    rows: list[dict[str, float | int | str]] = []
    for frame in ball_df.index.astype(int):
        b_row = ball_df.loc[frame]
        bx = float(b_row.get("pos_x", 0.0) or 0.0)
        by = float(b_row.get("pos_y", 0.0) or 0.0)
        bz = float(b_row.get("pos_z", 0.0) or 0.0)
        bvx = float(b_row.get("vel_x", 0.0) or 0.0)
        bvy = float(b_row.get("vel_y", 0.0) or 0.0)
        bvz = float(b_row.get("vel_z", 0.0) or 0.0)
        nearest_blue, nearest_orange = _nearest_team_distances(frame, bx, by, player_pos)
        boost_blue, boost_orange = _team_boost_context(frame, game_df, player_pos)

        blue_belief = _poss_belief(nearest_blue, nearest_orange, by, "Blue")
        orange_belief = 1.0 - blue_belief
        pressure_blue = float(max(0.0, 1.0 - (nearest_blue / 2000.0)))
        pressure_orange = float(max(0.0, 1.0 - (nearest_orange / 2000.0)))

        rows.append(
            {
                "MatchID": match_id,
                "Frame": int(frame),
                "BluePossessionBelief": blue_belief,
                "OrangePossessionBelief": orange_belief,
                "BallPosX": bx,
                "BallPosY": by,
                "BallPosZ": bz,
                "BallVelX": bvx,
                "BallVelY": bvy,
                "BallVelZ": bvz,
                "BallSpeedUUPerSec": float(np.sqrt(bvx**2 + bvy**2 + bvz**2)),
                "NearestBlueDist": nearest_blue,
                "NearestOrangeDist": nearest_orange,
                "TeamBoostAvgBlue": boost_blue,
                "TeamBoostAvgOrange": boost_orange,
                "PressureBlue": pressure_blue,
                "PressureOrange": pressure_orange,
                "BlueAttacking": float(by > 0.0),
                "OrangeAttacking": float(by < 0.0),
                "SecondsRemaining": max(0.0, (max_frame - int(frame)) / float(REPLAY_FPS)),
            }
        )

    return pd.DataFrame(rows, columns=CANONICAL_STATE_COLUMNS)


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
    state_lookup = states.set_index(["MatchID", "Frame"], drop=False)
    valued = events.copy()
    value_3s = []
    value_10s = []

    for row in valued.itertuples(index=False):
        match_id = getattr(row, "MatchID", None)
        frame = int(getattr(row, "Frame", 0) or 0)
        team = getattr(row, "Team", "Blue") or "Blue"
        post_frame = int(getattr(row, "PostFrame", frame + 1) or (frame + 1))

        pre_state = _get_state(state_lookup, match_id, frame)
        post_state = _get_state(state_lookup, match_id, post_frame)
        pre_pred = estimator.predict_state_value(pre_state)
        post_pred = estimator.predict_state_value(post_state)

        direction = 1.0 if team == "Blue" else -1.0
        value_3s.append(direction * (post_pred["3s"] - pre_pred["3s"]))
        value_10s.append(direction * (post_pred["10s"] - pre_pred["10s"]))

    valued["ValueDelta_3s"] = np.round(value_3s, 4)
    valued["ValueDelta_10s"] = np.round(value_10s, 4)
    valued["VAEP"] = valued["ValueDelta_3s"]
    return valued


def _get_state(state_lookup: pd.DataFrame, match_id: str, frame: int) -> pd.Series:
    key = (match_id, frame)
    if key in state_lookup.index:
        return state_lookup.loc[key]
    match_states = state_lookup[state_lookup["MatchID"] == match_id]
    if match_states.empty:
        return pd.Series(dtype=float)
    nearest_idx = int(np.argmin(np.abs(match_states["Frame"].to_numpy(dtype=int) - frame)))
    return match_states.iloc[nearest_idx]

