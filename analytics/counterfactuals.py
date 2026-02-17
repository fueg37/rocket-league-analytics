"""Counterfactual tactical simulation and coach-report generation.

This module builds an action library constrained by Rocket League movement/boost limits,
extracts high-leverage decision moments, and scores candidate tactical decisions via
possession-value projections (with win-probability fallback).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from constants import FIELD_HALF_X, FIELD_HALF_Y, REPLAY_FPS, SUPERSONIC_SPEED_UU_PER_SEC
from analytics.possession_value import TransitionValueEstimator, predict_state_value


@dataclass(slots=True)
class ActionConstraint:
    max_speed_uu_per_sec: float
    max_accel_uu_per_sec2: float
    boost_cost: float
    duration_s: float


@dataclass(slots=True)
class TacticalAction:
    name: str
    role_targets: tuple[str, ...]
    attack_bias: float
    pressure_delta: float
    possession_delta: float
    constraint: ActionConstraint


@dataclass(slots=True)
class ReplayPriors:
    speed_p95: float
    accel_p95: float
    boost_usage_p95: float


ACTION_LIBRARY: dict[str, TacticalAction] = {
    "challenge_now": TacticalAction(
        name="challenge_now",
        role_targets=("first man",),
        attack_bias=0.35,
        pressure_delta=0.20,
        possession_delta=0.08,
        constraint=ActionConstraint(max_speed_uu_per_sec=2300, max_accel_uu_per_sec2=950, boost_cost=18, duration_s=1.2),
    ),
    "fake_challenge": TacticalAction(
        name="fake_challenge",
        role_targets=("first man", "second man"),
        attack_bias=0.05,
        pressure_delta=0.12,
        possession_delta=0.03,
        constraint=ActionConstraint(max_speed_uu_per_sec=2000, max_accel_uu_per_sec2=750, boost_cost=8, duration_s=1.0),
    ),
    "rotate_backpost": TacticalAction(
        name="rotate_backpost",
        role_targets=("second man", "third man"),
        attack_bias=-0.28,
        pressure_delta=-0.05,
        possession_delta=-0.01,
        constraint=ActionConstraint(max_speed_uu_per_sec=2100, max_accel_uu_per_sec2=700, boost_cost=24, duration_s=1.8),
    ),
    "third_man_hold": TacticalAction(
        name="third_man_hold",
        role_targets=("third man",),
        attack_bias=-0.10,
        pressure_delta=0.04,
        possession_delta=0.01,
        constraint=ActionConstraint(max_speed_uu_per_sec=1650, max_accel_uu_per_sec2=520, boost_cost=5, duration_s=1.4),
    ),
    "shadow_defend": TacticalAction(
        name="shadow_defend",
        role_targets=("second man", "third man"),
        attack_bias=-0.2,
        pressure_delta=0.02,
        possession_delta=0.02,
        constraint=ActionConstraint(max_speed_uu_per_sec=1900, max_accel_uu_per_sec2=600, boost_cost=10, duration_s=1.5),
    ),
}

RANK_WEIGHT_ABS_SWING = 0.5
RANK_WEIGHT_LEVERAGE = 0.3
RANK_WEIGHT_CONFIDENCE = 0.2
ACTIONABLE_SWING_EPSILON = 0.02
ACTIONABLE_CONFIDENCE_FLOOR = 0.45
DEDUP_FRAME_DISTANCE = int(1.5 * REPLAY_FPS)


def derive_replay_priors(states: pd.DataFrame) -> ReplayPriors:
    if states is None or states.empty:
        return ReplayPriors(speed_p95=SUPERSONIC_SPEED_UU_PER_SEC, accel_p95=900.0, boost_usage_p95=35.0)

    speed = pd.to_numeric(states.get("BallSpeedUUPerSec", pd.Series(dtype=float)), errors="coerce").fillna(0)
    vel_y = pd.to_numeric(states.get("BallVelY", pd.Series(dtype=float)), errors="coerce").fillna(0)
    accel = vel_y.diff().abs() * REPLAY_FPS
    boost_pressure = (
        pd.to_numeric(states.get("TeamBoostAvgBlue", pd.Series(dtype=float)), errors="coerce").fillna(0)
        - pd.to_numeric(states.get("TeamBoostAvgOrange", pd.Series(dtype=float)), errors="coerce").fillna(0)
    ).abs()
    return ReplayPriors(
        speed_p95=float(np.nanpercentile(speed, 95)) if len(speed) else SUPERSONIC_SPEED_UU_PER_SEC,
        accel_p95=float(np.nanpercentile(accel, 95)) if len(accel) else 900.0,
        boost_usage_p95=float(np.nanpercentile(boost_pressure, 95)) if len(boost_pressure) else 35.0,
    )


def build_action_library(
    snapshot: Mapping[str, float],
    priors: ReplayPriors,
    candidate_actions: Iterable[str] | None = None,
) -> list[TacticalAction]:
    names = list(candidate_actions) if candidate_actions else list(ACTION_LIBRARY.keys())
    boost_available = float(snapshot.get("TeamBoostAvgBlue", snapshot.get("TeamBoostAvgOrange", 0.0)) or 0.0)
    feasible: list[TacticalAction] = []
    for name in names:
        action = ACTION_LIBRARY.get(name)
        if action is None:
            continue
        if action.constraint.boost_cost > max(boost_available, priors.boost_usage_p95):
            continue
        if action.constraint.max_speed_uu_per_sec > max(priors.speed_p95 * 1.2, SUPERSONIC_SPEED_UU_PER_SEC):
            continue
        if action.constraint.max_accel_uu_per_sec2 > priors.accel_p95 * 1.4:
            continue
        feasible.append(action)
    return feasible


def extract_decision_moments(
    momentum_series: pd.Series,
    win_prob_df: pd.DataFrame,
    *,
    top_n: int = 24,
    min_spacing_seconds: int = 5,
) -> pd.DataFrame:
    if momentum_series is None or momentum_series.empty:
        return pd.DataFrame(columns=["Frame", "Time", "Leverage", "WindowStartFrame", "WindowEndFrame"])

    mom = momentum_series.astype(float)
    mom_grad = mom.diff().abs().fillna(0)
    mom_signal = (mom.abs() / max(mom.abs().quantile(0.95), 1.0)) + (mom_grad / max(mom_grad.quantile(0.95), 1.0))

    wp_signal = pd.Series(0.0, index=mom.index)
    if win_prob_df is not None and not win_prob_df.empty and "Time" in win_prob_df.columns and "WinProb" in win_prob_df.columns:
        wp = win_prob_df[["Time", "WinProb"]].copy().dropna()
        wp["Time"] = pd.to_numeric(wp["Time"], errors="coerce")
        wp["WinProb"] = pd.to_numeric(wp["WinProb"], errors="coerce")
        wp = wp.dropna().sort_values("Time")
        if not wp.empty:
            wp["swing"] = wp["WinProb"].diff().abs().fillna(0) / 100.0
            wp_signal = pd.Series(np.interp(mom.index.to_numpy(float), wp["Time"].to_numpy(), wp["swing"].to_numpy()), index=mom.index)

    leverage = mom_signal + wp_signal
    candidate = leverage.sort_values(ascending=False)

    selected: list[dict[str, float]] = []
    chosen_times: list[float] = []
    for t, score in candidate.items():
        t_val = float(t)
        if any(abs(t_val - prior) < min_spacing_seconds for prior in chosen_times):
            continue
        chosen_times.append(t_val)
        frame = int(round(t_val * REPLAY_FPS))
        selected.append(
            {
                "Frame": frame,
                "Time": t_val,
                "Leverage": float(score),
                "WindowStartFrame": max(0, frame - int(2.5 * REPLAY_FPS)),
                "WindowEndFrame": frame + int(2.5 * REPLAY_FPS),
            }
        )
        if len(selected) >= top_n:
            break

    return pd.DataFrame(selected)


def score_candidate_actions(
    snapshot: Mapping[str, float],
    actions: Iterable[TacticalAction],
    *,
    team: str,
    value_model: TransitionValueEstimator | None,
    win_prob_df: pd.DataFrame | None,
    reference_time: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    base = pd.Series(snapshot).copy()
    team_key = "Blue" if team == "Blue" else "Orange"
    pressure_col = f"Pressure{team_key}"
    poss_col = f"{team_key}PossessionBelief"

    baseline_value = predict_state_value(base, model=value_model).get("10s", 0.0)
    wp_baseline = _local_wp(win_prob_df, reference_time)

    for action in actions:
        post = base.copy()
        post["BallPosY"] = float(post.get("BallPosY", 0.0)) + action.attack_bias * FIELD_HALF_Y * 0.2 * (1 if team == "Blue" else -1)
        post["BallPosX"] = float(np.clip(post.get("BallPosX", 0.0), -FIELD_HALF_X, FIELD_HALF_X))
        post[pressure_col] = float(np.clip(float(post.get(pressure_col, 0.0)) + action.pressure_delta, 0.0, 1.0))
        post[poss_col] = float(np.clip(float(post.get(poss_col, 0.5)) + action.possession_delta, 0.0, 1.0))

        model_value = predict_state_value(post, model=value_model).get("10s", baseline_value)
        model_swing = (model_value - baseline_value) * (1 if team == "Blue" else -1)

        wp_future = _local_wp(win_prob_df, reference_time + action.constraint.duration_s)
        wp_swing = (wp_future - wp_baseline) * 0.01
        if not np.isfinite(model_swing):
            expected_swing = wp_swing
            source = "win_probability"
        else:
            expected_swing = 0.75 * model_swing + 0.25 * wp_swing
            source = "possession_value_graph"

        confidence = float(np.clip(0.55 + 0.2 * post.get(pressure_col, 0.0) + 0.25 * min(1.0, action.constraint.max_speed_uu_per_sec / SUPERSONIC_SPEED_UU_PER_SEC), 0.0, 1.0))
        rows.append(
            {
                "Action": action.name,
                "ExpectedSwing": float(expected_swing),
                "Confidence": confidence,
                "RoleTargets": ", ".join(action.role_targets),
                "ModelSource": source,
            }
        )
    return pd.DataFrame(rows).sort_values("ExpectedSwing", ascending=False).reset_index(drop=True)


def build_coach_report(
    states: pd.DataFrame,
    momentum_series: pd.Series,
    win_prob_df: pd.DataFrame,
    rotation_timeline: pd.DataFrame,
    rotation_summary: pd.DataFrame,
    *,
    team: str = "Blue",
    top_n: int = 5,
) -> pd.DataFrame:
    if states is None or states.empty:
        return pd.DataFrame()

    priors = derive_replay_priors(states)
    moments = extract_decision_moments(momentum_series, win_prob_df, top_n=max(15, top_n * 4))
    if moments.empty:
        return pd.DataFrame()

    rows = []
    for m in moments.itertuples(index=False):
        state_row = _nearest_state(states, int(m.Frame))
        actions = build_action_library(state_row, priors)
        if not actions:
            continue
        scored = score_candidate_actions(
            state_row,
            actions,
            team=team,
            value_model=None,
            win_prob_df=win_prob_df,
            reference_time=float(m.Time),
        )
        if scored.empty:
            continue
        best = scored.iloc[0]
        baseline = scored[scored["Action"] == "third_man_hold"]
        baseline_swing = float(baseline["ExpectedSwing"].iloc[0]) if not baseline.empty else float(scored["ExpectedSwing"].median())
        missed = float(best["ExpectedSwing"] - baseline_swing)
        role = _infer_role(rotation_timeline, rotation_summary, int(m.Frame), team)
        rows.append(
            {
                "Frame": int(m.Frame),
                "Time": float(m.Time),
                "Leverage": float(m.Leverage),
                "ExpectedSwing": float(best["ExpectedSwing"]),
                "MissedSwing": missed,
                "Confidence": float(best["Confidence"]),
                "RecommendedAction": str(best["Action"]),
                "Role": role,
                "RecommendationText": _recommendation_text(role, str(best["Action"])),
                "ModelSource": str(best["ModelSource"]),
                "ClipKey": f"frame:{int(m.Frame)}|window:{int(m.WindowStartFrame)}-{int(m.WindowEndFrame)}",
            }
        )

    report = pd.DataFrame(rows)
    if report.empty:
        return report

    report["AbsExpectedSwing"] = report["ExpectedSwing"].abs()
    report["NormAbsExpectedSwing"] = _normalize_component(report["AbsExpectedSwing"])
    report["NormLeverage"] = _normalize_component(report["Leverage"])
    report["NormConfidence"] = _normalize_component(report["Confidence"])
    report["RankScore"] = (
        RANK_WEIGHT_ABS_SWING * report["NormAbsExpectedSwing"]
        + RANK_WEIGHT_LEVERAGE * report["NormLeverage"]
        + RANK_WEIGHT_CONFIDENCE * report["NormConfidence"]
    )
    report["ActionabilityFlag"] = (
        (report["AbsExpectedSwing"] > ACTIONABLE_SWING_EPSILON)
        & (report["Confidence"] > ACTIONABLE_CONFIDENCE_FLOOR)
    )

    ranked = _sort_ranked_report(report)
    actionable = _dedupe_report_windows(ranked[ranked["ActionabilityFlag"]], frame_distance=DEDUP_FRAME_DISTANCE)
    if len(actionable) >= top_n:
        final = actionable.head(top_n)
    else:
        fallback_pool = ranked[~ranked.index.isin(actionable.index)]
        combined = pd.concat([actionable, fallback_pool], axis=0)
        final = _dedupe_report_windows(combined, frame_distance=DEDUP_FRAME_DISTANCE).head(top_n)
    return final.reset_index(drop=True)


def _normalize_component(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    max_value = float(numeric.max()) if not numeric.empty else 0.0
    if max_value <= 0.0:
        return pd.Series(0.0, index=numeric.index, dtype=float)
    return (numeric / max_value).clip(0.0, 1.0)


def _sort_ranked_report(report: pd.DataFrame) -> pd.DataFrame:
    return report.sort_values(
        ["RankScore", "AbsExpectedSwing", "Leverage", "Confidence", "Frame", "RecommendedAction", "Role"],
        ascending=[False, False, False, False, True, True, True],
    )


def _dedupe_report_windows(report: pd.DataFrame, *, frame_distance: int) -> pd.DataFrame:
    selected_indices: list[int] = []
    selected_keys: list[tuple[str, str, int]] = []

    for idx, row in report.iterrows():
        frame = int(row["Frame"])
        key = (str(row["RecommendedAction"]), str(row["Role"]), frame)
        is_duplicate = any(
            prior_action == key[0] and prior_role == key[1] and abs(prior_frame - key[2]) <= frame_distance
            for prior_action, prior_role, prior_frame in selected_keys
        )
        if is_duplicate:
            continue
        selected_indices.append(int(idx))
        selected_keys.append(key)

    if not selected_indices:
        return report.iloc[0:0]
    return report.loc[selected_indices]


def _nearest_state(states: pd.DataFrame, frame: int) -> pd.Series:
    idx = np.argmin(np.abs(pd.to_numeric(states["Frame"], errors="coerce").fillna(0).to_numpy(dtype=float) - float(frame)))
    return states.iloc[int(idx)]


def _local_wp(win_prob_df: pd.DataFrame | None, time_s: float) -> float:
    if win_prob_df is None or win_prob_df.empty or "Time" not in win_prob_df.columns or "WinProb" not in win_prob_df.columns:
        return 50.0
    times = pd.to_numeric(win_prob_df["Time"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    probs = pd.to_numeric(win_prob_df["WinProb"], errors="coerce").fillna(50.0).to_numpy(dtype=float)
    return float(np.interp(time_s, times, probs))


def _infer_role(rotation_timeline: pd.DataFrame, rotation_summary: pd.DataFrame, frame: int, team: str) -> str:
    if rotation_timeline is not None and not rotation_timeline.empty:
        tol = int(2 * REPLAY_FPS)
        window = rotation_timeline[(rotation_timeline["Team"] == team) & (rotation_timeline["Frame"].sub(frame).abs() <= tol)]
        if not window.empty and "Role" in window.columns:
            role = window["Role"].mode().iloc[0]
            return {"1st": "first man", "2nd": "second man"}.get(str(role), "third man")
    if rotation_summary is not None and not rotation_summary.empty and "Time_1st%" in rotation_summary.columns:
        team_rows = rotation_summary[rotation_summary["Team"] == team]
        if not team_rows.empty:
            mean_first = float(team_rows["Time_1st%"].mean())
            if mean_first > 42:
                return "first man"
            if mean_first > 28:
                return "second man"
    return "third man"


def _recommendation_text(role: str, action: str) -> str:
    action_text = action.replace("_", " ")
    return f"As {role}, prioritize '{action_text}' to improve pressure timing while preserving recoverability."
