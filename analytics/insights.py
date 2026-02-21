"""Deterministic key-insight selector for match narrative surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analytics.shot_quality import COL_XG, SHOT_COL_RESULT


@dataclass(frozen=True)
class InsightCard:
    headline: str
    explanation: str
    recommendation: str
    confidence_source_tag: str


def _safe_sum(frame: pd.DataFrame, columns: list[str]) -> float | None:
    for col in columns:
        if col in frame.columns:
            values = pd.to_numeric(frame[col], errors="coerce")
            if values.notna().any():
                return float(values.sum())
    return None


def _goal_count(df: pd.DataFrame, shot_df: pd.DataFrame) -> float | None:
    goals_from_df = _safe_sum(df, ["Goals"])
    if goals_from_df is not None:
        return goals_from_df

    if SHOT_COL_RESULT in shot_df.columns:
        return float((shot_df[SHOT_COL_RESULT].astype(str).str.lower() == "goal").sum())
    return None


def _score_xg_delta(df: pd.DataFrame, shot_df: pd.DataFrame) -> tuple[float, InsightCard] | None:
    if COL_XG not in shot_df.columns:
        return None

    xg_total = pd.to_numeric(shot_df[COL_XG], errors="coerce").sum(min_count=1)
    if pd.isna(xg_total):
        return None
    goals = _goal_count(df, shot_df)
    if goals is None:
        return None

    delta = float(goals - xg_total)
    score = min(1.0, abs(delta) / 1.25)
    if score <= 0.05:
        return None

    outperform = delta >= 0
    card = InsightCard(
        headline=(
            "Clinical finishing outpaced shot quality"
            if outperform
            else "Finishing lagged behind chance quality"
        ),
        explanation=(
            f"You scored {goals:.0f} from {xg_total:.2f} xG ({delta:+.2f} vs expectation), "
            "a decisive efficiency gap in this match."
        ),
        recommendation=(
            "Preserve shot selection patterns and continue prioritizing low-touch finishes in rotation."
            if outperform
            else "Increase first-touch shot prep and rebounding support to convert created chances."
        ),
        confidence_source_tag="high • xG_delta",
    )
    return score, card


def _score_pressure_trend(momentum_series: pd.Series | None) -> tuple[float, InsightCard] | None:
    if momentum_series is None:
        return None
    series = pd.to_numeric(momentum_series, errors="coerce").dropna()
    if series.size < 6:
        return None

    ordered = series.sort_index()
    midpoint = ordered.size // 2
    first_half = ordered.iloc[:midpoint]
    second_half = ordered.iloc[midpoint:]
    if first_half.empty or second_half.empty:
        return None

    shift = float(second_half.mean() - first_half.mean())
    score = min(0.95, abs(shift) / 8.0)
    if score <= 0.08:
        return None

    improved = shift > 0
    card = InsightCard(
        headline=("Pressure control improved across the match" if improved else "Pressure control faded over time"),
        explanation=(
            f"Average momentum shifted by {shift:+.2f} from first half to second half, "
            "indicating a meaningful trend in territorial control."
        ),
        recommendation=(
            "Keep second-man challenge timing aggressive; your late-match pressure profile was a strength."
            if improved
            else "Shorten defensive clear cycles and rotate earlier to avoid late pressure collapse."
        ),
        confidence_source_tag="medium • momentum_trend",
    )
    return score, card


def _score_rotation_risk(coach_report_df: pd.DataFrame) -> tuple[float, InsightCard] | None:
    if coach_report_df is None or coach_report_df.empty or "MissedSwing" not in coach_report_df.columns:
        return None
    swing = pd.to_numeric(coach_report_df["MissedSwing"], errors="coerce").dropna()
    if swing.empty:
        return None

    risk_load = float(swing.abs().sum())
    score = min(0.9, risk_load / 2.5)
    if score <= 0.08:
        return None

    needs_attention = float(swing.mean()) < 0
    card = InsightCard(
        headline=("Rotation discipline is creating leverage loss" if needs_attention else "Rotation windows were mostly stable"),
        explanation=(
            f"Coach opportunities accumulated {risk_load:.2f} weighted swing across {len(swing)} flagged windows, "
            "pointing to repeatable rotation outcomes."
        ),
        recommendation=(
            "Prioritize third-man spacing and earlier handoffs after challenges in midfield."
            if needs_attention
            else "Keep current rotation cadence while tightening recoveries after low-value pushes."
        ),
        confidence_source_tag="medium • coach_rotation",
    )
    return score, card


def _score_defensive_workload(df: pd.DataFrame) -> tuple[float, InsightCard] | None:
    saves = _safe_sum(df, ["Saves"])
    shots_faced = _safe_sum(df, ["Shots Faced", "ShotsFaced"])
    xga = _safe_sum(df, ["xGA"])
    if saves is None or shots_faced is None:
        return None

    workload = (shots_faced / 10.0) + (xga / 2.0 if xga is not None else 0.0)
    score = min(0.88, workload / 2.0)
    if score <= 0.08:
        return None

    card = InsightCard(
        headline="Defensive workload shaped the match",
        explanation=(
            f"The team absorbed {shots_faced:.0f} shots faced with {saves:.0f} saves"
            + (f" and {xga:.2f} xGA" if xga is not None else "")
            + ", forcing repeated high-leverage defensive sequences."
        ),
        recommendation="Invest in cleaner first clears and faster exit support to reduce repeated defensive holds.",
        confidence_source_tag="medium • defensive_workload",
    )
    return score, card


def _score_conversion_efficiency(df: pd.DataFrame, shot_df: pd.DataFrame) -> tuple[float, InsightCard] | None:
    goals = _goal_count(df, shot_df)
    shots = _safe_sum(df, ["Shots"])
    if goals is None:
        return None
    if shots is None and not shot_df.empty:
        shots = float(len(shot_df))
    if shots is None or shots <= 0:
        return None

    conversion = float(goals / shots)
    baseline = 0.18
    delta = conversion - baseline
    score = min(0.82, abs(delta) / 0.22)
    if score <= 0.08:
        return None

    card = InsightCard(
        headline=("Shot conversion was an advantage" if delta >= 0 else "Shot conversion limited scoring output"),
        explanation=(
            f"Conversion finished at {conversion:.1%} across {shots:.0f} shots ({delta:+.1%} vs {baseline:.0%} baseline)."
        ),
        recommendation=(
            "Keep feeding high-tempo shot chains after recoveries; finishing rate is currently above baseline."
            if delta >= 0
            else "Add infield passing or rebound pressure to raise shot quality before release."
        ),
        confidence_source_tag="high • conversion_efficiency",
    )
    return score, card


def build_key_insight(
    df: pd.DataFrame,
    shot_df: pd.DataFrame,
    momentum_series: pd.Series | None,
    coach_report_df: pd.DataFrame | None,
) -> dict[str, str]:
    """Return one deterministic top narrative for the match.

    Ranked heuristics:
    1) xG over/under-performance
    2) pressure trend
    3) rotation risk
    4) defensive workload
    5) conversion efficiency
    """
    frame = df if df is not None else pd.DataFrame()
    shots = shot_df if shot_df is not None else pd.DataFrame()
    report = coach_report_df if coach_report_df is not None else pd.DataFrame()

    candidates: list[tuple[float, int, InsightCard]] = []
    ranked_heuristics = [
        _score_xg_delta(frame, shots),
        _score_pressure_trend(momentum_series),
        _score_rotation_risk(report),
        _score_defensive_workload(frame),
        _score_conversion_efficiency(frame, shots),
    ]

    for idx, candidate in enumerate(ranked_heuristics):
        if candidate is None:
            continue
        score, card = candidate
        if np.isfinite(score):
            candidates.append((float(score), idx, card))

    if not candidates:
        fallback = InsightCard(
            headline="Insufficient data for a targeted key insight",
            explanation="Required match metrics were missing, so no ranked heuristic could be evaluated reliably.",
            recommendation="Record complete shot, momentum, and coach-opportunity streams to unlock deterministic insights.",
            confidence_source_tag="low • fallback_missing_metrics",
        )
        return fallback.__dict__

    best = sorted(candidates, key=lambda item: (-item[0], item[1]))[0][2]
    return best.__dict__
