"""Win probability chart builders and event extraction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from charts.theme import apply_chart_theme
from constants import REPLAY_FPS, TEAM_COLORS


STATE_BLUE = "Favored Blue"
STATE_TOSSUP = "Toss-up"
STATE_ORANGE = "Favored Orange"


def extract_goal_events(
    proto: Any,
    pid_team: dict[str, str],
    pid_name_map: dict[str, str] | None = None,
    max_frame: int | None = None,
) -> list[dict[str, Any]]:
    """Extract goal timeline events for chart annotation layers."""
    events: list[dict[str, Any]] = []
    if not hasattr(proto, "game_metadata") or not hasattr(proto.game_metadata, "goals"):
        return events

    for idx, goal in enumerate(proto.game_metadata.goals, start=1):
        frame = int(getattr(goal, "frame_number", getattr(goal, "frame", 0)) or 0)
        if max_frame is not None:
            frame = min(frame, int(max_frame))

        scorer_pid = (
            str(goal.player_id.id)
            if hasattr(goal, "player_id") and hasattr(goal.player_id, "id")
            else ""
        )
        team = pid_team.get(scorer_pid, "Blue")
        scorer_name = pid_name_map.get(scorer_pid, "Unknown") if pid_name_map else "Unknown"

        events.append(
            {
                "time": frame / float(REPLAY_FPS),
                "team": team,
                "label": f"{team[0]} Goal",
                "hover": f"Goal {idx}<br>{team}: {scorer_name}",
            }
        )

    events.sort(key=lambda e: e["time"])
    return events


def _smooth_probability(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered rolling mean for visual smoothness without changing data semantics."""
    if window <= 1:
        return values
    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def _assign_states(probs: np.ndarray, low: float = 45.0, high: float = 55.0) -> np.ndarray:
    states = np.full(len(probs), STATE_TOSSUP, dtype=object)
    states[probs > high] = STATE_BLUE
    states[probs < low] = STATE_ORANGE
    return states


def _debounce_states(states: np.ndarray, min_run: int = 3) -> np.ndarray:
    """Reduce one-second flicker between adjacent state bands."""
    if len(states) <= 2 or min_run <= 1:
        return states

    out = states.copy()
    i = 0
    n = len(out)
    while i < n:
        j = i + 1
        while j < n and out[j] == out[i]:
            j += 1
        run_len = j - i

        if run_len < min_run:
            left = out[i - 1] if i > 0 else None
            right = out[j] if j < n else None
            replacement = left if left == right and left is not None else (left or right or out[i])
            out[i:j] = replacement
        i = j

    return out


def _masked_series(times: np.ndarray, values: np.ndarray, states: np.ndarray, target: str) -> tuple[np.ndarray, np.ndarray]:
    mask = states == target
    return np.where(mask, times, np.nan), np.where(mask, values, np.nan)


def build_win_probability_chart(
    win_prob_df: pd.DataFrame,
    is_overtime: bool,
    model_meta: dict[str, Any] | None = None,
    events: list[dict[str, Any]] | None = None,
    tier: str = "detail",
    title_prefix: str = "🏆 ",
) -> go.Figure:
    """Build a semantic win-probability chart with event overlays."""
    fig = apply_chart_theme(go.Figure(), tier=tier, intent="neutral")
    if win_prob_df is None or win_prob_df.empty:
        fig.update_layout(title=f"{title_prefix}Win Probability")
        return fig

    times = pd.to_numeric(win_prob_df["Time"], errors="coerce").fillna(0).to_numpy(dtype=float)
    probs_raw = (
        pd.to_numeric(win_prob_df["WinProb"], errors="coerce").fillna(50).clip(0, 100).to_numpy(dtype=float)
    )
    score_diff = (
        pd.to_numeric(win_prob_df.get("ScoreDiff", 0), errors="coerce").fillna(0).to_numpy(dtype=float)
    )
    probs = _smooth_probability(probs_raw, window=5)

    states = _assign_states(probs)
    states = _debounce_states(states, min_run=3)

    model_label = (model_meta or {}).get("label", "Heuristic")
    subtitle = (model_meta or {}).get("subtitle", "")

    blue_color = TEAM_COLORS["Blue"]["primary"]
    orange_color = TEAM_COLORS["Orange"]["primary"]

    # Toss-up confidence band
    fig.add_shape(
        type="rect",
        x0=float(times.min()),
        x1=float(times.max()),
        y0=45,
        y1=55,
        fillcolor="rgba(148,163,184,0.10)",
        line=dict(width=0),
        layer="below",
    )

    # Threshold-centered fill for directional control.
    base = np.full_like(probs, 50.0)
    above = np.where(probs >= 50, probs, 50)
    below = np.where(probs <= 50, probs, 50)
    fig.add_trace(go.Scatter(x=times, y=base, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=times,
            y=above,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0,123,255,0.16)",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(go.Scatter(x=times, y=base, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=times,
            y=below,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(255,153,0,0.16)",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    customdata = np.column_stack([score_diff, np.full_like(probs, model_label, dtype=object), probs_raw])
    hovertemplate = (
        "<b>t=%{x:.0f}s</b><br>Blue Win: %{y:.1f}%"
        "<br>Raw model output: %{customdata[2]:.1f}%"
        "<br>Score Diff (Blue-Orange): %{customdata[0]:.0f}"
        "<br>Model: %{customdata[1]}<extra></extra>"
    )

    for name, color in (
        (STATE_BLUE, blue_color),
        (STATE_TOSSUP, "#c9d1d9"),
        (STATE_ORANGE, orange_color),
    ):
        x_seg, y_seg = _masked_series(times, probs, states, name)
        fig.add_trace(
            go.Scatter(
                x=x_seg,
                y=y_seg,
                mode="lines",
                line=dict(color=color, width=3, shape="spline", smoothing=0.55),
                name=name,
                legendgroup=name,
                customdata=customdata,
                hovertemplate=hovertemplate,
                connectgaps=False,
            )
        )

    # Neutral threshold line.
    fig.add_shape(
        type="line",
        x0=float(times.min()),
        y0=50,
        x1=float(times.max()),
        y1=50,
        line=dict(color="rgba(201,209,217,0.65)", width=1, dash="dot"),
    )

    if is_overtime:
        fig.add_vline(x=300, line_dash="dash", line_color="rgba(255,204,0,0.75)")
        fig.add_annotation(
            x=300,
            y=102,
            text="OT Start",
            showarrow=False,
            font=dict(size=10, color="#ffcc00"),
        )
        fig.add_vrect(
            x0=300,
            x1=float(times.max()),
            fillcolor="rgba(255,204,0,0.06)",
            line_width=0,
            layer="below",
        )

    # Goal event layer: short top stems to avoid full-chart clutter.
    for evt in events or []:
        color = blue_color if evt.get("team") == "Blue" else orange_color
        x = float(evt.get("time", 0))
        fig.add_shape(
            type="line",
            x0=x,
            x1=x,
            y0=84,
            y1=100,
            line=dict(color=color, width=1, dash="dot"),
        )
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[100],
                mode="markers",
                marker=dict(size=8, symbol="diamond", color=color, line=dict(color="white", width=1)),
                name=evt.get("label", "Goal"),
                showlegend=False,
                hovertemplate=f"{evt.get('hover', 'Goal')}<br>t={x:.1f}s<extra></extra>",
            )
        )

    title = f"{title_prefix}Win Probability" + (" (Overtime)" if is_overtime else "")
    fig.update_layout(
        title=title,
        yaxis=dict(title="Blue Win %", range=[0, 100], dtick=25, showgrid=True),
        xaxis=dict(title="Time (Seconds)", showgrid=True, dtick=30),
        height=320,
        margin=dict(l=24, r=24, t=84, b=44),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
    )

    if subtitle:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0,
            y=1.11,
            text=subtitle,
            showarrow=False,
            xanchor="left",
            font=dict(size=10, color="#c9d1d9"),
        )

    return fig
