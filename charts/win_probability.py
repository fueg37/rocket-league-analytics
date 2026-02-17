"""Win probability chart builders and event extraction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from charts.theme import apply_chart_theme
from constants import REPLAY_FPS, TEAM_COLORS


def extract_goal_events(proto: Any, pid_team: dict[str, str], pid_name_map: dict[str, str] | None = None, max_frame: int | None = None) -> list[dict[str, Any]]:
    """Extract goal timeline events for chart annotation layers."""
    events: list[dict[str, Any]] = []
    if not hasattr(proto, "game_metadata") or not hasattr(proto.game_metadata, "goals"):
        return events

    for idx, goal in enumerate(proto.game_metadata.goals, start=1):
        frame = int(getattr(goal, "frame_number", getattr(goal, "frame", 0)) or 0)
        if max_frame is not None:
            frame = min(frame, int(max_frame))

        scorer_pid = str(goal.player_id.id) if hasattr(goal, "player_id") and hasattr(goal.player_id, "id") else ""
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


def _state_series(times: np.ndarray, values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.where(mask, times, np.nan)
    y = np.where(mask, values, np.nan)
    return x, y


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
    probs = pd.to_numeric(win_prob_df["WinProb"], errors="coerce").fillna(50).clip(0, 100).to_numpy(dtype=float)
    score_diff = pd.to_numeric(win_prob_df.get("ScoreDiff", 0), errors="coerce").fillna(0).to_numpy(dtype=float)

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
        fillcolor="rgba(148,163,184,0.12)",
        line=dict(width=0),
        layer="below",
    )

    # Advantage area fills around 50% threshold
    base = np.full_like(probs, 50.0)
    above = np.where(probs >= 50, probs, 50)
    below = np.where(probs <= 50, probs, 50)

    fig.add_trace(go.Scatter(x=times, y=base, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=times, y=above, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(0,123,255,0.18)", hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=times, y=base, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=times, y=below, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(255,153,0,0.18)", hoverinfo="skip", showlegend=False))

    # Segmented semantic lines
    favored_blue = probs > 55
    tossup = (probs >= 45) & (probs <= 55)
    favored_orange = probs < 45
    segments = [
        (favored_blue, "Favored Blue", blue_color),
        (tossup, "Toss-up", "#c9d1d9"),
        (favored_orange, "Favored Orange", orange_color),
    ]

    customdata = np.column_stack([score_diff, np.full_like(probs, model_label, dtype=object)])
    hovertemplate = (
        "<b>t=%{x:.0f}s</b><br>Blue Win: %{y:.1f}%"
        "<br>Score Diff (Blue-Orange): %{customdata[0]:.0f}"
        "<br>Model: %{customdata[1]}<extra></extra>"
    )

    for mask, name, color in segments:
        x_seg, y_seg = _state_series(times, probs, mask)
        fig.add_trace(
            go.Scatter(
                x=x_seg,
                y=y_seg,
                mode="lines",
                line=dict(color=color, width=3),
                name=name,
                legendgroup=name,
                customdata=customdata,
                hovertemplate=hovertemplate,
                connectgaps=False,
            )
        )

    # Threshold and phase markers
    fig.add_shape(
        type="line",
        x0=float(times.min()),
        y0=50,
        x1=float(times.max()),
        y1=50,
        line=dict(color="rgba(201,209,217,0.7)", width=1, dash="dot"),
    )

    if is_overtime:
        fig.add_vline(x=300, line_dash="dash", line_color="rgba(255,204,0,0.8)")
        fig.add_annotation(x=300, y=103, text="OT Start", showarrow=False, font=dict(size=10, color="#ffcc00"))
        fig.add_vrect(x0=300, x1=float(times.max()), fillcolor="rgba(255,204,0,0.07)", line_width=0, layer="below")

    # Goal event layer
    for evt in events or []:
        color = blue_color if evt.get("team") == "Blue" else orange_color
        x = float(evt.get("time", 0))
        fig.add_vline(x=x, line_width=1, line_dash="dot", line_color=f"{color}")
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[99],
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
        xaxis=dict(
            title="Time (Seconds)",
            showgrid=True,
            dtick=30,
            rangeslider=dict(visible=True, thickness=0.08, bgcolor="rgba(255,255,255,0.05)"),
        ),
        height=320,
        margin=dict(l=24, r=24, t=70, b=26),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    if subtitle:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0,
            y=1.2,
            text=subtitle,
            showarrow=False,
            xanchor="left",
            font=dict(size=10, color="#c9d1d9"),
        )

    return fig
