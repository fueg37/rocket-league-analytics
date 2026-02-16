"""Reusable chart factories for player comparisons."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from constants import GOAL_HALF_W, GOAL_HEIGHT, REPLAY_FPS, TEAM_COLORS
from charts.theme import apply_chart_theme
from analytics.shot_quality import (
    COL_ON_TARGET,
    COL_TARGET_X,
    COL_TARGET_Z,
    COL_XG,
    COL_XGOT,
    SHOT_COL_FRAME,
    SHOT_COL_PLAYER,
    SHOT_COL_RESULT,
    SHOT_COL_TEAM,
)

_TEAM_ACCENT = {
    "Blue": TEAM_COLORS["Blue"]["primary"],
    "Orange": TEAM_COLORS["Orange"]["primary"],
}

_STEM_COLOR = {
    "Blue": "rgba(74, 124, 196, 0.35)",
    "Orange": "rgba(196, 138, 74, 0.35)",
}


def _format_values(series: pd.Series) -> list[str]:
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    if (values % 1 == 0).all():
        return [f"{v:,.0f}" for v in values]
    if values.abs().max() < 10:
        return [f"{v:,.2f}" for v in values]
    return [f"{v:,.1f}" for v in values]


def player_rank_lollipop(df, metric_col, name_col="Name", team_col="Team"):
    """Render a player ranking lollipop chart for one metric.

    Sorts descending by metric and uses muted stems with team-color marker accents.
    """
    cols = [name_col, team_col, metric_col]
    rank_df = df[cols].copy()
    rank_df[metric_col] = pd.to_numeric(rank_df[metric_col], errors="coerce").fillna(0)
    rank_df = rank_df.sort_values(metric_col, ascending=False).reset_index(drop=True)

    fig = go.Figure()
    labels = _format_values(rank_df[metric_col])

    for i, row in rank_df.iterrows():
        team = row[team_col]
        stem_color = _STEM_COLOR.get(team, "rgba(165, 171, 184, 0.35)")
        accent = _TEAM_ACCENT.get(team, "#9aa4b2")
        val = float(row[metric_col])

        fig.add_shape(
            type="line",
            x0=0,
            x1=val,
            y0=i,
            y1=i,
            line=dict(color=stem_color, width=2),
        )
        fig.add_trace(
            go.Scatter(
                x=[val],
                y=[i],
                mode="markers+text",
                marker=dict(size=11, color="rgba(255,255,255,0.9)", line=dict(color=accent, width=2.5)),
                text=[labels[i]],
                textposition="middle right",
                textfont=dict(color=accent, size=11),
                hovertemplate=f"<b>{row[name_col]}</b><br>{metric_col}: %{{x}}<extra>{team}</extra>",
                showlegend=False,
            )
        )

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(rank_df))),
        ticktext=rank_df[name_col].tolist(),
        autorange="reversed",
        title=None,
    )
    fig.update_xaxes(title=metric_col, rangemode="tozero")
    fig.update_layout(title=metric_col)

    return apply_chart_theme(fig, tier="support")


def comparison_dumbbell(
    df,
    entity_col,
    left_col,
    right_col,
    left_label,
    right_label,
    sort_by=None,
):
    """Render a dumbbell chart for two-value comparisons per entity.

    Args:
        sort_by: Optional ordering mode. Use "delta" or "abs_delta" to sort by
            directional / absolute difference. Defaults to None, preserving input order.
    """
    comp_df = df[[entity_col, left_col, right_col]].copy()
    comp_df[left_col] = pd.to_numeric(comp_df[left_col], errors="coerce").fillna(0)
    comp_df[right_col] = pd.to_numeric(comp_df[right_col], errors="coerce").fillna(0)
    comp_df["delta"] = comp_df[right_col] - comp_df[left_col]
    comp_df["abs_delta"] = comp_df["delta"].abs()

    if sort_by == "delta":
        comp_df = comp_df.sort_values("delta", ascending=True).reset_index(drop=True)
    elif sort_by == "abs_delta":
        comp_df = comp_df.sort_values("abs_delta", ascending=False).reset_index(drop=True)
    else:
        comp_df = comp_df.reset_index(drop=True)

    left_color = "#8C9AAD"   # muted slate
    right_color = "#00CC96"  # positive endpoint accent
    connector_color = "rgba(165, 171, 184, 0.45)"

    fig = go.Figure()

    for i, row in comp_df.iterrows():
        left_val = float(row[left_col])
        right_val = float(row[right_col])
        delta = right_val - left_val
        if delta > 0:
            delta_text = f"+{abs(delta):.2f}"
        elif delta < 0:
            delta_text = f"-{abs(delta):.2f}"
        else:
            delta_text = "±0.00"

        fig.add_shape(
            type="line",
            x0=left_val,
            x1=right_val,
            y0=i,
            y1=i,
            line=dict(color=connector_color, width=2.5),
            layer="below",
        )

        fig.add_trace(
            go.Scatter(
                x=[left_val],
                y=[i],
                mode="markers+text",
                marker=dict(size=11, color="white", line=dict(color=left_color, width=2.5)),
                text=[_format_values(pd.Series([left_val]))[0]],
                textposition="middle left",
                textfont=dict(size=10, color=left_color),
                hovertemplate=f"<b>{row[entity_col]}</b><br>{left_label}: %{{x}}<extra></extra>",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[right_val],
                y=[i],
                mode="markers+text",
                marker=dict(size=11, color="white", line=dict(color=right_color, width=2.5)),
                text=[_format_values(pd.Series([right_val]))[0]],
                textposition="middle right",
                textfont=dict(size=10, color=right_color),
                hovertemplate=f"<b>{row[entity_col]}</b><br>{right_label}: %{{x}}<extra></extra>",
                showlegend=False,
            )
        )

        mid_x = (left_val + right_val) / 2
        fig.add_annotation(
            x=mid_x,
            y=i,
            text=delta_text,
            showarrow=False,
            font=dict(size=10, color="#D7DEE9"),
            yshift=-16,
            align="center",
        )

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(comp_df))),
        ticktext=comp_df[entity_col].tolist(),
        autorange="reversed",
        title=None,
    )
    fig.update_xaxes(title=f"{left_label} vs {right_label}", zeroline=False)
    fig.update_layout(
        margin=dict(l=10, r=10, t=45, b=10),
        title=f"{left_label} vs {right_label} (Δ = {right_label} − {left_label})",
    )

    # Legend-style endpoint labels in-chart for quick semantic mapping.
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.0,
        y=1.08,
        text=f"<span style='color:{left_color}'>●</span> {left_label}",
        showarrow=False,
        xanchor="left",
        font=dict(size=11),
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.18,
        y=1.08,
        text=f"<span style='color:{right_color}'>●</span> {right_label}",
        showarrow=False,
        xanchor="left",
        font=dict(size=11),
    )

    return apply_chart_theme(fig, tier="support")


def goal_mouth_scatter(df, team=None, player=None, include_xgot=True, on_target_only=True):
    """Render a goal-mouth scatter using post-shot target coordinates.

    Args:
        df: Shot dataframe that includes TargetX/TargetZ and shot metadata.
        team: Optional team filter ("Blue"/"Orange").
        player: Optional player-name filter.
        include_xgot: Scale marker size by xGOT values.
        on_target_only: Keep only shots marked as on-target.
    """
    fig = go.Figure()

    required_cols = {COL_TARGET_X, COL_TARGET_Z, SHOT_COL_PLAYER, SHOT_COL_RESULT, SHOT_COL_TEAM}
    missing = sorted(c for c in required_cols if c not in df.columns)
    if missing:
        fig.update_layout(title="Goal Mouth (missing data)")
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text=f"Missing columns: {', '.join(missing)}",
            showarrow=False,
            font=dict(size=11, color="#D7DEE9"),
        )
        return apply_chart_theme(fig, tier="support")

    shots = df.copy()
    if team:
        shots = shots[shots[SHOT_COL_TEAM] == team]
    if player:
        shots = shots[shots[SHOT_COL_PLAYER] == player]
    if on_target_only and COL_ON_TARGET in shots.columns:
        shots = shots[shots[COL_ON_TARGET] == True]

    shots = shots.dropna(subset=[COL_TARGET_X, COL_TARGET_Z])
    if shots.empty:
        fig.update_layout(title="Goal Mouth")
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper", text="No target data", showarrow=False,
            font=dict(size=12, color="#D7DEE9"),
        )
        return apply_chart_theme(fig, tier="support")

    result_symbol = {"Goal": "star", "Shot": "circle", "Save": "x", "Post": "diamond"}
    default_symbol = "circle-open"

    shots = shots.copy()
    if SHOT_COL_FRAME in shots.columns:
        shots["_time"] = pd.to_numeric(shots[SHOT_COL_FRAME], errors="coerce").fillna(0) / float(REPLAY_FPS)
    else:
        shots["_time"] = 0
    shots["_xg"] = pd.to_numeric(shots.get(COL_XG, 0), errors="coerce").fillna(0)
    shots["_xgot"] = pd.to_numeric(shots.get(COL_XGOT, 0), errors="coerce").fillna(0)
    speed_col = "Speed" if "Speed" in shots.columns else None
    shots["_speed"] = pd.to_numeric(shots[speed_col], errors="coerce").fillna(0) if speed_col else 0

    marker_size = (shots["_xgot"].clip(lower=0) * 28 + 10).clip(8, 30) if include_xgot else 12

    for team_name, team_rows in shots.groupby(SHOT_COL_TEAM):
        team_color = TEAM_COLORS.get(team_name, {}).get("primary", "#9aa4b2")
        fig.add_trace(
            go.Scatter(
                x=team_rows[COL_TARGET_X],
                y=team_rows[COL_TARGET_Z],
                mode="markers",
                name=team_name,
                marker=dict(
                    size=marker_size.loc[team_rows.index] if include_xgot else marker_size,
                    color=team_rows["_xgot"] if include_xgot else team_color,
                    colorscale="Turbo",
                    cmin=0,
                    cmax=max(0.01, float(shots["_xgot"].max())),
                    symbol=[result_symbol.get(r, default_symbol) for r in team_rows[SHOT_COL_RESULT]],
                    line=dict(width=1, color="white"),
                    opacity=0.9,
                    showscale=False,
                ),
                customdata=list(
                    zip(
                        team_rows[SHOT_COL_PLAYER],
                        team_rows["_time"],
                        team_rows[SHOT_COL_RESULT],
                        team_rows["_xg"],
                        team_rows["_xgot"],
                        team_rows["_speed"],
                    )
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "t=%{customdata[1]:.1f}s | %{customdata[2]}<br>"
                    "xG %{customdata[3]:.2f} · xGOT %{customdata[4]:.2f}<br>"
                    "Speed %{customdata[5]:.0f} uu/s<extra></extra>"
                ),
            )
        )

    # Goal frame boundaries.
    frame_color = "rgba(255,255,255,0.7)"
    fig.add_shape(type="rect", x0=-GOAL_HALF_W, y0=0, x1=GOAL_HALF_W, y1=GOAL_HEIGHT, line=dict(color=frame_color, width=2))
    fig.add_shape(type="line", x0=0, y0=0, x1=0, y1=GOAL_HEIGHT, line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"))

    fig.update_xaxes(title="TargetX", range=[-GOAL_HALF_W * 1.05, GOAL_HALF_W * 1.05], constrain="domain")
    fig.update_yaxes(title="TargetZ", range=[-20, GOAL_HEIGHT * 1.05], scaleanchor="x", scaleratio=1)
    fig.update_layout(title="Goal Mouth", legend_title_text="Team")
    return apply_chart_theme(fig, tier="support")
