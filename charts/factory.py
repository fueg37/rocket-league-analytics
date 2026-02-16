"""Reusable chart factories for player comparisons."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from constants import TEAM_COLORS
from charts.theme import apply_chart_theme

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
):
    """Render a dumbbell chart for two-value comparisons per entity."""
    comp_df = df[[entity_col, left_col, right_col]].copy()
    comp_df[left_col] = pd.to_numeric(comp_df[left_col], errors="coerce").fillna(0)
    comp_df[right_col] = pd.to_numeric(comp_df[right_col], errors="coerce").fillna(0)
    comp_df["delta"] = comp_df[right_col] - comp_df[left_col]
    comp_df = comp_df.sort_values("delta", ascending=True).reset_index(drop=True)

    left_color = "#8C9AAD"   # muted slate
    right_color = "#00CC96"  # positive endpoint accent
    connector_color = "rgba(165, 171, 184, 0.45)"

    fig = go.Figure()

    for i, row in comp_df.iterrows():
        left_val = float(row[left_col])
        right_val = float(row[right_col])
        delta = right_val - left_val
        delta_sign = "+" if delta > 0 else ""

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
            text=f"{delta_sign}{delta:.2f}",
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
        title=f"{left_label} vs {right_label}",
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
