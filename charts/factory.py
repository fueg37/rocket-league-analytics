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


def _format_seconds(seconds: float) -> str:
    value = int(round(float(seconds)))
    minutes, secs = divmod(max(value, 0), 60)
    return f"{minutes}:{secs:02d}"


def _time_tick_values(max_time: float, step: int = 60) -> list[int]:
    max_floor = int(max(0, max_time))
    ticks = list(range(0, max_floor + 1, step))
    if not ticks or ticks[-1] != max_floor:
        ticks.append(max_floor)
    return sorted(set(ticks))


def time_series_chart(
    df,
    x_col,
    y_cols,
    labels,
    baseline=None,
    endpoint_labels=True,
    *,
    title=None,
    x_title="Time",
    y_title="Value",
    tier="detail",
    y_range=None,
    series_styles=None,
    hover_precision=2,
    grid_step=60,
    legend=True,
    time_axis=True,
):
    """Build a canonical timeline chart with consistent semantics.

    Args:
        labels: Either ordered list aligned to y_cols or dict of {column: label}.
        baseline: Optional number or iterable of numbers for reference lines.
        series_styles: Optional per-series style overrides keyed by y_col.
    """
    plot_df = df.copy() if df is not None else pd.DataFrame()
    series_styles = series_styles or {}

    fig = go.Figure()
    if plot_df.empty or x_col not in plot_df.columns:
        fig.update_layout(title=title)
        fig.update_xaxes(title=x_title)
        fig.update_yaxes(title=y_title)
        return apply_chart_theme(fig, tier=tier)

    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_col]).sort_values(x_col).reset_index(drop=True)

    label_map = (
        {col: labels.get(col, col) for col in y_cols}
        if isinstance(labels, dict)
        else {col: labels[i] if i < len(labels) else col for i, col in enumerate(y_cols)}
    )

    line_width = 2.6
    x_vals = plot_df[x_col]
    if time_axis:
        x_labels = [_format_seconds(v) for v in x_vals]
    else:
        x_labels = [f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}" for v in x_vals]

    for idx, y_col in enumerate(y_cols):
        if y_col not in plot_df.columns:
            continue

        style = series_styles.get(y_col, {})
        y_vals = pd.to_numeric(plot_df[y_col], errors="coerce")
        mode = style.get("mode", "lines")
        if "markers" not in mode and len(plot_df) <= 18:
            mode = "lines+markers"

        trace_name = label_map.get(y_col, y_col)
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode=mode,
                name=trace_name,
                line=dict(
                    color=style.get("color"),
                    width=style.get("width", line_width),
                    dash=style.get("dash", "solid"),
                    shape=style.get("shape", "linear"),
                ),
                marker=style.get("marker", dict(size=6)),
                fill=style.get("fill"),
                fillcolor=style.get("fillcolor"),
                opacity=style.get("opacity", 1),
                customdata=x_labels,
                hovertemplate=f"{x_title}: %{{customdata}}<br>{trace_name}: %{{y:.{hover_precision}f}}<extra></extra>",
                showlegend=style.get("showlegend", True),
            )
        )

        if endpoint_labels and not y_vals.dropna().empty:
            last_idx = y_vals.last_valid_index()
            if last_idx is not None:
                x_last = x_vals.iloc[last_idx]
                y_last = y_vals.iloc[last_idx]
                fig.add_annotation(
                    x=x_last,
                    y=y_last,
                    text=f"{trace_name} {y_last:.{hover_precision}f}",
                    showarrow=False,
                    xanchor="left",
                    xshift=8,
                    font=dict(size=11, color=style.get("color", "#d7dee9")),
                )

    if baseline is not None:
        baseline_values = baseline if isinstance(baseline, (list, tuple, set)) else [baseline]
        for b in baseline_values:
            fig.add_hline(
                y=float(b),
                line=dict(color="rgba(167,176,196,0.55)", width=1.2, dash="dot"),
            )

    fig.update_layout(title=title, showlegend=legend)
    if time_axis:
        max_time = float(plot_df[x_col].max()) if not plot_df.empty else 0
        ticks = _time_tick_values(max_time, step=grid_step)
        fig.update_xaxes(
            title=x_title,
            tickmode="array",
            tickvals=ticks,
            ticktext=[_format_seconds(v) for v in ticks],
            showgrid=True,
        )
    else:
        fig.update_xaxes(title=x_title, showgrid=True)
    fig.update_yaxes(title=y_title, range=y_range, showgrid=True)

    return apply_chart_theme(fig, tier=tier)


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
