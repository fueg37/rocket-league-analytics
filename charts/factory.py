"""Reusable chart factories for player comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from constants import FIELD_HALF_X, FIELD_HALF_Y, GOAL_HALF_W, GOAL_HEIGHT, WALL_HEIGHT, REPLAY_FPS, TEAM_COLORS
from charts.theme import apply_chart_theme, semantic_color
from charts.formatters import (
    format_metric_series,
    format_metric_value,
    reliability_badge,
    title_case_label,
)
from analytics.shot_quality import (
    COL_ON_TARGET,
    COL_GOALKEEPER_DIST,
    COL_SHOT_Z,
    COL_TARGET_X,
    COL_TARGET_Z,
    COL_XG,
    COL_XGOT,
    SHOT_COL_FRAME,
    SHOT_COL_PLAYER,
    SHOT_COL_RESULT,
    SHOT_COL_TEAM,
    SHOT_COL_X,
    SHOT_COL_Y,
)

_TEAM_ACCENT = {
    "Blue": TEAM_COLORS["Blue"]["primary"],
    "Orange": TEAM_COLORS["Orange"]["primary"],
}

_STEM_COLOR = {
    "Blue": "rgba(74, 124, 196, 0.35)",
    "Orange": "rgba(196, 138, 74, 0.35)",
}


def kickoff_kpi_indicator(win_rate: float, title: str, tier: str = "detail"):
    """Build a kickoff KPI with progress/bullet encoding and benchmark markers."""
    if win_rate >= 50:
        bar_color = semantic_color("threshold", "positive")
    elif win_rate < 30:
        bar_color = semantic_color("threshold", "negative")
    else:
        bar_color = semantic_color("threshold", "neutral")

    benchmark_traces = [
        (45, "Baseline", "dot", "rgba(255,255,255,0.95)"),
        (50, "Break-even", "dash", "rgba(255,255,255,0.8)"),
        (55, "Elite", "dot", "rgba(255,255,255,0.95)"),
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[float(win_rate)],
            y=[title],
            orientation="h",
            marker=dict(color=bar_color),
            text=[f"{win_rate:.1f}%"],
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate="Kickoff Win Rate: %{x:.1f}%<extra></extra>",
            name="Win Rate",
            showlegend=False,
        )
    )

    for x_pos, label, dash, color in benchmark_traces:
        fig.add_shape(
            type="line",
            x0=x_pos,
            x1=x_pos,
            y0=-0.45,
            y1=0.45,
            line=dict(color=color, width=2, dash=dash),
        )
        fig.add_annotation(
            x=x_pos,
            y=0.62,
            text=label,
            showarrow=False,
            font=dict(size=10, color=color),
        )

    fig.update_xaxes(range=[0, 100], ticksuffix="%", title="Win Rate")
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(height=260, barmode="overlay", margin=dict(l=20, r=20, t=60, b=30), title=title)
    return apply_chart_theme(fig, tier=tier, intent="threshold", variant="neutral")


def spatial_outcome_scatter(df, x_col: str, y_col: str, outcome_col: str, label_col: str | None = None, title: str = "", tier: str = "support", intent: str = "outcome", variant: str = "neutral"):
    """Build a spatial scatter split by outcomes using semantic outcome colors."""
    fig = go.Figure()
    outcome_symbols = {"Win": "circle", "Loss": "x", "Neutral": "diamond"}
    for outcome in ["Win", "Loss", "Neutral"]:
        subset = df[df[outcome_col] == outcome]
        if subset.empty:
            continue
        marker = dict(
            size=12,
            color=semantic_color("outcome", outcome.lower()),
            symbol=outcome_symbols.get(outcome, "circle"),
            opacity=0.85,
            line=dict(width=1, color="white"),
        )
        trace = go.Scatter(
            x=subset[x_col],
            y=subset[y_col],
            mode="markers",
            marker=marker,
            name=outcome,
            hovertemplate=("%{text}<br>Result: " + outcome + "<extra></extra>") if label_col else ("Result: " + outcome + "<extra></extra>"),
        )
        if label_col:
            trace.text = subset[label_col]
        fig.add_trace(trace)

    fig.update_layout(title=title)
    return apply_chart_theme(fig, tier=tier, intent=intent, variant=variant)


def rolling_trend_with_wl_markers(hero_df, hero_display_df, metric: str, hero: str, rolling_window: int, show_wl_markers: bool = True, teammate_df=None, teammate_name: str | None = None):
    """Build rolling trend chart with semantic dual-series and W/L marker colors."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hero_df["GameNum"],
        y=hero_display_df[metric],
        name=hero,
        mode="lines+markers" if show_wl_markers else "lines",
        line=dict(color=semantic_color("dual_series", "primary"), width=2, dash="dot" if show_wl_markers else "solid"),
        marker=dict(size=6, symbol="circle-open", line=dict(width=1.2, color=semantic_color("dual_series", "primary"))),
        opacity=0.4 if show_wl_markers else 1.0,
    ))

    if len(hero_df) >= rolling_window:
        rolling = hero_display_df[metric].rolling(window=rolling_window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=hero_df["GameNum"],
            y=rolling,
            name=f"{hero} ({rolling_window}g avg)",
            line=dict(color=semantic_color("dual_series", "primary"), width=3, dash="solid"),
        ))

    if show_wl_markers and "Won" in hero_df.columns:
        wins_display = hero_display_df[hero_df["Won"] == True]
        losses_display = hero_display_df[hero_df["Won"] == False]
        if not wins_display.empty:
            fig.add_trace(go.Scatter(x=wins_display["GameNum"], y=wins_display[metric], mode="markers", marker=dict(size=8, color=semantic_color("outcome", "win"), symbol="triangle-up", line=dict(width=1, color="white")), name="Win"))
        if not losses_display.empty:
            fig.add_trace(go.Scatter(x=losses_display["GameNum"], y=losses_display[metric], mode="markers", marker=dict(size=9, color=semantic_color("outcome", "loss"), symbol="triangle-down", line=dict(width=1, color="white")), name="Loss"))

    if teammate_df is not None and teammate_name:
        fig.add_trace(go.Scatter(
            x=teammate_df["GameNum"],
            y=teammate_df[metric],
            name=teammate_name,
            mode="lines+markers" if show_wl_markers else "lines",
            line=dict(color=semantic_color("dual_series", "secondary"), width=2, dash="dash"),
            marker=dict(size=6, symbol="square-open", line=dict(width=1.2, color=semantic_color("dual_series", "secondary"))),
            opacity=0.45 if show_wl_markers else 1.0,
        ))
        if len(teammate_df) >= rolling_window:
            mate_rolling = teammate_df[metric].rolling(window=rolling_window, min_periods=1).mean()
            fig.add_trace(go.Scatter(x=teammate_df["GameNum"], y=mate_rolling, name=f"{teammate_name} ({rolling_window}g avg)", line=dict(color=semantic_color("dual_series", "secondary"), width=3, dash="longdash")))

    fig.update_layout(title=f"{metric} over Time", yaxis_title=metric)
    return apply_chart_theme(fig, tier="support", intent="dual_series", variant="primary")


def coach_report_timeline_chart(
    win_prob_df: pd.DataFrame,
    momentum_series: pd.Series,
    coach_report_df: pd.DataFrame,
):
    """Build a tactical timeline combining win probability, momentum, and coach opportunities."""

    def _normalized_uncertainty_series(df: pd.DataFrame, column: str) -> pd.Series:
        """Return a numeric Series aligned to the plotting dataframe index."""
        if column not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype=float)

        values = df[column]
        if np.isscalar(values):
            return pd.Series([values] * len(df), index=df.index, dtype=float)

        series = pd.to_numeric(pd.Series(values), errors="coerce")
        return series.reindex(df.index)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    wp = win_prob_df.copy() if win_prob_df is not None else pd.DataFrame()
    if not wp.empty:
        time_col = "Time" if "Time" in wp.columns else None
        if time_col is not None:
            wp[time_col] = pd.to_numeric(wp[time_col], errors="coerce")
            wp_col = "WinProb" if "WinProb" in wp.columns else ("BlueWinProb" if "BlueWinProb" in wp.columns else None)
            if wp_col is not None:
                wp[wp_col] = pd.to_numeric(wp[wp_col], errors="coerce")
                wp = wp[[time_col, wp_col]].dropna().sort_values(time_col)
                if not wp.empty:
                    wp_values = wp[wp_col].to_numpy(dtype=float)
                    if wp_col == "BlueWinProb" and np.nanmax(wp_values) <= 1.0:
                        wp_values = wp_values * 100.0
                    fig.add_trace(
                        go.Scatter(
                            x=wp[time_col],
                            y=wp_values,
                            mode="lines",
                            line=dict(color=semantic_color("dual_series", "primary"), width=3),
                            name="Win probability",
                            hovertemplate="Time %{x:.1f}s<br>Win probability %{y:.1f}%<extra></extra>",
                        ),
                        secondary_y=False,
                    )

    if momentum_series is not None and not momentum_series.empty:
        mom = pd.to_numeric(momentum_series, errors="coerce").dropna()
        if not mom.empty:
            fig.add_trace(
                go.Scatter(
                    x=mom.index.to_numpy(dtype=float),
                    y=mom.to_numpy(dtype=float),
                    mode="lines",
                    line=dict(color=semantic_color("dual_series", "secondary"), width=2, dash="dash"),
                    name="Momentum",
                    hovertemplate="Time %{x:.1f}s<br>Momentum %{y:.3f}<extra></extra>",
                ),
                secondary_y=True,
            )

    report = coach_report_df.copy() if coach_report_df is not None else pd.DataFrame()
    if not report.empty and "Time" in report.columns:
        report["Time"] = pd.to_numeric(report["Time"], errors="coerce")
        report["MissedSwing"] = pd.to_numeric(report.get("MissedSwing", 0.0), errors="coerce").fillna(0.0)
        report["Confidence"] = pd.to_numeric(report.get("Confidence", 0.0), errors="coerce").fillna(0.0)
        report = report.dropna(subset=["Time"]).sort_values("Time")
        if not report.empty:
            max_missed = max(float(report["MissedSwing"].abs().max()), 1e-6)
            report["marker_size"] = 9 + 18 * (report["MissedSwing"].abs() / max_missed)
            report["action"] = pd.Series(report.get("RecommendedAction", "n/a"), index=report.index).fillna("n/a").astype(str)
            report["role"] = pd.Series(report.get("Role", "n/a"), index=report.index).fillna("n/a").astype(str)
            report["ExpectedSwingMean"] = _normalized_uncertainty_series(report, "ExpectedSwingMean")
            p10 = _normalized_uncertainty_series(report, "ExpectedSwingP10")
            p90 = _normalized_uncertainty_series(report, "ExpectedSwingP90")
            report["ExpectedSwingIntervalWidth"] = _normalized_uncertainty_series(report, "ExpectedSwingIntervalWidth")
            interval_present = pd.notna(p10) & pd.notna(p90)
            interval_labels = np.where(
                interval_present,
                "[" + p10.round(3).astype(str) + ", " + p90.round(3).astype(str) + "]",
                "n/a",
            )
            clip_key = pd.Series(report.get("ClipKey", "n/a"), index=report.index).fillna("n/a").astype(str)

            customdata = np.column_stack(
                [
                    report["action"],
                    report["role"],
                    interval_labels,
                    (report["Confidence"] * 100.0).round(1),
                    clip_key,
                    report["MissedSwing"].round(3),
                ]
            )
            marker_line = np.where(report["MissedSwing"] >= 0, "rgba(125, 238, 160, 0.95)", "rgba(255, 130, 130, 0.95)")
            fig.add_trace(
                go.Scatter(
                    x=report["Time"],
                    y=np.full(len(report), 50.0),
                    mode="markers",
                    name="Missed opportunities",
                    marker=dict(
                        size=report["marker_size"],
                        color=report["Confidence"],
                        colorscale="Turbo",
                        cmin=0.0,
                        cmax=1.0,
                        opacity=0.92,
                        line=dict(color=marker_line, width=2),
                        colorbar=dict(title="Confidence", tickformat=".0%"),
                        symbol="diamond",
                    ),
                    customdata=customdata,
                    hovertemplate=(
                        "Time %{x:.1f}s"
                        "<br>Action %{customdata[0]}"
                        "<br>Role %{customdata[1]}"
                        "<br>Expected swing interval %{customdata[2]}"
                        "<br>Confidence %{customdata[3]:.1f}%"
                        "<br>Missed swing %{customdata[5]}"
                        "<br>Clip %{customdata[4]}"
                        "<extra></extra>"
                    ),
                ),
                secondary_y=False,
            )

    fig.update_layout(
        title="Coach timeline: win probability, momentum, and missed opportunities",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(title_text="Match time (s)")
    fig.update_yaxes(title_text="Win probability (%)", range=[0, 100], secondary_y=False)
    fig.update_yaxes(title_text="Momentum", zeroline=True, secondary_y=True)
    return apply_chart_theme(fig, tier="hero", intent="dual_series", variant="primary")


def session_composite_chart(summary_df):
    """Build synchronized session subplots for win-rate and rating with sample-size context."""
    ordered = summary_df.sort_values(["Session"], ascending=[True], kind="mergesort").reset_index(drop=True)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.55, 0.45],
        subplot_titles=("Win Rate %", "Avg Rating"),
    )

    win_low = pd.to_numeric(ordered.get("Win Rate % CI Low", ordered["Win Rate %"]), errors="coerce").fillna(ordered["Win Rate %"])
    win_high = pd.to_numeric(ordered.get("Win Rate % CI High", ordered["Win Rate %"]), errors="coerce").fillna(ordered["Win Rate %"])
    rel_col = ordered.get("Win Rate % Reliability", pd.Series(["low"] * len(ordered)))

    hover_base = (
        "Session %{x}<br>"
        "Games per Session: %{customdata[0]}<br>"
        "Reliability: %{customdata[1]}<br>"
    )

    fig.add_trace(
        go.Bar(
            x=ordered["Session"],
            y=ordered["Win Rate %"],
            name="Win Rate %",
            marker_color=semantic_color("outcome", "win"),
            text=[reliability_badge(rel_col.iloc[i], int(ordered["Games per Session"].iloc[i])) for i in range(len(ordered))],
            textposition="outside",
            cliponaxis=False,
            customdata=np.column_stack([ordered["Games per Session"], rel_col, win_low, win_high]),
            hovertemplate=hover_base + "Win Rate %: %{y:.1f}%<br>95% CI: [%{customdata[2]:.1f}%, %{customdata[3]:.1f}%]<extra></extra>",
            error_y=dict(
                type="data",
                symmetric=False,
                array=(win_high - ordered["Win Rate %"]).clip(lower=0),
                arrayminus=(ordered["Win Rate %"] - win_low).clip(lower=0),
                color="rgba(255,255,255,0.6)",
                thickness=1,
                width=3,
            ),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ordered["Session"],
            y=ordered["Avg Rating"],
            name="Avg Rating",
            mode="lines+markers",
            line=dict(color=semantic_color("dual_series", "secondary"), width=3),
            marker=dict(size=8),
            customdata=np.column_stack([ordered["Games per Session"], rel_col]),
            hovertemplate=hover_base + "Avg Rating: %{y:.2f}<br>Use CI-aware summaries where available<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title="Session Performance Overview",
        legend=dict(x=0.01, y=1.02, orientation="h"),
        bargap=0.25,
    )
    fig.update_xaxes(title="Session", type="category", categoryorder="array", categoryarray=ordered["Session"].tolist(), row=2, col=1)
    fig.update_yaxes(title="Win Rate %", range=[0, 100], row=1, col=1)
    fig.update_yaxes(title="Avg Rating", rangemode="tozero", row=2, col=1)

    return apply_chart_theme(fig, tier="hero", intent="dual_series", variant="primary")


def player_rank_lollipop(df, metric_col, name_col="Name", team_col="Team"):
    """Render a player ranking lollipop chart for one metric.

    Sorts descending by metric and uses muted stems with team-color marker accents.
    """
    cols = [name_col, team_col, metric_col]
    rank_df = df[cols].copy()
    rank_df[metric_col] = pd.to_numeric(rank_df[metric_col], errors="coerce").fillna(0)
    rank_df = rank_df.sort_values([metric_col, name_col], ascending=[False, True], kind='mergesort').reset_index(drop=True)

    fig = go.Figure()
    labels = format_metric_series(rank_df[metric_col], metric_col, include_unit=False)

    for i, row in rank_df.iterrows():
        team = row[team_col]
        stem_color = _STEM_COLOR.get(team, "rgba(165, 171, 184, 0.35)")
        accent = _TEAM_ACCENT.get(team, semantic_color("threshold", "neutral"))
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
                hovertemplate=f"Player: {row[name_col]}<br>Team: {team}<br>Metric: {title_case_label(metric_col)}: {format_metric_value(row[metric_col], metric_col)}<extra></extra>",
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
    fig.update_xaxes(title=title_case_label(metric_col), rangemode="tozero")
    fig.update_layout(title=title_case_label(metric_col))

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

    left_color = semantic_color("dual_series", "comparison_left")
    right_color = semantic_color("dual_series", "comparison_right")
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
                text=[format_metric_value(left_val, left_label, include_unit=False)],
                textposition="middle left",
                textfont=dict(size=10, color=left_color),
                hovertemplate=f"Player: {row[entity_col]}<br>Metric: {title_case_label(left_label)}: {format_metric_value(left_val, left_label)}<extra></extra>",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[right_val],
                y=[i],
                mode="markers+text",
                marker=dict(size=11, color="white", line=dict(color=right_color, width=2.5)),
                text=[format_metric_value(right_val, right_label, include_unit=False)],
                textposition="middle right",
                textfont=dict(size=10, color=right_color),
                hovertemplate=f"Player: {row[entity_col]}<br>Metric: {title_case_label(right_label)}: {format_metric_value(right_val, right_label)}<extra></extra>",
                showlegend=False,
            )
        )

        mid_x = (left_val + right_val) / 2
        fig.add_annotation(
            x=mid_x,
            y=i,
            text=delta_text,
            showarrow=False,
            font=dict(size=10, color=semantic_color("threshold", "neutral")),
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
    fig.update_xaxes(title=f"{title_case_label(left_label)} vs {title_case_label(right_label)}", zeroline=False)
    fig.update_layout(
        margin=dict(l=10, r=10, t=45, b=10),
        title=f"{title_case_label(left_label)} vs {title_case_label(right_label)} (Δ = {title_case_label(right_label)} − {title_case_label(left_label)})",
    )

    # Legend-style endpoint labels in-chart for quick semantic mapping.
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.0,
        y=1.08,
        text=f"<span style='color:{left_color}'>●</span> {title_case_label(left_label)}",
        showarrow=False,
        xanchor="left",
        font=dict(size=11),
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.18,
        y=1.08,
        text=f"<span style='color:{right_color}'>●</span> {title_case_label(right_label)}",
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
            font=dict(size=11, color=semantic_color("threshold", "neutral")),
        )
        return apply_chart_theme(fig, tier="support")

    shots = df.copy()
    if team:
        shots = shots[shots[SHOT_COL_TEAM] == team]
    if player:
        shots = shots[shots[SHOT_COL_PLAYER] == player]
    if on_target_only and COL_ON_TARGET in shots.columns:
        shots = shots[(shots[COL_ON_TARGET] == True) | (shots[SHOT_COL_RESULT] == "Goal")]

    # Preserve goals even when target reconstruction failed by backfilling
    # a neutral in-frame estimate so every goal remains inspectable.
    goal_mask = shots[SHOT_COL_RESULT] == "Goal"
    missing_x = goal_mask & shots[COL_TARGET_X].isna()
    missing_z = goal_mask & shots[COL_TARGET_Z].isna()
    if missing_x.any():
        shots.loc[missing_x, COL_TARGET_X] = 0.0
    if missing_z.any():
        if COL_SHOT_Z in shots.columns:
            shots.loc[missing_z, COL_TARGET_Z] = pd.to_numeric(shots.loc[missing_z, COL_SHOT_Z], errors="coerce").fillna(GOAL_HEIGHT * 0.35)
        else:
            shots.loc[missing_z, COL_TARGET_Z] = GOAL_HEIGHT * 0.35

    shots[COL_TARGET_X] = pd.to_numeric(shots[COL_TARGET_X], errors="coerce")
    shots[COL_TARGET_Z] = pd.to_numeric(shots[COL_TARGET_Z], errors="coerce").clip(lower=0, upper=GOAL_HEIGHT)
    shots = shots.dropna(subset=[COL_TARGET_X, COL_TARGET_Z])
    if shots.empty:
        fig.update_layout(title="Goal Mouth")
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper", text="No target data", showarrow=False,
            font=dict(size=12, color=semantic_color("threshold", "neutral")),
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
        team_color = TEAM_COLORS.get(team_name, {}).get("primary", semantic_color("threshold", "neutral"))
        fig.add_trace(
            go.Scatter(
                x=team_rows[COL_TARGET_X],
                y=team_rows[COL_TARGET_Z],
                mode="markers",
                name=team_name,
                marker=dict(
                    size=marker_size.loc[team_rows.index] if include_xgot else marker_size,
                    color=team_color,
                    symbol=[result_symbol.get(r, default_symbol) for r in team_rows[SHOT_COL_RESULT]],
                    line=dict(width=1, color="white"),
                    opacity=0.92,
                ),
                customdata=list(
                    zip(
                        team_rows[SHOT_COL_PLAYER],
                        [team_name] * len(team_rows),
                        team_rows["_time"].map(lambda v: format_metric_value(v, "Time")),
                        team_rows["_xg"].map(lambda v: format_metric_value(v, "xG")),
                        team_rows["_xgot"].map(lambda v: format_metric_value(v, "xGOT")),
                        team_rows["_speed"],
                        team_rows["_speed"].map(lambda v: format_metric_value(v, "Speed")),
                    )
                ),
                hovertemplate=(
                    "Player: %{customdata[0]}<br>"
                    "Team: %{customdata[1]}<br>"
                    "Time: %{customdata[2]}<br>"
                    "Metric: xG: %{customdata[3]}<extra></extra>"
                ),
            )
        )

    # Goal frame boundaries.
    frame_color = "rgba(255,255,255,0.7)"
    fig.add_shape(type="rect", x0=-GOAL_HALF_W, y0=0, x1=GOAL_HALF_W, y1=GOAL_HEIGHT, line=dict(color=frame_color, width=2))
    fig.add_shape(type="line", x0=0, y0=0, x1=0, y1=GOAL_HEIGHT, line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"))

    fig.update_xaxes(title=title_case_label("target x"), range=[-GOAL_HALF_W * 1.05, GOAL_HALF_W * 1.05], constrain="domain")
    fig.update_yaxes(title=title_case_label("target z"), range=[-20, GOAL_HEIGHT * 1.05], scaleanchor="x", scaleratio=1)
    fig.update_layout(title="Goal Mouth", legend_title_text="Team")
    return apply_chart_theme(fig, tier="support")


def arena_3d_shot_chart(
    df: pd.DataFrame,
    *,
    team: str | None = None,
    player: str | None = None,
    on_target_only: bool = False,
    color_metric: str = "xgot",
    show_trajectories: bool = False,
    enable_3d: bool = True,
    save_events_df: pd.DataFrame | None = None,
):
    """Render Arena 3D shot geometry with metric-aware encoding and 2D fallback."""
    required_cols = {SHOT_COL_X, SHOT_COL_Y, SHOT_COL_TEAM, SHOT_COL_PLAYER, SHOT_COL_RESULT}
    shots = df.copy() if df is not None else pd.DataFrame()
    missing = sorted(c for c in required_cols if c not in shots.columns)
    if shots.empty or missing:
        fig = go.Figure()
        fig.update_layout(title="Arena 3D")
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", text="No shot geometry available" if shots.empty else f"Missing columns: {', '.join(missing)}", showarrow=False)
        return apply_chart_theme(fig, tier="support")

    if save_events_df is not None and not save_events_df.empty and "SaveImpact" not in shots.columns and {"Frame", "Shooter", "SaveImpact", "ExpectedSaveProb", "SaveDifficultyIndex"}.issubset(save_events_df.columns):
        save_view = save_events_df[["Frame", "Shooter", "SaveImpact", "ExpectedSaveProb", "SaveDifficultyIndex"]].copy()
        save_view = save_view.rename(columns={"Shooter": SHOT_COL_PLAYER})
        shots = shots.merge(save_view, on=["Frame", SHOT_COL_PLAYER], how="left")

    if team:
        shots = shots[shots[SHOT_COL_TEAM] == team]
    if player:
        shots = shots[shots[SHOT_COL_PLAYER] == player]
    if on_target_only and COL_ON_TARGET in shots.columns:
        shots = shots[(shots[COL_ON_TARGET] == True) | (shots[SHOT_COL_RESULT] == "Goal")]

    shots[SHOT_COL_X] = pd.to_numeric(shots[SHOT_COL_X], errors="coerce")
    shots[SHOT_COL_Y] = pd.to_numeric(shots[SHOT_COL_Y], errors="coerce")
    shots["_z"] = pd.to_numeric(shots.get(COL_SHOT_Z, 0), errors="coerce").fillna(0).clip(lower=0, upper=WALL_HEIGHT)
    shots["_target_x"] = pd.to_numeric(shots.get(COL_TARGET_X, np.nan), errors="coerce")
    shots["_target_z"] = pd.to_numeric(shots.get(COL_TARGET_Z, np.nan), errors="coerce").clip(lower=0, upper=GOAL_HEIGHT)
    shots["_goalkeeper_dist"] = pd.to_numeric(shots.get(COL_GOALKEEPER_DIST, np.nan), errors="coerce")
    shots = shots.dropna(subset=[SHOT_COL_X, SHOT_COL_Y, "_z"])
    if shots.empty:
        return goal_mouth_scatter(df, team=team, player=player, include_xgot=True, on_target_only=on_target_only)

    metric_map = {
        "xg": (COL_XG, "xG", "Viridis"),
        "xgot": (COL_XGOT, "xGOT", "Plasma"),
        "save_impact": ("SaveImpact", "Save Impact", "RdBu"),
        "expected_save_prob": ("ExpectedSaveProb", "Expected Save Prob", "Blues"),
        "save_difficulty": ("SaveDifficultyIndex", "Save Difficulty", "Magma"),
    }
    metric_col, metric_label, metric_scale = metric_map.get(color_metric, metric_map["xgot"])
    shots["_metric"] = pd.to_numeric(shots.get(metric_col, shots.get(COL_XGOT, 0.0)), errors="coerce").fillna(0.0)

    if not enable_3d:
        fig2d = go.Figure()
        fig2d.add_shape(type="rect", x0=-GOAL_HALF_W, y0=0, x1=GOAL_HALF_W, y1=GOAL_HEIGHT, line=dict(color="rgba(255,255,255,0.7)", width=2))
        fig2d.add_shape(type="line", x0=0, y0=0, x1=0, y1=GOAL_HEIGHT, line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"))
        fig2d.add_trace(go.Scatter(
            x=shots["_target_x"].fillna(0),
            y=shots["_target_z"].fillna(shots["_z"].clip(upper=GOAL_HEIGHT)),
            mode="markers",
            marker=dict(size=10, color=shots["_metric"], colorscale=metric_scale, showscale=True, colorbar=dict(title=metric_label), line=dict(width=1, color="white"), opacity=0.9),
            customdata=np.stack([shots[SHOT_COL_PLAYER], shots[SHOT_COL_TEAM], shots[SHOT_COL_RESULT], shots["_metric"]], axis=-1),
            hovertemplate="Player: %{customdata[0]}<br>Team: %{customdata[1]}<br>Result: %{customdata[2]}<br>Metric: %{customdata[3]:.2f}<extra></extra>",
            name="Events",
        ))
        fig2d.update_xaxes(title=title_case_label("target x"), range=[-GOAL_HALF_W * 1.05, GOAL_HALF_W * 1.05], constrain="domain")
        fig2d.update_yaxes(title=title_case_label("target z"), range=[-20, GOAL_HEIGHT * 1.05], scaleanchor="x", scaleratio=1)
        fig2d.update_layout(title=f"Arena 2D Fallback ({metric_label})")
        return apply_chart_theme(fig2d, tier="support")

    fig = go.Figure()

    # Goal mouth planes (both ends).
    for goal_y in (FIELD_HALF_Y, -FIELD_HALF_Y):
        fig.add_trace(
            go.Mesh3d(
                x=[-GOAL_HALF_W, GOAL_HALF_W, GOAL_HALF_W, -GOAL_HALF_W],
                y=[goal_y, goal_y, goal_y, goal_y],
                z=[0, 0, GOAL_HEIGHT, GOAL_HEIGHT],
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color="rgba(255,255,255,0.14)",
                hoverinfo="skip",
                name=f"Goal plane y={int(goal_y)}",
                showscale=False,
            )
        )
        env_depth = 700
        y0, y1 = (goal_y - env_depth, goal_y) if goal_y > 0 else (goal_y, goal_y + env_depth)
        fig.add_trace(
            go.Mesh3d(
                x=[-GOAL_HALF_W, GOAL_HALF_W, GOAL_HALF_W, -GOAL_HALF_W, -GOAL_HALF_W, GOAL_HALF_W, GOAL_HALF_W, -GOAL_HALF_W],
                y=[y0, y0, y1, y1, y0, y0, y1, y1],
                z=[0, 0, 0, 0, WALL_HEIGHT * 0.55, WALL_HEIGHT * 0.55, WALL_HEIGHT * 0.55, WALL_HEIGHT * 0.55],
                i=[0, 0, 4, 4, 0, 1, 2, 3, 0, 1, 2, 3],
                j=[1, 2, 5, 6, 4, 5, 6, 7, 1, 2, 3, 0],
                k=[2, 3, 6, 7, 5, 6, 7, 4, 4, 5, 6, 7],
                color="rgba(142, 196, 255, 0.10)",
                opacity=0.12,
                hoverinfo="skip",
                name="Keeper envelope",
                showscale=False,
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=shots[SHOT_COL_X],
            y=shots[SHOT_COL_Y],
            z=shots["_z"],
            mode="markers",
            name="Ball events",
            marker=dict(size=5, color=shots["_metric"], colorscale=metric_scale, cmin=float(shots["_metric"].min()), cmax=float(shots["_metric"].max() + 1e-9), colorbar=dict(title=metric_label), line=dict(width=0.5, color="white"), opacity=0.9),
            customdata=np.stack([
                shots[SHOT_COL_PLAYER],
                shots[SHOT_COL_TEAM],
                shots[SHOT_COL_RESULT],
                shots["_metric"],
                shots["_goalkeeper_dist"],
            ], axis=-1),
            hovertemplate="Player: %{customdata[0]}<br>Team: %{customdata[1]}<br>Result: %{customdata[2]}<br>Metric: %{customdata[3]:.2f}<br>Keeper dist: %{customdata[4]:.0f}<extra></extra>",
        )
    )

    if show_trajectories:
        traj_x, traj_y, traj_z = [], [], []
        for _, row in shots.dropna(subset=["_target_x", "_target_z"]).iterrows():
            target_y = FIELD_HALF_Y if float(row[SHOT_COL_Y]) <= 0 else -FIELD_HALF_Y
            traj_x.extend([float(row[SHOT_COL_X]), float(row["_target_x"]), None])
            traj_y.extend([float(row[SHOT_COL_Y]), target_y, None])
            traj_z.extend([float(row["_z"]), float(row["_target_z"]), None])
        if traj_x:
            fig.add_trace(
                go.Scatter3d(
                    x=traj_x,
                    y=traj_y,
                    z=traj_z,
                    mode="lines",
                    name="Projected trajectory",
                    line=dict(color="rgba(255,255,255,0.35)", width=2),
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        title=f"Arena 3D ({metric_label})",
        scene=dict(
            xaxis=dict(title="X", range=[-FIELD_HALF_X, FIELD_HALF_X]),
            yaxis=dict(title="Y", range=[-FIELD_HALF_Y - 200, FIELD_HALF_Y + 200]),
            zaxis=dict(title="Z", range=[0, WALL_HEIGHT]),
            aspectmode="manual",
            aspectratio=dict(x=1.25, y=1.5, z=0.55),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return apply_chart_theme(fig, tier="support")


def chemistry_network_chart(pair_df: pd.DataFrame, *, min_samples: int = 3, title: str = "Partnership Network"):
    """Render player chemistry graph with contract-aware edge and node semantics."""
    fig = go.Figure()
    if pair_df is None or pair_df.empty:
        fig.update_layout(title=title)
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", text="No chemistry data", showarrow=False)
        return apply_chart_theme(fig, tier="support")

    data = pair_df[pair_df["Samples"] >= int(min_samples)].copy()
    if data.empty:
        fig.update_layout(title=title)
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", text="Insufficient chemistry samples", showarrow=False)
        return apply_chart_theme(fig, tier="support")

    players = sorted(set(data["Player1"]).union(set(data["Player2"])))
    if not players:
        return apply_chart_theme(fig, tier="support")

    theta = np.linspace(0, 2 * np.pi, len(players), endpoint=False)
    radius = 1.0
    pos = {p: (radius * np.cos(t), radius * np.sin(t)) for p, t in zip(players, theta)}

    index_col = "Partnership Index" if "Partnership Index" in data.columns else ("ChemistryScore_Shrunk" if "ChemistryScore_Shrunk" in data.columns else "ChemistryScore")
    impact_col = "expected_xgd_lift_per_match" if "expected_xgd_lift_per_match" in data.columns else "ExpectedValueGain_Shrunk"
    confidence_col = "confidence_level" if "confidence_level" in data.columns else "Reliability"
    sample_col = "sample_count" if "sample_count" in data.columns else "Samples"
    driver_col = "primary_driver_label" if "primary_driver_label" in data.columns else None

    data[index_col] = pd.to_numeric(data.get(index_col, 0), errors="coerce").fillna(0.0)
    data[impact_col] = pd.to_numeric(data.get(impact_col, 0), errors="coerce").fillna(0.0)
    data[sample_col] = pd.to_numeric(data.get(sample_col, data.get("Samples", 0)), errors="coerce").fillna(0).astype(int)

    min_index, max_index = float(data[index_col].min()), float(data[index_col].max())
    index_span = max(1e-9, max_index - min_index)
    max_abs_impact = max(1e-9, float(data[impact_col].abs().max()))

    confidence_weight = {"high": 1.0, "medium": 0.75, "low": 0.5}

    def _impact_rgba(impact: float) -> str:
        intensity = min(1.0, abs(impact) / max_abs_impact)
        alpha = 0.18 + (0.72 * intensity)
        # Positive impact: cool cyan, negative impact: warm red.
        return f"rgba(99, 210, 255, {alpha:.2f})" if impact >= 0 else f"rgba(242, 99, 143, {alpha:.2f})"

    weighted_degree = {p: 0.0 for p in players}
    for _, row in data.iterrows():
        x0, y0 = pos[row["Player1"]]
        x1, y1 = pos[row["Player2"]]
        partnership_index = float(row[index_col])
        impact = float(row[impact_col])
        conf = str(row.get(confidence_col, "Low")).strip().lower()
        samples = int(row.get(sample_col, row.get("Samples", 0)))
        primary_driver = str(row.get(driver_col, "Balanced chemistry")).strip() if driver_col else "Balanced chemistry"
        norm = (partnership_index - min_index) / index_span
        width = 1.0 + 6.0 * norm
        color = _impact_rgba(impact)
        conf_weight = confidence_weight.get(conf, 0.5)
        weighted_value = partnership_index * conf_weight * max(1.0, np.sqrt(max(samples, 1) / max(min_samples, 1)))
        weighted_degree[row["Player1"]] += weighted_value
        weighted_degree[row["Player2"]] += weighted_value

        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines", showlegend=False,
            line=dict(width=width, color=color),
            customdata=[[partnership_index, impact, str(row.get(confidence_col, "Low")).title(), samples, primary_driver]],
            hovertemplate=(
                f"{row['Player1']} ↔ {row['Player2']}<br>"
                "Partnership Index: %{customdata[0]:.1f}<br>"
                "Projected xGD Impact: %{customdata[1]:+.3f}<br>"
                "Confidence: %{customdata[2]} (%{customdata[3]} samples)<br>"
                "Primary Driver: %{customdata[4]}<extra></extra>"
            ),
        ))

    min_degree = min(weighted_degree.values())
    max_degree = max(weighted_degree.values())
    degree_span = max(1e-9, max_degree - min_degree)
    node_size = [18 + 9 * (weighted_degree[p] - min_degree) / degree_span for p in players]
    fig.add_trace(go.Scatter(
        x=[pos[p][0] for p in players],
        y=[pos[p][1] for p in players],
        text=players,
        mode="markers+text",
        textposition="top center",
        marker=dict(size=node_size, color=semantic_color("dual_series", "primary"), line=dict(width=1.5, color="white")),
        hovertemplate="Player: %{text}<extra></extra>",
        showlegend=False,
    ))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(title=title)
    return apply_chart_theme(fig, tier="support", intent="dual_series")


def chemistry_ranking_table(pair_df: pd.DataFrame, *, top_n: int = 20) -> pd.DataFrame:
    """Season-level pair ranking table for UI display."""
    if pair_df is None or pair_df.empty:
        return pd.DataFrame(columns=[
            "Rank",
            "Team",
            "Partnership",
            "Partnership Index",
            "Impact (xGD/match)",
            "Win Lift",
            "Confidence Interval",
            "Samples",
            "Confidence",
        ])
    out = pair_df.copy()
    out["Partnership"] = out["Player1"].astype(str) + " + " + out["Player2"].astype(str)
    index_col = "Partnership Index" if "Partnership Index" in out.columns else ("ChemistryScore_Shrunk" if "ChemistryScore_Shrunk" in out.columns else "ChemistryScore")
    impact_col = "expected_xgd_lift_per_match" if "expected_xgd_lift_per_match" in out.columns else "ExpectedValueGain_Shrunk"
    win_lift_col = "win_rate_lift_points" if "win_rate_lift_points" in out.columns else None
    conf_col = "confidence_level" if "confidence_level" in out.columns else "Reliability"

    out = out.sort_values(index_col, ascending=False).head(int(top_n)).reset_index(drop=True)
    out["Rank"] = np.arange(1, len(out) + 1)
    out["Partnership Index"] = out[index_col].map(lambda v: f"{float(v):.1f}")
    out["Impact (xGD/match)"] = out[impact_col].map(lambda v: f"{float(v):+.3f}")
    out["Win Lift"] = out[win_lift_col].map(lambda v: f"{float(v):+.2f} pts") if win_lift_col else "N/A"
    out["Confidence Interval"] = out.apply(lambda r: f"[{float(r.get('ci_low', r.get('CI_Low', 0))):.1f}, {float(r.get('ci_high', r.get('CI_High', 0))):.1f}]", axis=1)
    out["Confidence"] = out[conf_col].astype(str).str.title()
    return out[[
        "Rank",
        "Team",
        "Partnership",
        "Partnership Index",
        "Impact (xGD/match)",
        "Win Lift",
        "Confidence Interval",
        "Samples",
        "Confidence",
    ]]


def value_timeline_chart(actions_df: pd.DataFrame, title: str = "Transition Value Timeline"):
    """Render cumulative and per-action transition value over match time."""
    if actions_df is None or actions_df.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
        return apply_chart_theme(fig, tier="support", intent="dual_series", variant="primary")

    df = actions_df.copy().sort_values("Time", kind="mergesort")
    df["CumulativeVAEP"] = pd.to_numeric(df.get("VAEP", 0.0), errors="coerce").fillna(0.0).cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Time"],
        y=df.get("VAEP", 0.0),
        mode="markers",
        name="Action VAEP",
        marker=dict(size=7, color=semantic_color("outcome", "neutral"), opacity=0.75),
        hovertemplate="Time: %{x:.1f}s<br>Action Value: %{y:+.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["Time"],
        y=df["CumulativeVAEP"],
        mode="lines",
        name="Cumulative Value",
        line=dict(width=3, color=semantic_color("dual_series", "primary")),
        hovertemplate="Time: %{x:.1f}s<br>Cumulative: %{y:+.3f}<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Goal Differential Value")
    return apply_chart_theme(fig, tier="support", intent="dual_series", variant="primary")


def action_type_value_decomposition_chart(actions_df: pd.DataFrame, title: str = "Value by Action Type"):
    """Render stacked decomposition of positive/negative value by action type."""
    if actions_df is None or actions_df.empty:
        return apply_chart_theme(go.Figure(), tier="support", intent="outcome", variant="neutral")

    df = actions_df.copy()
    df["EventType"] = df.get("EventType", "touch")
    df["VAEP"] = pd.to_numeric(df.get("VAEP", 0.0), errors="coerce").fillna(0.0)
    grouped = df.groupby("EventType", as_index=False).agg(
        PositiveValue=("VAEP", lambda s: float(s[s > 0].sum())),
        NegativeValue=("VAEP", lambda s: float(s[s < 0].sum())),
    )
    grouped = grouped.sort_values("PositiveValue", ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grouped["EventType"], y=grouped["PositiveValue"], name="Positive",
        marker_color=semantic_color("outcome", "win"),
        hovertemplate="Action: %{x}<br>Positive Value: %{y:+.3f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=grouped["EventType"], y=grouped["NegativeValue"], name="Negative",
        marker_color=semantic_color("outcome", "loss"),
        hovertemplate="Action: %{x}<br>Negative Value: %{y:+.3f}<extra></extra>",
    ))
    fig.update_layout(title=title, barmode="relative", xaxis_title="Action Type", yaxis_title="Value")
    return apply_chart_theme(fig, tier="support", intent="outcome", variant="neutral")


def teammate_synergy_matrix(shared_actions_df: pd.DataFrame, title: str = "Teammate Synergy Matrix"):
    """Render pairwise synergy from shared transition gains."""
    base_cols = ["Player1", "Player2", "SharedTransitionGain"]
    if shared_actions_df is None or shared_actions_df.empty:
        return apply_chart_theme(go.Figure(), tier="support", intent="dual_series", variant="primary")

    df = shared_actions_df.copy()
    if "SharedTransitionGain" not in df.columns and "VAEP" in df.columns:
        df["SharedTransitionGain"] = pd.to_numeric(df["VAEP"], errors="coerce").fillna(0.0)
    for col in base_cols:
        if col not in df.columns:
            if col in ("Player1", "Player2") and "Player" in df.columns:
                df[col] = df["Player"].astype(str)
            else:
                df[col] = 0.0

    pairs = (
        df.groupby(["Player1", "Player2"], as_index=False)["SharedTransitionGain"].sum()
    )
    players = sorted(set(pairs["Player1"]).union(set(pairs["Player2"])))
    matrix = pd.DataFrame(0.0, index=players, columns=players)
    for row in pairs.itertuples(index=False):
        matrix.loc[row.Player1, row.Player2] += float(row.SharedTransitionGain)
        matrix.loc[row.Player2, row.Player1] += float(row.SharedTransitionGain)

    fig = go.Figure(go.Heatmap(
        x=players,
        y=players,
        z=matrix.values,
        colorscale="RdBu",
        zmid=0.0,
        hovertemplate="%{y} + %{x}<br>Shared Gain: %{z:+.3f}<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis_title="Teammate", yaxis_title="Teammate")
    return apply_chart_theme(fig, tier="support", intent="dual_series", variant="primary")
