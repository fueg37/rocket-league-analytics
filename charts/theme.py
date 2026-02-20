"""Centralized Plotly chart theme helpers."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from .tokens import (
    BACKGROUNDS,
    DUAL_SERIES_DEFAULTS,
    EMPHASIS_ACCENT,
    GLOW,
    GRID_OPACITY,
    HEATMAP_COLORSCALES,
    OUTCOME_COLORS,
    ROLE_ZONE_PALETTES,
    SPACING,
    TEAM_ACCENTS,
    TEXT,
    THRESHOLD_ACCENTS,
    TYPOGRAPHY,
)




def semantic_color(intent: str, variant: str = "default") -> str:
    """Resolve semantic chart colors by intent/variant names."""
    key = (intent or "").lower()
    var = (variant or "default").lower()

    if key == "outcome":
        return OUTCOME_COLORS.get(var, OUTCOME_COLORS["neutral"])
    if key == "threshold":
        return THRESHOLD_ACCENTS.get(var, THRESHOLD_ACCENTS["neutral"])
    if key == "dual_series":
        return DUAL_SERIES_DEFAULTS.get(var, TEXT.primary)
    if key == "role_zone":
        if ":" in var:
            palette, token = var.split(":", 1)
            palette_map = ROLE_ZONE_PALETTES.get(palette, {})
            return palette_map.get(token, TEXT.primary)
        return TEXT.primary
    if key == "glow":
        # variant = "win" | "loss" | "blue" | "orange" | "neutral" | "emphasis" | "gold" | "ball"
        return getattr(GLOW, var, GLOW.neutral)
    if key == "emphasis":
        return EMPHASIS_ACCENT

    return TEAM_ACCENTS.get(key, TEXT.primary)


def team_heatmap_colorscale(team: str) -> list:
    """Return a team-aware Plotly colorscale list for density heatmaps.

    Args:
        team: "Blue", "Orange", or any other value for a neutral violet scale.
    """
    key = team.lower() if team else "neutral"
    return HEATMAP_COLORSCALES.get(key, HEATMAP_COLORSCALES["neutral"])


def add_glow_layer(
    fig: Any,
    x: list | Any,
    y: list | Any,
    glow_color: str,
    size: int = 20,
    row: int | None = None,
    col: int | None = None,
) -> Any:
    """Add a semi-transparent radial glow behind scatter markers.

    Inserts a larger, translucent circle trace *before* the primary markers
    so it renders beneath them, creating a soft halo effect.

    Args:
        fig: Plotly Figure to mutate.
        x, y: Coordinate arrays matching the primary trace.
        glow_color: RGBA string from GLOW palette.
        size: Diameter of the glow circle in pixels (default 20).
        row/col: Subplot row/col if using make_subplots.
    """
    kwargs: dict[str, Any] = dict(
        x=x,
        y=y,
        mode="markers",
        marker=dict(size=size, color=glow_color, line=dict(width=0)),
        hoverinfo="skip",
        showlegend=False,
    )
    if row is not None and col is not None:
        fig.add_trace(go.Scatter(**kwargs), row=row, col=col)
    else:
        fig.add_trace(go.Scatter(**kwargs))
    return fig


PRESETS = {
    "hero": {
        "height": 520,
        "margin": dict(l=SPACING.lg, r=SPACING.lg, t=SPACING.xl, b=SPACING.lg),
        "title_size": TYPOGRAPHY.title,
    },
    "support": {
        "height": 340,
        "margin": dict(l=SPACING.md, r=SPACING.md, t=SPACING.lg, b=SPACING.md),
        "title_size": TYPOGRAPHY.subtitle,
    },
    "detail": {
        "height": 260,
        "margin": dict(l=SPACING.sm, r=SPACING.sm, t=SPACING.md, b=SPACING.sm),
        "title_size": TYPOGRAPHY.body,
    },
}


def apply_chart_theme(fig: Any, tier: str = "support", intent: str | None = None, variant: str = "default"):
    """Apply app-wide defaults for Plotly figures."""
    preset = PRESETS.get(tier, PRESETS["support"])
    accent = semantic_color((intent or ""), variant=variant)

    fig.update_layout(
        font=dict(family=TYPOGRAPHY.family, size=TYPOGRAPHY.body, color=TEXT.primary),
        title=dict(font=dict(size=preset["title_size"], color=accent), x=0.01, xanchor="left"),
        margin=preset["margin"],
        height=preset["height"],
        plot_bgcolor=BACKGROUNDS.panel,
        paper_bgcolor=BACKGROUNDS.canvas,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(color=TEXT.secondary, size=TYPOGRAPHY.annotation),
            title_font=dict(color=TEXT.secondary, size=TYPOGRAPHY.annotation),
        ),
        hoverlabel=dict(
            bgcolor=BACKGROUNDS.elevated,
            bordercolor=BACKGROUNDS.elevated,
            font=dict(
                color=TEXT.primary,
                size=TYPOGRAPHY.annotation,
                family=TYPOGRAPHY.family,
            ),
        ),
    )

    fig.update_xaxes(
        showline=False,
        zeroline=False,
        gridcolor=f"rgba(255,255,255,{GRID_OPACITY})",
        ticks="outside",
        tickfont=dict(color=TEXT.secondary, size=TYPOGRAPHY.annotation),
    )
    fig.update_yaxes(
        showline=False,
        zeroline=False,
        gridcolor=f"rgba(255,255,255,{GRID_OPACITY})",
        ticks="outside",
        tickfont=dict(color=TEXT.secondary, size=TYPOGRAPHY.annotation),
    )

    return fig
