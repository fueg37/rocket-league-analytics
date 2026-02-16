"""Centralized Plotly chart theme helpers."""

from __future__ import annotations

from typing import Any

from .tokens import BACKGROUNDS, GRID_OPACITY, SPACING, TEAM_ACCENTS, TEXT, TYPOGRAPHY

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


def apply_chart_theme(fig: Any, tier: str = "support", intent: str | None = None):
    """Apply app-wide defaults for Plotly figures."""
    preset = PRESETS.get(tier, PRESETS["support"])
    accent = TEAM_ACCENTS.get((intent or "").lower(), TEXT.primary)

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
            font=dict(color=TEXT.primary, size=TYPOGRAPHY.annotation),
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
