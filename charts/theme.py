"""Centralized Plotly chart theme helpers."""

from __future__ import annotations

from typing import Any

from .tokens import (
    BACKGROUNDS,
    DUAL_SERIES_DEFAULTS,
    GRID_OPACITY,
    OUTCOME_COLORS,
    ROLE_ZONE_PALETTES,
    SPACING,
    TEAM_ACCENTS,
    TEXT,
    THRESHOLD_ACCENTS,
    TYPOGRAPHY,
    EVENT_TYPE_MARKERS,
    CONFIDENCE_OPACITY,
    INTENT_COLORS,
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

    return TEAM_ACCENTS.get(key, TEXT.primary)

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



def event_style(*, team: str | None = None, intent: str | None = None, event_type: str | None = None, confidence: float | None = None) -> dict[str, object]:
    """Resolve canonical Director Mode visual grammar tokens."""
    team_key = (team or "").lower()
    intent_key = (intent or "neutral").lower()
    color = TEAM_ACCENTS.get(team_key, INTENT_COLORS.get(intent_key, TEXT.primary))
    marker = EVENT_TYPE_MARKERS.get((event_type or "default").lower(), EVENT_TYPE_MARKERS["default"])
    if confidence is None:
        opacity = CONFIDENCE_OPACITY["medium"]
    elif confidence >= 0.8:
        opacity = CONFIDENCE_OPACITY["high"]
    elif confidence >= 0.55:
        opacity = CONFIDENCE_OPACITY["medium"]
    else:
        opacity = CONFIDENCE_OPACITY["low"]
    return {"color": color, "symbol": marker, "opacity": opacity}
