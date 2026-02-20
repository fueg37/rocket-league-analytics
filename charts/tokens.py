"""Immutable design tokens for chart theming."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class TypographyScale:
    family: str = "Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
    title: int = 20
    subtitle: int = 16
    body: int = 13
    annotation: int = 11
    weight_bold: int = 700
    weight_semibold: int = 600
    weight_normal: int = 400


@dataclass(frozen=True)
class SpacingScale:
    xxs: int = 4
    xs: int = 8
    sm: int = 12
    md: int = 16
    lg: int = 24
    xl: int = 32


@dataclass(frozen=True)
class PanelBackgrounds:
    canvas: str = "rgba(0,0,0,0)"
    panel: str = "#1e1e1e"
    elevated: str = "#252525"
    deep: str = "#141418"


@dataclass(frozen=True)
class NeutralTextColors:
    primary: str = "#f3f4f6"
    secondary: str = "#c9d1d9"
    muted: str = "#94a3b8"


@dataclass(frozen=True)
class GlowPalette:
    """Faint colored halos used behind primary markers to create depth."""
    win: str = "rgba(34,197,94,0.22)"
    loss: str = "rgba(239,68,68,0.22)"
    blue: str = "rgba(59,130,246,0.22)"
    orange: str = "rgba(251,146,60,0.22)"
    neutral: str = "rgba(148,163,184,0.18)"
    emphasis: str = "rgba(167,139,250,0.25)"
    gold: str = "rgba(255,204,0,0.28)"
    ball: str = "rgba(255,240,180,0.30)"


TYPOGRAPHY = TypographyScale()
SPACING = SpacingScale()
BACKGROUNDS = PanelBackgrounds()
TEXT = NeutralTextColors()
GLOW = GlowPalette()
GRID_OPACITY = 0.14

# Violet accent for hero/emphasis single-metric highlights.
EMPHASIS_ACCENT: str = "#a78bfa"

_TEAM_ACCENTS: Mapping[str, str] = {
    "blue": "#3b82f6",
    "orange": "#fb923c",
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral": "#94a3b8",
}
TEAM_ACCENTS = MappingProxyType(dict(_TEAM_ACCENTS))

OUTCOME_COLORS = MappingProxyType(
    {
        "win": TEAM_ACCENTS["positive"],
        "loss": TEAM_ACCENTS["negative"],
        "neutral": "#636efa",
    }
)

ROLE_ZONE_PALETTES = MappingProxyType(
    {
        "positioning": MappingProxyType(
            {
                "defense": "#EF553B",
                "midfield": "#FFA15A",
                "offense": "#00CC96",
            }
        ),
        "granular_zone": MappingProxyType(
            {
                "defensive_third": "#636efa",
                "backboard": "#EF553B",
                "midfield": "#AB63FA",
                "offensive_third": "#00CC96",
            }
        ),
    }
)

THRESHOLD_ACCENTS = MappingProxyType(
    {
        "positive": TEAM_ACCENTS["positive"],
        "negative": TEAM_ACCENTS["negative"],
        "neutral": "#ffcc00",
    }
)

DUAL_SERIES_DEFAULTS = MappingProxyType(
    {
        "primary": TEAM_ACCENTS["blue"],
        "secondary": TEAM_ACCENTS["orange"],
        "comparison_left": "#8C9AAD",
        "comparison_right": TEAM_ACCENTS["positive"],
    }
)

# Team-aware density heatmap colorscales (Plotly [[stop, color], ...] format).
# Each scale starts fully transparent so the pitch image bleeds through at
# low-density areas, then saturates to a vivid team-hued hot colour.
HEATMAP_COLORSCALES: dict = {
    "blue": [
        [0.00, "rgba(0,0,0,0)"],
        [0.18, "rgba(15,40,100,0.40)"],
        [0.38, "rgba(30,90,200,0.58)"],
        [0.58, "rgba(60,150,255,0.72)"],
        [0.78, "rgba(120,200,255,0.86)"],
        [1.00, "rgba(205,235,255,0.96)"],
    ],
    "orange": [
        [0.00, "rgba(0,0,0,0)"],
        [0.18, "rgba(90,30,0,0.40)"],
        [0.38, "rgba(190,85,0,0.58)"],
        [0.58, "rgba(245,145,30,0.72)"],
        [0.78, "rgba(255,205,80,0.86)"],
        [1.00, "rgba(255,245,170,0.96)"],
    ],
    "neutral": [
        [0.00, "rgba(0,0,0,0)"],
        [0.18, "rgba(35,15,75,0.40)"],
        [0.38, "rgba(90,30,180,0.58)"],
        [0.58, "rgba(165,65,240,0.72)"],
        [0.78, "rgba(220,130,255,0.86)"],
        [1.00, "rgba(245,210,255,0.96)"],
    ],
}
