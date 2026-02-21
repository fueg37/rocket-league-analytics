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


@dataclass(frozen=True)
class NeutralTextColors:
    primary: str = "#f3f4f6"
    secondary: str = "#c9d1d9"
    muted: str = "#94a3b8"


TYPOGRAPHY = TypographyScale()
SPACING = SpacingScale()
BACKGROUNDS = PanelBackgrounds()
TEXT = NeutralTextColors()
GRID_OPACITY = 0.14

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


EVENT_TYPE_MARKERS = MappingProxyType(
    {
        "win_probability_swing": "diamond",
        "shot_chance": "circle",
        "kickoff": "square",
        "value_swing": "triangle-up",
        "save": "x",
        "default": "circle-open",
    }
)

CONFIDENCE_OPACITY = MappingProxyType(
    {
        "high": 0.95,
        "medium": 0.7,
        "low": 0.45,
    }
)

INTENT_COLORS = MappingProxyType(
    {
        "attack": TEAM_ACCENTS["orange"],
        "defense": TEAM_ACCENTS["blue"],
        "neutral": TEAM_ACCENTS["neutral"],
    }
)
