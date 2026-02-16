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
