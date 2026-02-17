"""Centralized display formatting for charts and table surfaces."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from utils import uu_per_sec_to_mph

_ACRONYM_PRESERVE = {
    "xg": "xG",
    "xgot": "xGOT",
    "xga": "xGA",
    "vaep": "VAEP",
    "sdi": "SDI",
}

_FAMILY_KEYWORDS = {
    "percent": ("%", "rate", "win rate", "shooting", "possession"),
    "seconds": ("time", "second", "seconds", "duration"),
    "speed": ("speed",),
    "xg": ("xg", "xgot", "xga", "expected"),
    "integer": (
        "goal",
        "shot",
        "save",
        "assist",
        "score",
        "event",
        "count",
        "hit",
    ),
}


def title_case_label(label: str) -> str:
    """Title-case labels while preserving analytics acronyms."""
    if not label:
        return ""
    words = re.split(r"(\s+)", str(label).replace("_", " ").strip())
    out: list[str] = []
    for token in words:
        lowered = token.lower()
        out.append(_ACRONYM_PRESERVE.get(lowered, token.title()))
    return "".join(out)


def metric_family(metric_name: str) -> str:
    name = str(metric_name or "").lower()
    for family, needles in _FAMILY_KEYWORDS.items():
        if any(needle in name for needle in needles):
            return family
    return "decimal"


def unit_suffix(metric_name: str, family: str | None = None) -> str:
    fam = family or metric_family(metric_name)
    if fam == "percent":
        return "%"
    if fam == "seconds":
        return "s"
    if fam == "speed":
        return "mph"
    if fam == "xg":
        return "xG"
    return ""


def decimal_precision(metric_name: str, family: str | None = None) -> int:
    fam = family or metric_family(metric_name)
    if fam == "integer":
        return 0
    if fam in {"percent", "seconds", "speed"}:
        return 1
    if fam == "xg":
        return 2
    return 2


def format_metric_value(value: Any, metric_name: str, include_unit: bool = True) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "N/A"

    fam = metric_family(metric_name)
    display_value = float(numeric)
    if fam == "speed":
        display_value = uu_per_sec_to_mph(display_value)

    precision = decimal_precision(metric_name, family=fam)
    value_text = f"{display_value:,.{precision}f}"
    if precision == 0:
        value_text = f"{display_value:,.0f}"

    if include_unit:
        suffix = unit_suffix(metric_name, family=fam)
        if suffix:
            return f"{value_text} {suffix}"
    return value_text


def format_metric_series(series: pd.Series, metric_name: str, include_unit: bool = False) -> list[str]:
    return [format_metric_value(value, metric_name, include_unit=include_unit) for value in series]


def tooltip_template(metric_label: str, *, include_player: bool = False, include_team: bool = False, include_time: bool = False, value_index: int = 0) -> str:
    """Create a standard tooltip template using shared context field order."""
    lines: list[str] = []
    cursor = 0
    if include_player:
        lines.append(f"Player: %{{customdata[{cursor}]}}")
        cursor += 1
    if include_team:
        lines.append(f"Team: %{{customdata[{cursor}]}}")
        cursor += 1
    if include_time:
        lines.append(f"Time: %{{customdata[{cursor}]}}")
        cursor += 1
    lines.append(f"Metric: {title_case_label(metric_label)}: %{{customdata[{value_index}]}}")
    return "<br>".join(lines) + "<extra></extra>"


def dataframe_formatter(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Return dataframe styler with metric-aware value formatting."""
    formatters: dict[str, Any] = {}
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            formatters[column] = lambda value, col=column: format_metric_value(value, col)
    return df.style.format(formatters)



def reliability_badge(reliability: str | None, sample_size: int | None = None) -> str:
    """Render compact reliability badge text for tooltips/labels."""
    level = str(reliability or "low").lower()
    icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(level, "⚪")
    n_text = "" if sample_size is None else f" (n={int(sample_size)})"
    return f"{icon} {level.title()}{n_text}"


def format_confidence_interval(
    value: Any,
    ci_low: Any,
    ci_high: Any,
    metric_name: str,
    *,
    include_unit: bool = True,
) -> str:
    """Format a metric with uncertainty interval for UI surfaces."""
    center = format_metric_value(value, metric_name, include_unit=include_unit)
    low = format_metric_value(ci_low, metric_name, include_unit=include_unit)
    high = format_metric_value(ci_high, metric_name, include_unit=include_unit)
    return f"{center} [{low}, {high}]"
