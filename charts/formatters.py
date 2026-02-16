"""Shared chart formatting helpers for labels and hover content."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

EXPECTED_VALUE_METRICS = {"xg", "xs", "vaep"}


@dataclass(frozen=True)
class MetricStyle:
    """Declarative style for a metric family."""

    key: str
    precision: int
    units: str


METRIC_STYLES = {
    "percentage": MetricStyle("percentage", precision=1, units="%"),
    "duration": MetricStyle("duration", precision=0, units="mm:ss"),
    "rate": MetricStyle("rate", precision=2, units="rate"),
    "integer": MetricStyle("integer", precision=0, units="count"),
    "expected_value": MetricStyle("expected_value", precision=2, units="xMetric"),
    "decimal": MetricStyle("decimal", precision=2, units="value"),
}


def format_percentage(value: float) -> str:
    return f"{float(value):.1f}%"


def format_duration(seconds: float) -> str:
    value = int(round(float(seconds)))
    minutes, secs = divmod(max(value, 0), 60)
    return f"{minutes}:{secs:02d}"


def format_rate(value: float, *, suffix: str = "") -> str:
    return f"{float(value):.2f}{suffix}"


def format_integer(value: float) -> str:
    return f"{float(value):,.0f}"


def format_expected_value(value: float, *, metric: str = "xMetric") -> str:
    _ = metric
    return f"{float(value):.2f}"


def infer_metric_style(metric_name: str | None) -> MetricStyle:
    normalized = (metric_name or "").lower()
    compact = normalized.replace("_", " ")

    if any(m in normalized for m in EXPECTED_VALUE_METRICS):
        return METRIC_STYLES["expected_value"]
    if "%" in normalized or "pct" in normalized or "percent" in normalized:
        return METRIC_STYLES["percentage"]
    if "time" in compact or "duration" in compact or "airborne" in compact:
        return METRIC_STYLES["duration"]
    if "per " in compact or "rate" in compact or "/" in normalized:
        return METRIC_STYLES["rate"]
    if any(token in compact for token in ("count", "shots", "goals", "assists", "saves", "score", "hits")):
        return METRIC_STYLES["integer"]

    return METRIC_STYLES["decimal"]


def format_metric_value(value: float, *, metric_name: str | None = None, style: MetricStyle | None = None) -> str:
    metric_style = style or infer_metric_style(metric_name)

    if pd.isna(value):
        value = 0

    if metric_style.key == "percentage":
        return format_percentage(value)
    if metric_style.key == "duration":
        return format_duration(value)
    if metric_style.key == "rate":
        return format_rate(value)
    if metric_style.key == "integer":
        return format_integer(value)
    if metric_style.key == "expected_value":
        metric = (metric_name or "xMetric").strip()
        return format_expected_value(value, metric=metric)

    return f"{float(value):,.2f}"


def format_metric_series(values: Iterable[float], *, metric_name: str | None = None) -> list[str]:
    metric_style = infer_metric_style(metric_name)
    return [format_metric_value(v, metric_name=metric_name, style=metric_style) for v in values]


def hover_template(
    *,
    entity: str,
    primary_label: str,
    context_label: str,
    units: str,
    source_note: str | None = None,
) -> str:
    lines = [
        f"Entity: {entity}",
        f"Primary: {primary_label}",
        f"Context: {context_label}",
        f"Units: {units}",
    ]
    if source_note:
        lines.append(f"Source: {source_note}")
    lines.append("<extra></extra>")
    return "<br>".join(lines)
