"""Shared metric contract for scalar analytics outputs.

This module defines a canonical uncertainty-aware shape used across event,
aggregate, chart, and export layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ScalarMetricContract:
    """Canonical scalar metric with uncertainty and reliability metadata."""

    value: float
    ci_low: float
    ci_high: float
    sample_size: int
    reliability: str

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "value": float(self.value),
            "ci_low": float(self.ci_low),
            "ci_high": float(self.ci_high),
            "sample_size": int(self.sample_size),
            "reliability": str(self.reliability),
        }


def metric_contract(
    value: float,
    *,
    ci_low: float | None = None,
    ci_high: float | None = None,
    sample_size: int = 1,
    reliability: str = "low",
) -> ScalarMetricContract:
    """Build a canonical scalar metric contract."""
    lo = float(value if ci_low is None else ci_low)
    hi = float(value if ci_high is None else ci_high)
    if lo > hi:
        lo, hi = hi, lo
    return ScalarMetricContract(
        value=float(value),
        ci_low=lo,
        ci_high=hi,
        sample_size=max(0, int(sample_size)),
        reliability=reliability,
    )


def flatten_metric_contract(metric_name: str, contract: ScalarMetricContract) -> dict[str, float | int | str]:
    """Flatten to stable export columns while retaining a legacy scalar alias."""
    return {
        metric_name: contract.value,
        f"{metric_name}_Value": contract.value,
        f"{metric_name}_CI_Low": contract.ci_low,
        f"{metric_name}_CI_High": contract.ci_high,
        f"{metric_name}_SampleSize": contract.sample_size,
        f"{metric_name}_Reliability": contract.reliability,
    }


def read_metric_contract(row: Mapping[str, object], metric_name: str) -> ScalarMetricContract:
    """Read contract columns from row-like data, defaulting to legacy scalar values."""
    value = float(row.get(f"{metric_name}_Value", row.get(metric_name, 0.0)) or 0.0)
    ci_low = float(row.get(f"{metric_name}_CI_Low", value) or value)
    ci_high = float(row.get(f"{metric_name}_CI_High", value) or value)
    sample_size = int(row.get(f"{metric_name}_SampleSize", 1) or 1)
    reliability = str(row.get(f"{metric_name}_Reliability", "low") or "low")
    return metric_contract(
        value,
        ci_low=ci_low,
        ci_high=ci_high,
        sample_size=sample_size,
        reliability=reliability,
    )
