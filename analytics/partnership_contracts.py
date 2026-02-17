"""Partnership Intelligence contract and deterministic normalization rules.

The partnership contract translates raw/shrunk chemistry features into a stable,
product-facing 0-100 scoring surface.

Formulas
--------
Let ``s`` be a shrunk chemistry value and ``clip01(x)=min(1,max(0,x))``.

1) Primary index (Partnership Index, 0-100)
   ``partnership_index = mean(value_lift, rotation_fit, handoff_quality, pressure_escape)``

2) Component subscales (all 0-100)
   * Value Lift (from expected value gain):
     ``value_lift = 50 + 50 * tanh(expected_value_gain_shrunk / 0.35)``
     then clipped to [0, 100]. This yields ~50 at neutral lift, compresses outliers,
     and is deterministic for arbitrary magnitudes.
   * Rotation Fit:
     ``rotation_fit = 100 * clip01(rotational_complementarity_shrunk)``
   * Handoff Quality:
     ``handoff_quality = 100 * clip01(possession_handoff_efficiency_shrunk)``
   * Pressure Escape:
     ``pressure_escape = 100 * clip01(pressure_release_reliability_shrunk)``

3) Confidence translation
   Reliability buckets map to descriptor labels:
   ``low -> Low``, ``medium -> Medium``, ``high -> High``.

4) Uncertainty projection on index
   Raw chemistry confidence interval bounds are projected through the same Value Lift
   transform used for the primary index domain:
   ``ci_low = index_from_chemistry_score(ci_low_raw)``
   ``ci_high = index_from_chemistry_score(ci_high_raw)``
   with bounds sorted to ensure ``ci_low <= ci_high``.

5) Impact translations
   * ``expected_xgd_lift_per_match = expected_value_gain_shrunk``
   * ``win_rate_lift_points = expected_xgd_lift_per_match * 8.0``

The 8.0 coefficient is a deterministic conversion constant used for a coarse
xGD-to-win-rate interpretation layer. It should be treated as a model policy
parameter, not an empirical law.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

VALUE_LIFT_TANH_SCALE = 0.35
WIN_RATE_POINTS_PER_XGD = 8.0


@dataclass(frozen=True)
class PartnershipContractColumns:
    partnership_index: str = "Partnership Index"
    value_lift: str = "Value Lift"
    rotation_fit: str = "Rotation Fit"
    handoff_quality: str = "Handoff Quality"
    pressure_escape: str = "Pressure Escape"
    confidence_level: str = "confidence_level"
    ci_low: str = "ci_low"
    ci_high: str = "ci_high"
    sample_count: str = "sample_count"
    expected_xgd_lift_per_match: str = "expected_xgd_lift_per_match"
    win_rate_lift_points: str = "win_rate_lift_points"


def _clip_0_100(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)


def value_lift_to_100(value_gain: pd.Series) -> pd.Series:
    raw = pd.to_numeric(value_gain, errors="coerce").fillna(0.0).astype(float)
    transformed = 50.0 + (50.0 * np.tanh(raw / VALUE_LIFT_TANH_SCALE))
    return _clip_0_100(pd.Series(transformed, index=raw.index))


def probability_to_100(values: pd.Series) -> pd.Series:
    return (pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0) * 100.0).astype(float)


def confidence_from_reliability(reliability: str) -> str:
    bucket = str(reliability or "low").strip().lower()
    if bucket == "high":
        return "High"
    if bucket == "medium":
        return "Medium"
    return "Low"


def chemistry_score_to_index(scores: pd.Series) -> pd.Series:
    return value_lift_to_100(scores)


def apply_partnership_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Attach contract fields to a pairwise chemistry DataFrame."""
    if df.empty:
        return df.copy()

    cols = PartnershipContractColumns()
    out = df.copy()

    out[cols.value_lift] = value_lift_to_100(out.get("ExpectedValueGain_Shrunk", out.get("ExpectedValueGain", 0.0)))
    out[cols.rotation_fit] = probability_to_100(out.get("RotationalComplementarity_Shrunk", out.get("RotationalComplementarity", 0.0)))
    out[cols.handoff_quality] = probability_to_100(out.get("PossessionHandoffEfficiency_Shrunk", out.get("PossessionHandoffEfficiency", 0.0)))
    out[cols.pressure_escape] = probability_to_100(out.get("PressureReleaseReliability_Shrunk", out.get("PressureReleaseReliability", 0.0)))
    out[cols.partnership_index] = out[[cols.value_lift, cols.rotation_fit, cols.handoff_quality, cols.pressure_escape]].mean(axis=1)

    ci_low_raw = pd.to_numeric(out.get("CI_Low", out.get("ChemistryScore_Shrunk", 0.0)), errors="coerce").fillna(0.0)
    ci_high_raw = pd.to_numeric(out.get("CI_High", out.get("ChemistryScore_Shrunk", 0.0)), errors="coerce").fillna(0.0)
    ci_low_idx = chemistry_score_to_index(ci_low_raw)
    ci_high_idx = chemistry_score_to_index(ci_high_raw)
    out[cols.ci_low] = np.minimum(ci_low_idx, ci_high_idx)
    out[cols.ci_high] = np.maximum(ci_low_idx, ci_high_idx)

    out[cols.sample_count] = pd.to_numeric(out.get("Samples", 0), errors="coerce").fillna(0).astype(int)
    out[cols.confidence_level] = out.get("Reliability", "low").apply(confidence_from_reliability)

    out[cols.expected_xgd_lift_per_match] = pd.to_numeric(
        out.get("ExpectedValueGain_Shrunk", out.get("ExpectedValueGain", 0.0)),
        errors="coerce",
    ).fillna(0.0).astype(float)
    out[cols.win_rate_lift_points] = out[cols.expected_xgd_lift_per_match] * WIN_RATE_POINTS_PER_XGD
    return out
