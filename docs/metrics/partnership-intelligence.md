# Partnership Intelligence Contract

`analytics/partnership_contracts.py` defines the canonical pairwise chemistry output contract.

## Contract fields

- **Primary index**: `Partnership Index` (0–100)
- **Component scores** (0–100 each):
  - `Value Lift`
  - `Rotation Fit`
  - `Handoff Quality`
  - `Pressure Escape`
- **Uncertainty**:
  - `confidence_level` (`Low`, `Medium`, `High`)
  - `ci_low` / `ci_high` (index-scale bounds)
  - `sample_count`
- **Impact translations**:
  - `expected_xgd_lift_per_match`
  - `win_rate_lift_points`

## Deterministic scaling rules

Let `clip01(x)=min(1,max(0,x))`.

1. `Value Lift = clip(50 + 50*tanh(ExpectedValueGain_Shrunk/0.35), 0, 100)`
2. `Rotation Fit = 100*clip01(RotationalComplementarity_Shrunk)`
3. `Handoff Quality = 100*clip01(PossessionHandoffEfficiency_Shrunk)`
4. `Pressure Escape = 100*clip01(PressureReleaseReliability_Shrunk)`
5. `Partnership Index = mean(Value Lift, Rotation Fit, Handoff Quality, Pressure Escape)`

## Uncertainty and confidence

- `ci_low` and `ci_high` are produced by projecting legacy chemistry CI bounds (`CI_Low`, `CI_High`) through the same index transform and sorting bounds.
- Reliability buckets map to confidence labels:
  - `low -> Low`
  - `medium -> Medium`
  - `high -> High`

## Impact translations

- `expected_xgd_lift_per_match = ExpectedValueGain_Shrunk`
- `win_rate_lift_points = expected_xgd_lift_per_match * 8.0`

## Migration compatibility

`analytics/chemistry.py` now emits contract fields and keeps legacy chemistry columns (`ChemistryScore_*`, component shrunk fields, `CI_Low`, `CI_High`, `Reliability`, etc.) as compatibility aliases during migration.
