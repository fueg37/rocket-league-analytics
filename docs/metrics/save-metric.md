# Save Metric Contract

## Overview
This project now separates save analytics into three explicit concepts:

1. **SaveDifficultyIndex (SDI)**: heuristic difficulty index in `[0, 1]`.
2. **ExpectedSaveProb**: expected probability the defending side saves the shot in `[0, 1]`.
3. **SaveImpact**: impact for successful save events, computed as `1 - ExpectedSaveProb`.

In heuristic mode (`heuristic-v2`), `ExpectedSaveProb` is currently derived from SDI by
`ExpectedSaveProb = 1 - SDI` and therefore `SaveImpact = SDI`.

## Event-level fields
- `SaveDifficultyIndex`
- `ExpectedSaveProb`
- `SaveImpact`
- `AttributionSource`
- `AttributionConfidence`

## Summary-level fields
- `SaveEvents`
- `Total_SaveDifficulty`
- `Avg_SaveDifficulty`
- `Total_ExpectedSaves`
- `Actual_Saves`
- `Total_SaveImpact`
- `Avg_SaveImpact`
- `HighDifficultySaves` (quantile-based, default q80)

## Backward compatibility aliases
For compatibility with legacy downstream views:
- `Saves_Nearby` → `SaveEvents`
- `Total_xS` → `Total_SaveDifficulty`
- `Avg_xS` → `Avg_SaveDifficulty`
- `Hard_Saves` → `HighDifficultySaves`

## Invariants
- `0 <= SaveDifficultyIndex <= 1`
- `0 <= ExpectedSaveProb <= 1`
- `0 <= SaveImpact <= 1`
- deterministic rounding for display:
  - totals: 2 decimals
  - averages/event values: 3 decimals

## Developer note: speed unit boundary (design lock)
- **Canonical storage/computation unit:** `uu/s`
- **Default display unit:** `mph`
- **Architecture rule:** unit conversion happens **only** in the presentation layer.

Treat `uu/s` as the foundational domain unit across ingestion, analytics, and model logic.
Any conversion to `mph` should be deferred to chart labels, tooltip text, table formatting, and
other user-facing output.

### Correct usage examples

**Analytics/model code (canonical `uu/s` only):**

```python
# analytics/save_metrics.py (domain logic)
shot_speed = float(shot.get("Speed", 0.0) or 0.0)  # uu/s
speed_norm = min(features.shot_speed / 4000.0, 1.5)  # still uu/s-based
```

**UI/tooltip/metric code (formatted `mph`):**

```python
# charts/factory.py (presentation logic)
from utils import format_speed

speed_display = format_speed(speed_uu, unit="mph", precision=1)  # e.g., "49.2 mph"
```

### Shared conversion helper
Reuse the shared helper in `utils.py`:
- `format_speed(...)` for display-safe formatting
- `uu_per_sec_to_mph(...)` for raw conversion when formatting is handled separately

Do not reimplement conversion constants or ad hoc speed formatters in feature code.
