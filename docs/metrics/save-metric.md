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
