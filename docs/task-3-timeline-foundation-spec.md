# Task 3 Spec: Canonical Timeline Foundation + Narrative Trio Migration

## Why this task exists now

You already have:
- global design tokens/theme (`charts/tokens.py`, `charts/theme.py`), and
- canonical comparison/ranking factories (`comparison_dumbbell`, `player_rank_lollipop`).

The next highest-leverage gap is timeline consistency across the Single Match Narrative trio:
- **SM-MN-02** Win Probability,
- **SM-MN-03** Cumulative xG,
- **SM-MN-04** Pressure Index.

This task makes timeline grammar foundational rather than ad hoc.

---

## Task mapping (matrix IDs)

- `SM-MN-02` (`line/area`, hero)
- `SM-MN-03` (`line`, hero)
- `SM-MN-04` (`line/area`, support)

Reference: `docs/chart-migration-matrix.md`.

---

## Foundational redesign principle

If this app had been designed with timeline analytics as a first-class assumption from day one, all temporal charts would share:

1. one constructor surface,
2. one hover grammar,
3. one axis/time semantic model,
4. one event annotation system,
5. one style hierarchy (`hero`/`support`/`detail`).

This task establishes exactly that.

---

## Deliverables

### D1) Add canonical timeline factory
Create `timeline_chart(...)` in `charts/factory.py`.

**Required capabilities:**
- Multi-series line rendering (for team/player traces).
- Optional area fills (`fill_to_zero`, `fill_to_next`).
- Optional reference lines (e.g., 50% win prob).
- Optional vertical markers (e.g., OT at 300s).
- Optional event markers (e.g., goals on cumulative xG).
- Deterministic series order.
- Optional endpoint labels.

**Proposed signature (minimum):**
```python
def timeline_chart(
    df: pd.DataFrame,
    x_col: str,
    y_specs: list[dict],
    *,
    title: str,
    x_title: str,
    y_title: str,
    tier: str = "support",
    y_range: tuple[float, float] | None = None,
    reference_lines: list[dict] | None = None,
    vertical_markers: list[dict] | None = None,
    event_markers: list[dict] | None = None,
    endpoint_labels: bool = False,
    hover_kind: str = "value",
):
    ...
```

### D2) Add shared timeline formatting helpers
Create `charts/formatters.py` with timeline-safe formatters:
- `fmt_percent_1dp`
- `fmt_decimal_2dp`
- `fmt_seconds_int`
- `build_hover(metric_label, unit, precision)`

All three migrated charts must use these helpers (no inline hover string duplication).

### D3) Migrate Narrative trio to timeline factory
In `app.py`, replace ad hoc construction for:
- Win Probability,
- Cumulative xG,
- Pressure Index,

with `timeline_chart(...)` calls + explicit semantic config.

### D4) Add synchronization semantics
Ensure the three charts share compatible time-domain semantics:
- same x-axis unit (`Time (s)`),
- same OT marker rule,
- consistent hover `Time` formatting.

### D5) Matrix + docs update
When complete:
- mark `SM-MN-02/03/04` as done in migration matrix,
- add one short note to `CLAUDE.md` indicating: *Timeline charts must use `timeline_chart` + `charts/formatters.py`*.

---

## Non-negotiable invariants for this task

1. **No inline style literals** in `app.py` call sites for these migrated charts.
2. **Deterministic ordering** of series (Blue/Orange or declared order).
3. **Unified hover language**: `Metric: value unit` with consistent precision.
4. **Accessibility fallback**: each chart has a one-sentence textual summary below it.
5. **Tier compliance**: `hero` for Win Prob + Cumulative xG; `support` for Pressure.

---

## Implementation plan (recommended PR slicing)

### PR A: Foundation only
- Add `timeline_chart(...)` and `charts/formatters.py`.
- Add/adjust tests for constructor behavior.
- No `app.py` migration yet.

### PR B: Win Probability + Cumulative xG
- Migrate `SM-MN-02` and `SM-MN-03`.
- Validate legend order, hover precision, OT marker.

### PR C: Pressure Index + cleanup
- Migrate `SM-MN-04`.
- Remove duplicate old helpers if obsolete.
- Update matrix statuses and `CLAUDE.md` note.

This split keeps rollback simple and review focused.

---

## Validation checklist

### Functional
- Chart renders without errors for:
  - regulation matches,
  - overtime matches,
  - matches with sparse events.
- Event markers appear in correct time positions.
- Reference line appears only when configured.

### Consistency
- Same hover grammar across all three charts.
- Same time-axis label and tick strategy.
- Stable trace ordering across reruns.

### Accessibility
- Contrast remains readable in dark theme.
- Text summary present below each migrated chart.
- Color is not the only cue where critical meaning exists.

---

## Definition of done

Task 3 is complete when:
1. `timeline_chart(...)` is the sole constructor path for these three charts,
2. matrix rows `SM-MN-02/03/04` are marked complete,
3. no regression in chart rendering for OT/non-OT scenarios,
4. code review confirms invariants are met.

---

## Optional stretch (if time permits)

- Add shared x-domain linking utility so future timeline families can opt into synchronized zoom.
- Add endpoint annotation helper for last-known value callouts.
- Add one visual regression screenshot for each migrated chart type.
