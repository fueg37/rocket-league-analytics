# CLAUDE.md — Rocket League Analytics

## Project Overview

Streamlit-based analytics platform for Rocket League replay analysis. Parses `.replay` files using the `carball` library, computes 25+ advanced metrics (xG, xGOT, save difficulty, win probability, VAEP, momentum, etc.), and renders interactive Plotly visualizations with a unified theming system. Supports single-match deep dives and season-long career tracking.

## Tech Stack

- **Language:** Python 3.11+
- **Web framework:** Streamlit
- **Visualizations:** Plotly (2D/3D), Pillow (image compositing)
- **Data processing:** pandas, NumPy
- **Replay parsing:** carball (Sprocket framework)
- **ML (optional):** scikit-learn (logistic regression for win probability)
- **Image export (optional):** kaleido

## File Structure

```
app.py                          # Main application (~4,500 lines) — all UI, analytics orchestration, and rendering
constants.py                    # Field geometry, team colors, persistence config, replay FPS, speed units
utils.py                        # Player/team mapping helpers, frame/time conversion, speed formatting
requirements.txt                # Python dependencies
simple-pitch.png                # Pitch background image for 2D visualizations
launch_rocket_app.bat           # Windows launch script
career_heatmaps.json            # Cached heatmap data (generated at runtime)

analytics/
  __init__.py                   # Analytics package init
  save_metrics.py               # Canonical save difficulty/impact metrics (SDI, xS, attribution)
  shot_quality.py               # Shot quality metrics (xG, xGOT, trajectory analysis)

charts/
  factory.py                    # Reusable chart builders (kickoff KPIs, spatial scatters, lollipops, dumbbells, goal mouth)
  formatters.py                 # Metric-aware display formatting (speed, time, xG, percentages)
  theme.py                      # Centralized Plotly theme with semantic color system
  tokens.py                     # Immutable design tokens (typography, spacing, backgrounds, palettes)
  win_probability.py            # Win probability chart builders and event extraction

scripts/
  lint_no_hex_in_app_charts.py # Pre-commit linter to prevent hardcoded hex colors in chart call sites

tests/
  test_canonical_ordering.py    # Tests for deterministic categorical ordering
  test_save_metrics.py          # Tests for save metric contract and calculations
  test_speed_units.py           # Tests for speed unit conversion and normalization

docs/
  metrics/save-metric.md        # Save metric contract specification (SDI, ExpectedSaveProb, SaveImpact)
  chart-migration-matrix.md     # Chart migration planning matrix for theming system
```

## Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

Windows shortcut: `launch_rocket_app.bat` (calls `py -3.11 -m streamlit run app.py`)

## Architecture

### Module Responsibilities

#### Core Modules

- **`constants.py`** — Single source of truth for field geometry (Unreal Units), team colors (RGBA/hex), persistence file paths, app configuration (`REPLAY_FPS = 30`, `MAX_STORED_MATCHES = 12`), speed unit conversions (`UU_PER_SEC_TO_MPH`), and canonical categorical orderings (kickoff spawns, game states, zones, teams).
- **`utils.py`** — Reusable utility functions for player/team mapping, frame/time conversion, speed formatting, and deterministic sorting. Eliminates 14+ duplicated mapping patterns from app.py. Functions: `build_pid_team_map`, `build_pid_name_map`, `build_player_team_map`, `get_team_players`, `build_player_positions`, `frame_to_seconds`, `seconds_to_frame`, `fmt_time`, `uu_per_sec_to_mph`, `normalize_speed_uu_per_sec`, `format_speed`, `apply_categorical_order`, `stable_sort`.
- **`app.py`** — Main Streamlit application containing UI rendering, analytics orchestration, data persistence, and export logic. Delegates to specialized analytics and chart modules for computation and visualization.

#### Analytics Package (`analytics/`)

- **`shot_quality.py`** — Shot quality computations and trajectory analysis. Provides xG (pre-shot expected goals) and xGOT (post-shot expected goals on target) as first-class metrics. Includes shot feature extraction, goal-plane projection, and canonical shot event schema.
- **`save_metrics.py`** — Save analytics with explicit separation of concerns: SaveDifficultyIndex (heuristic-based), ExpectedSaveProb (probability model), and SaveImpact (successful save value). Handles save-touch attribution (explicit save flags vs. nearest-defender fallback) and player-level aggregation. Model version: `heuristic-v2`.

#### Charts Package (`charts/`)

- **`theme.py`** — Centralized Plotly theming system with `apply_chart_theme()` function supporting three tiers: hero (520px), support (340px), detail (260px). Provides `semantic_color()` resolver for intent-based color selection (outcome, threshold, dual_series, role_zone, team).
- **`tokens.py`** — Immutable design tokens using dataclasses and frozen mappings. Defines typography scale, spacing scale, panel backgrounds, text colors, grid opacity, team accents, outcome colors, role/zone palettes, threshold accents, and dual-series defaults.
- **`formatters.py`** — Metric-aware display formatting with family detection (percent, seconds, speed, xG, integer, decimal). Functions: `title_case_label`, `metric_family`, `unit_suffix`, `decimal_precision`, `format_metric_value`, `format_metric_series`, `tooltip_template`, `dataframe_formatter`.
- **`factory.py`** — Reusable chart factory functions: `kickoff_kpi_indicator`, `spatial_outcome_scatter`, `rolling_trend_with_wl_markers`, `session_composite_chart`, `player_rank_lollipop`, `comparison_dumbbell`, `goal_mouth_scatter`. All use semantic colors and shared theme presets.
- **`win_probability.py`** — Win probability chart builders with semantic state coloring (Blue favored, Toss-up, Orange favored), state debouncing to reduce flicker, and goal event overlay. Includes `extract_goal_events()` for timeline annotation.

### Data Flow

```
Replay file upload → carball parsing (cached) → _compute_match_analytics()
  → 15+ specialized calculation functions → unified results dict
  → session_state storage (12-match LRU cache) → UI tab rendering
  → CSV/JSON persistence for career tracking
```

### Application Modes

1. **Single Match Analysis** — Upload a `.replay` file, explore 11 tabs: Kickoffs, Narrative, Shot Map, Shot Viewer, Pass Map, Heatmaps, Speed, Advanced, Rotation, Tactical (3D), Export.
2. **Season Batch Processor** — Upload multiple replays, explore 9 tabs: Performance, Season Kickoffs, Playstyle, Radar, Insights, Situational, Log, Sessions, Export.

### Key Computation Functions

#### In `app.py`

| Function | Purpose |
|----------|---------|
| `_compute_match_analytics()` | Central orchestrator — calls all analysis functions, returns unified dict |
| `calculate_kickoff_stats()` | Kickoff success/failure patterns with spawn location tracking |
| `calculate_shot_data()` | Shot event tracking — delegates to `analytics.shot_quality` for xG/xGOT computation |
| `calculate_win_probability()` | Match win probability curve using logistic regression or fallback heuristics |
| `calculate_contextual_momentum()` | Real-time momentum with contextual weighting (goals, saves, shots, possessions) |
| `calculate_advanced_passing()` | Pass accuracy, flow detection, and network topology |
| `calculate_aerial_stats()` | Aerial mechanics tracking and success rate |
| `calculate_defensive_pressure()` | Shadow defense metrics and pressure time |
| `calculate_vaep()` | Value Added by Each Player (event-level impact scoring) |
| `calculate_rotation_analysis()` | Positional rotation patterns and double-commit detection |
| `calculate_situational_stats()` | Clutch performance, game state splits (leading/tied/trailing) |
| `calculate_luck_percentage()` | Poisson binomial luck metric (actual goals vs. expected goals) |
| `calculate_recovery_time()` | Ground recovery tracking (time to supersonic after hit) |

#### In `analytics.shot_quality`

| Function | Purpose |
|----------|---------|
| `compute_shot_features()` | Derive geometry/trajectory features (distance, angle, speed, on-target projection) |
| `calculate_xg_probability()` | Pre-shot expected-goal probability (xG) using angle, distance, speed, height |
| `calculate_xgot_probability()` | Post-shot expected-goal-on-target probability (xGOT) using target placement |
| `project_to_goal_plane()` | Project ball trajectory to goal plane (Y=target_y) returning (target_x, target_z) |
| `validate_shot_metric_columns()` | Validate required shot metric columns before chart rendering |

#### In `analytics.save_metrics`

| Function | Purpose |
|----------|---------|
| `build_save_events()` | Build event-level save analytics rows with attribution and scoring |
| `aggregate_save_summary()` | Aggregate canonical save summary per player with legacy aliases |
| `calculate_save_analytics()` | Main entrypoint — returns event + summary dataframes |
| `resolve_saver_for_shot()` | Resolve saver with explicit save-touch attribution first, nearest fallback second |
| `score_save_heuristic()` | Heuristic scoring backend (SDI, ExpectedSaveProb, SaveImpact) |
| `build_save_features()` | Build canonical features for save scoring (speed, distance, angle, height, saver distance) |

### Caching Strategy

```python
@st.cache_data      — get_field_layout() (field SVG)
@st.cache_resource  — get_3d_field_traces(), get_parsed_replay_data(), load_win_prob_model()
```

Session state holds up to 12 matches in `st.session_state.match_store` (dict keyed by filename), with `match_order` tracking insertion order and `active_match` pointing to the currently viewed match.

### Architecture Improvements (Recent Refactorings)

**Modularization:**
- Extracted reusable analytics to `analytics/` package (save metrics, shot quality)
- Extracted chart theming to `charts/` package (theme, tokens, formatters, factory, win probability)
- Reduced app.py from monolithic ~3,900 lines to more maintainable ~4,500 lines with clear module boundaries

**Metric Quality:**
- **Shot quality:** xG and xGOT now co-equal first-class metrics with canonical schema
- **Save analytics:** Explicit semantic separation of SaveDifficultyIndex, ExpectedSaveProb, SaveImpact
- **Speed units:** Architectural lock enforcing uu/s for computation, mph for display only
- **Attribution:** Save-touch attribution uses explicit game events first, nearest-defender fallback second

**Chart Consistency:**
- **Design tokens:** Immutable typography, spacing, color palettes shared across 90+ charts
- **Semantic colors:** Intent-based color resolution (outcome/threshold/dual_series) eliminates hardcoded hex
- **Factory patterns:** Reusable chart builders reduce duplication for common patterns (lollipops, dumbbells, spatial scatters)
- **Formatters:** Metric-aware display formatting with family detection (speed, time, xG, percent, integer)

**Quality Tooling:**
- **Linter:** Pre-commit check prevents hardcoded hex colors in new chart code
- **Tests:** 198 lines of tests covering save metrics, speed units, canonical ordering
- **Documentation:** Formal metric contracts and migration tracking matrix

### Data Persistence

| File | Format | Purpose |
|------|--------|---------|
| `career_stats.csv` | CSV | Player match statistics (append-only, deduplicated by MatchID) |
| `career_kickoffs.csv` | CSV | Kickoff-specific career data |
| `career_heatmaps.json` | JSON | Pre-processed position heatmap matrices |
| `win_prob_model.json` | JSON | Trained logistic regression model coefficients |

### Optional Dependencies & Feature Flags

```python
SPROCKET_AVAILABLE  # carball — REQUIRED, app stops without it
KALEIDO_AVAILABLE   # kaleido — enables PNG image export
PIL_AVAILABLE       # Pillow — enables image compositing for export panels
SKLEARN_AVAILABLE   # scikit-learn — enables trained win probability model
```

The app degrades gracefully: if optional deps are missing, related features are hidden rather than erroring.

## Coding Conventions

### Core Patterns

- **Analytics logic** is split between `app.py` (orchestration, UI-specific analytics) and `analytics/` modules (reusable metric computation). All analytics are functional — no classes.
- **Team identification** uses `"Blue"` / `"Orange"` string literals mapped via `is_orange` boolean from carball proto objects.
- **Field coordinates** are in Unreal Units (X = sideline ±4096, Y = end line ±5120, Z = height up to 2044). Constants defined in `constants.py`: `FIELD_HALF_X`, `FIELD_HALF_Y`, `WALL_HEIGHT`, `GOAL_HALF_W`, `GOAL_HEIGHT`.
- **Frame/time conversion** uses `REPLAY_FPS = 30` from constants. Always use `frame_to_seconds()` / `seconds_to_frame()` from utils.
- **Player ID keys** are always `str(p.id.id)` — string, not int.
- **Proto object** refers to the carball protobuf `AnalysisProto` returned by `get_parsed_replay_data()`.
- Section comments in `app.py` use the pattern `# --- N. SECTION NAME ---`.

### Speed Unit Architecture (Design Lock)

**CRITICAL:** Speed values have a strict canonical storage/display boundary:

- **Canonical storage/computation unit:** `uu/s` (Unreal Units per second)
- **Default display unit:** `mph` (miles per hour)
- **Conversion constant:** `UU_PER_SEC_TO_MPH = 0.0223694` (defined in `constants.py`)
- **Architecture rule:** Unit conversion happens **only** in the presentation layer.

**Correct usage:**

```python
# Analytics/model code (canonical uu/s only):
from constants import SUPERSONIC_SPEED_UU_PER_SEC
shot_speed = float(shot.get("Speed", 0.0) or 0.0)  # uu/s
speed_norm = min(shot_speed / 4000.0, 1.5)  # still uu/s-based

# UI/tooltip/metric code (formatted mph):
from charts.formatters import format_metric_value
speed_display = format_metric_value(speed_uu, "Speed")  # e.g., "49.2 mph"

# Or raw conversion when formatting is handled separately:
from utils import uu_per_sec_to_mph
speed_mph = uu_per_sec_to_mph(speed_uu)
```

**Do NOT:**
- Reimplement conversion constants
- Store speeds in mph in analytics dataframes
- Mix units in computation logic

See `docs/metrics/save-metric.md` for full specification.

### Chart Theming System

**CRITICAL:** No hardcoded color literals in chart call sites:

- **Use semantic colors:** `semantic_color(intent, variant)` from `charts.theme`
  - Intents: `"outcome"`, `"threshold"`, `"dual_series"`, `"role_zone"`, `"team"`
  - Common variants: `"win"/"loss"/"neutral"`, `"positive"/"negative"`, `"primary"/"secondary"`
- **Use theme presets:** `apply_chart_theme(fig, tier="hero"|"support"|"detail")`
- **Use formatters:** `format_metric_value(value, metric_name)` for consistent units/precision
- **Use factory functions:** Prefer `charts.factory` helpers over inline Plotly construction for common patterns

**Linting:** `scripts/lint_no_hex_in_app_charts.py` enforces no new hex literals in `app.py` chart code.

**Correct usage:**

```python
from charts.theme import apply_chart_theme, semantic_color
from charts.factory import player_rank_lollipop

fig = player_rank_lollipop(df, "Total_SaveImpact", name_col="Name", team_col="Team")
# Factory function handles theming internally

# Or for custom charts:
fig = go.Figure()
fig.add_trace(go.Bar(
    x=values,
    y=labels,
    marker_color=semantic_color("outcome", "win")  # NOT #22c55e
))
fig = apply_chart_theme(fig, tier="support", intent="outcome", variant="win")
```

**Do NOT:**
- Use hardcoded hex codes like `"#3b82f6"` or `"rgba(59,130,246,1)"`
- Duplicate theming logic across chart sites
- Skip `apply_chart_theme()` on new charts

### Canonical Column Schemas

**Shot events** (from `analytics.shot_quality`):
```python
SHOT_EVENT_COLUMNS = (
    "Player", "Team", "Frame", "xG", "xGOT", "OnTarget",
    "TargetX", "TargetZ", "ShotZ", "GoalkeeperDist",
    "ShotAngle", "DistToGoal", "Result", "BigChance", "X", "Y", "Speed"
)
```

**Save events** (from `analytics.save_metrics`):
```python
CANONICAL_EVENT_COLUMNS = [
    "Saver", "Team", "Frame", "Time", "Shooter",
    "AttributionSource", "AttributionConfidence",
    "ShotSpeed", "DistToGoal", "AngleOffCenter", "ShotHeight", "SaverDist",
    "SaveDifficultyIndex", "ExpectedSaveProb", "SaveImpact"
]

CANONICAL_SUMMARY_COLUMNS = [
    "Name", "Team", "SaveEvents",
    "Total_SaveDifficulty", "Avg_SaveDifficulty",
    "Total_ExpectedSaves", "Actual_Saves",
    "Total_SaveImpact", "Avg_SaveImpact", "HighDifficultySaves"
]
```

### Deterministic Sorting

All categorical data must use explicit, stable sorting:

```python
from utils import apply_categorical_order, stable_sort
from constants import KICKOFF_SPAWN_ORDER, GAME_STATE_ORDER, ZONE_ORDER, TEAM_ORDER

# For categorical columns:
df = apply_categorical_order(df, "Spawn", KICKOFF_SPAWN_ORDER)

# For numeric sorting with tie-breaks:
df = stable_sort(df, by=["Score", "Name"], ascending=[False, True])
```

**Canonical orderings** (in `constants.py`):
- `KICKOFF_SPAWN_ORDER = ["Center", "Off-Center", "Diagonal"]`
- `GAME_STATE_ORDER = ["Leading", "Tied", "Trailing"]`
- `ZONE_ORDER = ["Def", "Mid", "Off", "Wall"]`
- `TEAM_ORDER = ["Blue", "Orange"]`

## Common Tasks

### Adding a new analytics metric

**For app-specific metrics:**
1. Write a `calculate_<metric>()` function in `app.py` following existing patterns (takes `proto`, `df`, or analytics dict as input, returns computed data).
2. Call it from `_compute_match_analytics()` and add results to the returned dict.
3. Add UI rendering in the appropriate tab section.

**For reusable metrics (shot quality, defensive metrics, etc.):**
1. Add computation function to appropriate `analytics/` module (`save_metrics.py`, `shot_quality.py`, or create new module).
2. Define canonical column schema constants at module level.
3. Import and call from `_compute_match_analytics()` in `app.py`.
4. Add metric documentation to `docs/metrics/` if complex.
5. Add tests to `tests/` directory.

### Adding a new chart

1. **Determine if it's reusable** — If the chart pattern is used in 2+ locations, add a factory function to `charts/factory.py`. Otherwise, build inline in the tab.
2. **Use semantic theming:**
   ```python
   from charts.theme import apply_chart_theme, semantic_color
   fig = go.Figure()
   # ... add traces with semantic_color() ...
   fig = apply_chart_theme(fig, tier="support", intent="outcome", variant="win")
   ```
3. **Use formatters for tooltips/labels:**
   ```python
   from charts.formatters import format_metric_value, title_case_label
   hovertemplate = f"Metric: {title_case_label(metric_name)}: {format_metric_value(value, metric_name)}"
   ```
4. **Validate with linter** — Run `python scripts/lint_no_hex_in_app_charts.py` to ensure no hardcoded hex colors.

### Adding a new constant

Add it to `constants.py` and import where needed. Keep field geometry, colors, persistence paths, speed conversions, and categorical orderings centralized there.

### Adding a utility function

Add it to `utils.py` if it's a reusable player/team/time mapping, speed conversion, or sorting helper. Import in `app.py` or analytics modules.

### Adding a design token

Add it to `charts/tokens.py` using immutable patterns (frozen dataclasses or `MappingProxyType`). Update `charts/theme.py` `semantic_color()` if it's a new semantic intent.

### Updating metric documentation

Update `docs/metrics/` with canonical schemas, invariants, and usage examples. See `docs/metrics/save-metric.md` as a template.

## Testing & Validation

### Automated Tests

The project now has a formal test suite in the `tests/` directory:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_save_metrics.py -v
python -m pytest tests/test_speed_units.py -v
python -m pytest tests/test_canonical_ordering.py -v
```

**Test coverage:**
- `test_save_metrics.py` (86 lines) — Save metric contract, attribution logic, scoring invariants
- `test_speed_units.py` (79 lines) — Speed unit conversion, normalization, formatting
- `test_canonical_ordering.py` (33 lines) — Deterministic categorical ordering

### Code Quality Checks

```bash
# Syntax check all Python files
python -m py_compile app.py constants.py utils.py
python -m py_compile analytics/*.py
python -m py_compile charts/*.py

# Chart theming lint (no hardcoded hex colors)
python scripts/lint_no_hex_in_app_charts.py
```

### Manual Validation

- **Single match flow:** Upload test `.replay` file, verify all 11 tabs render correctly
- **Season batch flow:** Upload multiple replays, verify career tracking and session detection
- **Export functionality:** Test CSV/JSON/PNG export for both single match and season modes
- **Regression checks:** Compare chart outputs before/after changes for visual parity

## Documentation

### Project Documentation

- **`CLAUDE.md`** (this file) — Comprehensive project documentation, architecture overview, coding conventions
- **`docs/metrics/save-metric.md`** — Save metric contract specification (SDI, ExpectedSaveProb, SaveImpact, speed unit boundaries)
- **`docs/chart-migration-matrix.md`** — Chart migration planning matrix tracking conversion of 90+ charts to canonical theming system. Includes global invariants, interaction baselines, and execution protocol.

### Inline Documentation

- Analytics modules (`analytics/`) use module-level docstrings describing purpose and contract
- Chart factory functions include docstring descriptions of parameters and semantic intent
- Design tokens (`charts/tokens.py`) use frozen dataclasses for self-documenting constants

### Migration Tracking

The chart migration matrix (`docs/chart-migration-matrix.md`) is the **single source of truth** for chart refactoring:
- 63 rows covering single-match charts (interactive + export)
- 27 rows covering season-batch charts (interactive + export)
- Each row tracks: Chart ID, location, current type, analytic intent, target canonical type, required data, style tier, interaction rules, accessibility notes, priority/risk
- Status marked `[x]` only when visual parity confirmed, invariants pass, and regression checks pass

**Current migration status:**
- Completed: 9 charts (kickoff KPIs, session composite, spatial scatters, win probability base)
- In progress: Ongoing conversion to semantic theming and factory patterns
- Remaining: ~80 charts to migrate

## Git Conventions

- Commit messages are imperative, descriptive, and focus on what changed (e.g., "Extract save metrics to analytics module; add canonical schema").
- PRs are created from `claude/` prefixed branches and merged into `master`.
- `.gitignore` excludes `__pycache__/`, virtual environments, `.env`, `*.replay`, and `*.csv`.
- Use conventional commit prefixes where helpful: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`.
