# CLAUDE.md — Rocket League Analytics

## Project Overview

Streamlit-based analytics platform for Rocket League replay analysis. Parses `.replay` files using the `carball` library, computes 20+ advanced metrics (xG, win probability, VAEP, momentum, etc.), and renders interactive Plotly visualizations. Supports single-match deep dives and season-long career tracking.

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
app.py              # Main application (~3,900 lines) — all UI, analytics, and rendering
constants.py        # Field geometry, team colors, persistence config, replay FPS
utils.py            # Player/team mapping helpers, frame/time conversion
requirements.txt    # Python dependencies
simple-pitch.png    # Pitch background image for 2D visualizations
launch_rocket_app.bat  # Windows launch script
career_heatmaps.json   # Cached heatmap data (generated at runtime)
```

## Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

Windows shortcut: `launch_rocket_app.bat` (calls `py -3.11 -m streamlit run app.py`)

## Architecture

### Module Responsibilities

- **`constants.py`** — Single source of truth for field geometry (Unreal Units), team colors (RGBA/hex), persistence file paths, and app configuration like `REPLAY_FPS = 30` and `MAX_STORED_MATCHES = 12`.
- **`utils.py`** — Seven reusable functions replacing 14+ duplicated player/team mapping patterns. Imports `REPLAY_FPS` from constants. Functions: `build_pid_team_map`, `build_pid_name_map`, `build_player_team_map`, `get_team_players`, `frame_to_seconds`, `seconds_to_frame`, `fmt_time`.
- **`app.py`** — Contains all analytics computation, UI rendering, data persistence, and export logic.

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

### Key Computation Functions in app.py

| Function | Purpose |
|----------|---------|
| `_compute_match_analytics()` | Central orchestrator — calls all analysis functions, returns unified dict |
| `calculate_kickoff_stats()` | Kickoff success/failure patterns |
| `calculate_shot_data()` | Shot tracking and mapping |
| `calculate_xg_probability()` | Expected Goals model |
| `calculate_win_probability()` | Match win probability curve |
| `calculate_contextual_momentum()` | Real-time momentum with contextual weighting |
| `calculate_advanced_passing()` | Pass accuracy, flow detection |
| `calculate_aerial_stats()` | Aerial mechanics and success rate |
| `calculate_defensive_pressure()` | Shadow defense metrics |
| `calculate_vaep()` | Value Added by Each Player |
| `calculate_rotation_analysis()` | Positional rotation patterns |
| `calculate_xs_probability()` | Expected Saves |
| `calculate_situational_stats()` | Clutch performance, game state splits |
| `calculate_luck_percentage()` | Poisson binomial luck metric |
| `calculate_recovery_time()` | Ground recovery tracking |

### Caching Strategy

```python
@st.cache_data      — get_field_layout() (field SVG)
@st.cache_resource  — get_3d_field_traces(), get_parsed_replay_data(), load_win_prob_model()
```

Session state holds up to 12 matches in `st.session_state.match_store` (dict keyed by filename), with `match_order` tracking insertion order and `active_match` pointing to the currently viewed match.

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

- **All analytics logic lives in `app.py`** as module-level functions. There are no classes for analytics — everything is functional.
- **Team identification** uses `"Blue"` / `"Orange"` string literals mapped via `is_orange` boolean from carball proto objects.
- **Field coordinates** are in Unreal Units (X = sideline ±4096, Y = end line ±5120, Z = height up to 2044).
- **Frame/time conversion** uses `REPLAY_FPS = 30` from constants. Always use `frame_to_seconds()` / `seconds_to_frame()` from utils.
- **Team colors** must come from `TEAM_COLORS` / `TEAM_COLOR_MAP` in constants — no hardcoded rgba strings.
- **Player ID keys** are always `str(p.id.id)` — string, not int.
- **Proto object** refers to the carball protobuf `AnalysisProto` returned by `get_parsed_replay_data()`.
- Section comments in `app.py` use the pattern `# --- N. SECTION NAME ---`.

## Common Tasks

### Adding a new analytics metric

1. Write a `calculate_<metric>()` function in `app.py` following the existing pattern (takes `proto`, `df`, or analytics dict as input, returns computed data).
2. Call it from `_compute_match_analytics()` and add results to the returned dict.
3. Add UI rendering in the appropriate tab section.

### Adding a new constant

Add it to `constants.py` and import it in `app.py`. Keep field geometry, colors, and persistence paths centralized there.

### Adding a utility function

Add it to `utils.py` if it's a reusable player/team/time mapping. Import in `app.py`.

## Validation

```bash
# Syntax check all Python files
python -m py_compile app.py
python -m py_compile constants.py
python -m py_compile utils.py
```

There is no formal test suite. Validation is done via syntax checking and manual testing through the Streamlit UI.

## Git Conventions

- Commit messages are imperative, descriptive, and focus on what changed (e.g., "Extract constants.py + utils.py; replace 23 inline map builders").
- PRs are created from `claude/` prefixed branches and merged into `master`.
- `.gitignore` excludes `__pycache__/`, virtual environments, `.env`, `*.replay`, and `*.csv`.
