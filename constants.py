"""Rocket League Analytics — shared constants.

Single source of truth for field geometry, team colors, and app configuration.
"""

# ── Replay ──────────────────────────────────────────────────────────────
REPLAY_FPS = 30  # Standard replay frame rate

# ── Persistence ─────────────────────────────────────────────────────────
DB_FILE = "career_stats.csv"
KICKOFF_DB_FILE = "career_kickoffs.csv"
WIN_PROB_MODEL_FILE = "win_prob_model.json"
MAX_STORED_MATCHES = 12

# ── Field geometry (Unreal Units) ───────────────────────────────────────
FIELD_HALF_X = 4096    # Sideline (half-width)
FIELD_HALF_Y = 5120    # End line (half-length)
WALL_HEIGHT = 2044
GOAL_HALF_WIDTH = 1784  # Goal opening half-width
GOAL_WIDTH = GOAL_HALF_WIDTH * 2
GOAL_DEPTH = 880       # Goal depth behind end line
GOAL_HEIGHT = 642
CENTER_CIRCLE_R = 1140

# Backward-compatible alias (prefer GOAL_HALF_WIDTH for new code)
GOAL_HALF_W = GOAL_HALF_WIDTH

# Axis padding beyond field walls (room for goal depth + labels)
AXIS_PAD_X = 600
AXIS_PAD_Y = 1080

# ── Team colors ─────────────────────────────────────────────────────────
TEAM_COLORS = {
    "Blue": {
        "primary": "#007bff",
        "solid":   "rgba(0,123,255,1)",
        "trail":   "rgba(0,123,255,0.35)",
        "light":   "rgba(0,123,255,0.6)",
    },
    "Orange": {
        "primary": "#ff9900",
        "solid":   "rgba(255,153,0,1)",
        "trail":   "rgba(255,153,0,0.35)",
        "light":   "rgba(255,153,0,0.6)",
    },
}

# Convenience: Plotly color_discrete_map used in px.bar / px.scatter
TEAM_COLOR_MAP = {t: TEAM_COLORS[t]["primary"] for t in TEAM_COLORS}
