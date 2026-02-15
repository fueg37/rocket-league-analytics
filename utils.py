"""Rocket League Analytics — shared utility functions.

Eliminates the 14+ duplicated player/team mapping patterns scattered across app.py.
"""
import numpy as np

from constants import REPLAY_FPS


# ── Player / Team Mappings ──────────────────────────────────────────────

def build_pid_team_map(proto):
    """Player ID (str) → team name.

    Replaces: {str(p.id.id): "Orange" if p.is_orange else "Blue" for p in proto.players}
    """
    return {str(p.id.id): ("Orange" if p.is_orange else "Blue") for p in proto.players}


def build_pid_name_map(proto):
    """Player ID (str) → display name.

    Replaces: {str(p.id.id): p.name for p in proto.players}
    """
    return {str(p.id.id): p.name for p in proto.players}


def build_player_team_map(proto):
    """Player display name → team name.

    Replaces: {p.name: "Orange" if p.is_orange else "Blue" for p in proto.players}
    """
    return {p.name: ("Orange" if p.is_orange else "Blue") for p in proto.players}


def get_team_players(proto, team):
    """Return list of player names belonging to *team* ("Blue" or "Orange")."""
    return [p.name for p in proto.players
            if ("Orange" if p.is_orange else "Blue") == team]


def build_player_positions(proto, game_df):
    """Pre-compute per-player position numpy arrays for fast frame lookups.

    Returns {name: {'team': str, 'x': ndarray, 'y': ndarray, 'frames': ndarray}}
    for every player whose position data is available in game_df.
    """
    positions = {}
    for p in proto.players:
        if p.name in game_df:
            pdf = game_df[p.name]
            if 'pos_x' in pdf.columns:
                positions[p.name] = {
                    'team': "Orange" if p.is_orange else "Blue",
                    'x': pdf['pos_x'].values,
                    'y': pdf['pos_y'].values,
                    'frames': pdf.index.values,
                }
    return positions


# ── Frame / Time Helpers ────────────────────────────────────────────────

def frame_to_seconds(frame, fps=REPLAY_FPS):
    """Convert a frame index to seconds."""
    return frame / float(fps)


def seconds_to_frame(seconds, fps=REPLAY_FPS):
    """Convert seconds to the nearest frame index."""
    return int(seconds * fps)


def fmt_time(seconds):
    """Format a time in seconds as *M:SS*."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"
