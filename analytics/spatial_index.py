"""Spatial indexing helpers for per-frame player proximity queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class _PlayerTrack:
    name: str
    team: str
    frames: np.ndarray
    x: np.ndarray
    y: np.ndarray


class PlayerFrameAccessor:
    """Precomputed player frame index for vectorized nearest-player lookups."""

    def __init__(self, tracks: list[_PlayerTrack]):
        self._names = np.array([track.name for track in tracks], dtype=object)
        self._teams = np.array([track.team for track in tracks], dtype=object)

        frame_chunks = [track.frames for track in tracks if track.frames.size > 0]
        self._frames = np.unique(np.concatenate(frame_chunks)) if frame_chunks else np.array([], dtype=int)

        player_count = len(tracks)
        frame_count = len(self._frames)
        self._x = np.full((player_count, frame_count), np.nan, dtype=float)
        self._y = np.full((player_count, frame_count), np.nan, dtype=float)

        if frame_count == 0:
            return

        for idx, track in enumerate(tracks):
            if track.frames.size == 0:
                continue
            frame_positions = np.searchsorted(self._frames, track.frames)
            self._x[idx, frame_positions] = track.x
            self._y[idx, frame_positions] = track.y

    @classmethod
    def from_player_positions(cls, player_pos: Mapping[str, Mapping]):
        tracks: list[_PlayerTrack] = []
        for name, info in player_pos.items():
            team = info.get("team")
            frames = np.asarray(info.get("frames", []), dtype=int)
            xs = np.asarray(info.get("x", []), dtype=float)
            ys = np.asarray(info.get("y", []), dtype=float)
            if not team or frames.size == 0 or xs.size == 0 or ys.size == 0:
                continue
            limit = min(len(frames), len(xs), len(ys))
            tracks.append(
                _PlayerTrack(
                    name=name,
                    team=str(team),
                    frames=frames[:limit],
                    x=xs[:limit],
                    y=ys[:limit],
                )
            )
        return cls(tracks)

    @classmethod
    def from_game_df(cls, proto, game_df):
        tracks: list[_PlayerTrack] = []
        for player in proto.players:
            name = player.name
            if name not in game_df:
                continue
            pdf = game_df[name]
            if "pos_x" not in pdf.columns or "pos_y" not in pdf.columns:
                continue
            frames = np.asarray(pdf.index.to_numpy(), dtype=int)
            tracks.append(
                _PlayerTrack(
                    name=name,
                    team="Orange" if player.is_orange else "Blue",
                    frames=frames,
                    x=np.asarray(pdf["pos_x"].to_numpy(), dtype=float),
                    y=np.asarray(pdf["pos_y"].to_numpy(), dtype=float),
                )
            )
        return cls(tracks)

    def nearest_defender(self, frame: int, team: str, x: float, y: float):
        """Return (name, distance) for nearest player on *team* at *frame*."""
        if self._frames.size == 0 or self._names.size == 0:
            return None, np.nan

        frame_idx = min(np.searchsorted(self._frames, int(frame)), len(self._frames) - 1)
        xs = self._x[:, frame_idx]
        ys = self._y[:, frame_idx]

        team_mask = self._teams == team
        valid_mask = team_mask & np.isfinite(xs) & np.isfinite(ys)
        if not np.any(valid_mask):
            return None, np.nan

        dist = np.hypot(xs[valid_mask] - float(x), ys[valid_mask] - float(y))
        nearest_idx = int(np.argmin(dist))
        nearest_name = self._names[valid_mask][nearest_idx]
        nearest_dist = float(dist[nearest_idx])
        return str(nearest_name), nearest_dist
