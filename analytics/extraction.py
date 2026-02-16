from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from constants import REPLAY_FPS
from utils import build_pid_name_map, build_pid_team_map
from analytics.schema import AnalyticsTables, SCHEMA_VERSION, normalize_tables


def _safe_ball_state(ball_df: pd.DataFrame, frame: int) -> Dict[str, float]:
    if ball_df.empty or frame not in ball_df.index:
        return {"ball_x": np.nan, "ball_y": np.nan, "ball_z": np.nan, "ball_vx": np.nan, "ball_vy": np.nan, "ball_vz": np.nan}
    row = ball_df.loc[frame]
    return {
        "ball_x": float(row.get("pos_x", np.nan)),
        "ball_y": float(row.get("pos_y", np.nan)),
        "ball_z": float(row.get("pos_z", np.nan)),
        "ball_vx": float(row.get("vel_x", np.nan)),
        "ball_vy": float(row.get("vel_y", np.nan)),
        "ball_vz": float(row.get("vel_z", np.nan)),
    }


def _build_score_by_frame(proto) -> Dict[int, tuple[int, int]]:
    team_by_pid = {str(p.id.id): ("Orange" if p.is_orange else "Blue") for p in proto.players}
    score_blue = 0
    score_orange = 0
    score_by_frame: Dict[int, tuple[int, int]] = {}
    goals = sorted(getattr(proto.game_metadata, "goals", []), key=lambda g: getattr(g, "frame_number", 0))
    for goal in goals:
        frame = int(getattr(goal, "frame_number", 0))
        pid = str(getattr(getattr(goal, "player_id", None), "id", ""))
        team = team_by_pid.get(pid, "Blue")
        if team == "Orange":
            score_orange += 1
        else:
            score_blue += 1
        score_by_frame[frame] = (score_blue, score_orange)
    return score_by_frame


def _infer_possessing_player(game_df: pd.DataFrame, proto, frame: int) -> tuple[str | None, str | None]:
    if "ball" not in game_df or frame not in game_df.index:
        return None, None
    ball_row = game_df["ball"].loc[frame]
    bx, by = float(ball_row.get("pos_x", 0.0)), float(ball_row.get("pos_y", 0.0))
    best_name, best_team, best_dist = None, None, float("inf")
    for p in proto.players:
        name = p.name
        if name not in game_df or frame not in game_df.index:
            continue
        prow = game_df[name].loc[frame]
        dist = float(np.hypot(float(prow.get("pos_x", 0.0)) - bx, float(prow.get("pos_y", 0.0)) - by))
        if dist < best_dist:
            best_dist = dist
            best_name = name
            best_team = "Orange" if p.is_orange else "Blue"
    return (best_name, best_team) if best_dist <= 900 else (None, None)


def build_schema_tables(manager, game_df: pd.DataFrame, proto, match_id: str, file_name: str, shot_df: pd.DataFrame) -> AnalyticsTables:
    game = manager.game
    ball_df = game_df["ball"] if "ball" in game_df else pd.DataFrame()
    max_frame = int(game_df.index.max()) if not game_df.empty else 0

    match_df = pd.DataFrame([{
        "schema_version": SCHEMA_VERSION,
        "match_id": str(match_id),
        "file_name": file_name,
        "map_name": str(getattr(game, "map", "")),
        "playlist": str(getattr(game, "playlist", "")),
        "num_frames": max_frame,
        "duration_seconds": round(max_frame / float(REPLAY_FPS), 2),
        "overtime_seconds": float(getattr(game, "overtime_seconds", 0.0) or 0.0),
    }])

    score_by_frame = _build_score_by_frame(proto)
    frame_rows: List[Dict] = []
    cur_blue, cur_orange = 0, 0
    for frame in range(0, max_frame + 1):
        if frame in score_by_frame:
            cur_blue, cur_orange = score_by_frame[frame]
        poss_player, poss_team = _infer_possessing_player(game_df, proto, frame)
        frame_rows.append({
            "schema_version": SCHEMA_VERSION,
            "match_id": str(match_id),
            "frame": frame,
            "time_seconds": frame / float(REPLAY_FPS),
            **_safe_ball_state(ball_df, frame),
            "score_blue": cur_blue,
            "score_orange": cur_orange,
            "possessing_player": poss_player,
            "possessing_team": poss_team,
        })
    frame_state_df = pd.DataFrame(frame_rows)

    pid_name = build_pid_name_map(proto)
    pid_team = build_pid_team_map(proto)
    event_rows: List[Dict] = []

    for idx, hit in enumerate(getattr(proto.game_stats, "hits", [])):
        if not hit.player_id:
            continue
        frame = int(hit.frame_number)
        state = _safe_ball_state(ball_df, frame)
        pid = str(hit.player_id.id)
        event_rows.append({
            "schema_version": SCHEMA_VERSION,
            "match_id": str(match_id),
            "event_id": f"touch-{idx}",
            "frame": frame,
            "time_seconds": frame / float(REPLAY_FPS),
            "event_type": "touch",
            "player_id": pid,
            "player_name": pid_name.get(pid),
            "team": pid_team.get(pid),
            "x": state["ball_x"],
            "y": state["ball_y"],
            "z": state["ball_z"],
            "result": None,
            "xg": np.nan,
            "speed": np.nan,
            "is_key_play": False,
        })

    if not shot_df.empty:
        for idx, row in shot_df.iterrows():
            result = str(row.get("Result", "Shot"))
            event_rows.append({
                "schema_version": SCHEMA_VERSION,
                "match_id": str(match_id),
                "event_id": f"shot-{idx}",
                "frame": int(row.get("Frame", 0)),
                "time_seconds": float(row.get("Frame", 0)) / float(REPLAY_FPS),
                "event_type": "shot",
                "player_id": None,
                "player_name": row.get("Player"),
                "team": row.get("Team"),
                "x": float(row.get("X", np.nan)),
                "y": float(row.get("Y", np.nan)),
                "z": np.nan,
                "result": result,
                "xg": float(row.get("xG", 0.0)),
                "speed": float(row.get("Speed", np.nan)),
                "is_key_play": bool(row.get("BigChance", False)),
            })

    for p in proto.players:
        name, pid, team = p.name, str(p.id.id), ("Orange" if p.is_orange else "Blue")
        if name not in game_df or "boost" not in game_df[name].columns:
            continue
        boosts = game_df[name]["boost"].ffill().fillna(0)
        pickup_frames = boosts.diff().fillna(0)
        for frame in pickup_frames[pickup_frames > 12].index.tolist():
            prow = game_df[name].loc[frame]
            event_rows.append({
                "schema_version": SCHEMA_VERSION,
                "match_id": str(match_id),
                "event_id": f"boost-{pid}-{int(frame)}",
                "frame": int(frame),
                "time_seconds": int(frame) / float(REPLAY_FPS),
                "event_type": "boost_pickup",
                "player_id": pid,
                "player_name": name,
                "team": team,
                "x": float(prow.get("pos_x", np.nan)),
                "y": float(prow.get("pos_y", np.nan)),
                "z": float(prow.get("pos_z", np.nan)),
                "result": None,
                "xg": np.nan,
                "speed": np.nan,
                "is_key_play": False,
            })

    for idx, frame in enumerate(getattr(game, "kickoff_frames", [])):
        event_rows.append({
            "schema_version": SCHEMA_VERSION,
            "match_id": str(match_id),
            "event_id": f"kickoff-{idx}",
            "frame": int(frame),
            "time_seconds": int(frame) / float(REPLAY_FPS),
            "event_type": "kickoff",
            "player_id": None,
            "player_name": None,
            "team": None,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "result": None,
            "xg": np.nan,
            "speed": np.nan,
            "is_key_play": False,
        })

    event_df = pd.DataFrame(event_rows)

    poss = frame_state_df[["frame", "time_seconds", "possessing_player", "possessing_team"]].dropna(subset=["possessing_player"])
    segments: List[Dict] = []
    if not poss.empty:
        seg_start = poss.iloc[0]
        prev = poss.iloc[0]
        segment_idx = 0
        for _, row in poss.iloc[1:].iterrows():
            if int(row["frame"]) != int(prev["frame"]) + 1 or row["possessing_player"] != prev["possessing_player"]:
                segments.append({
                    "schema_version": SCHEMA_VERSION,
                    "match_id": str(match_id),
                    "segment_id": f"seg-{segment_idx}",
                    "start_frame": int(seg_start["frame"]),
                    "end_frame": int(prev["frame"]),
                    "start_time": float(seg_start["time_seconds"]),
                    "end_time": float(prev["time_seconds"]),
                    "duration_seconds": float(prev["time_seconds"] - seg_start["time_seconds"]),
                    "player_name": seg_start["possessing_player"],
                    "team": seg_start["possessing_team"],
                })
                segment_idx += 1
                seg_start = row
            prev = row
        segments.append({
            "schema_version": SCHEMA_VERSION,
            "match_id": str(match_id),
            "segment_id": f"seg-{segment_idx}",
            "start_frame": int(seg_start["frame"]),
            "end_frame": int(prev["frame"]),
            "start_time": float(seg_start["time_seconds"]),
            "end_time": float(prev["time_seconds"]),
            "duration_seconds": float(prev["time_seconds"] - seg_start["time_seconds"]),
            "player_name": seg_start["possessing_player"],
            "team": seg_start["possessing_team"],
        })

    return normalize_tables(AnalyticsTables(
        match=match_df,
        frame_state=frame_state_df,
        event=event_df,
        possession_segment=pd.DataFrame(segments),
    ))


def event_table_to_shot_df(event_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Player", "Team", "Frame", "xG", "Result", "BigChance", "X", "Y", "Speed"]
    if event_df.empty:
        return pd.DataFrame(columns=cols)
    subset = event_df[event_df["event_type"] == "shot"].copy()
    if subset.empty:
        return pd.DataFrame(columns=cols)
    if "result" in subset.columns:
        subset["Result"] = subset["result"].fillna("Shot")
    else:
        subset["Result"] = np.where(subset.get("event_type", "shot") == "shot", "Shot", "Shot")
    if "xg" in subset.columns:
        subset["xG"] = pd.to_numeric(subset["xg"], errors="coerce").fillna(0.0)
    else:
        subset["xG"] = 0.0
    if "speed" in subset.columns:
        subset["Speed"] = pd.to_numeric(subset["speed"], errors="coerce").fillna(0.0)
    else:
        subset["Speed"] = 0.0
    subset["BigChance"] = subset["is_key_play"].fillna(False)
    subset = subset.rename(columns={"player_name": "Player", "team": "Team", "frame": "Frame", "x": "X", "y": "Y"})
    return subset[cols].sort_values("Frame")


def event_table_to_kickoff_df(event_df: pd.DataFrame, match_id: str = "") -> pd.DataFrame:
    cols = ["MatchID", "Frame", "Player", "Team", "BoostUsed", "Result", "Goal (5s)", "End_X", "End_Y"]
    if event_df.empty:
        return pd.DataFrame(columns=cols)
    kickoff = event_df[event_df["event_type"] == "kickoff"].copy()
    if kickoff.empty:
        return pd.DataFrame(columns=cols)
    kickoff = kickoff.rename(columns={"frame": "Frame"})
    kickoff["MatchID"] = str(match_id)
    kickoff["Player"] = "Unknown"
    kickoff["Team"] = "Unknown"
    kickoff["BoostUsed"] = 0
    kickoff["Result"] = "Unknown"
    kickoff["Goal (5s)"] = False
    kickoff["End_X"] = 0.0
    kickoff["End_Y"] = 0.0
    return kickoff[cols].sort_values("Frame")
