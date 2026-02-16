from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from analytics.features import classify_pressure_context
from analytics.schema import (
    AnalyticsTables,
    EVENT_CONTRACT,
    EventType,
    POSSESSION_SEGMENT_CONTRACT,
    SCHEMA_VERSION,
    normalize_tables,
)
from constants import REPLAY_FPS
from utils import build_pid_name_map, build_pid_team_map


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


def _nearest_defender_distance(game_df: pd.DataFrame, proto, frame: int, shooter_team: str, ball_x: float, ball_y: float) -> float:
    defenders = [p.name for p in proto.players if ("Orange" if p.is_orange else "Blue") != shooter_team]
    min_dist = np.nan
    if frame not in game_df.index:
        return min_dist
    for d_name in defenders:
        if d_name not in game_df:
            continue
        d_data = game_df[d_name].loc[frame]
        dist = float(np.hypot(ball_x - float(d_data.get("pos_x", 0.0)), ball_y - float(d_data.get("pos_y", 0.0))))
        if np.isnan(min_dist) or dist < min_dist:
            min_dist = dist
    return min_dist


def _shooter_boost(game_df: pd.DataFrame, player_name: str, frame: int) -> float:
    if player_name not in game_df or frame not in game_df.index:
        return np.nan
    return float(game_df[player_name].loc[frame].get("boost", np.nan))


def _shot_angle(ball_x: float, ball_y: float, team: str) -> float:
    target_y = 5120.0 if team == "Blue" else -5120.0
    return float(np.arctan2(abs(ball_x), abs(target_y - ball_y)))


def _distance_to_goal(ball_x: float, ball_y: float, team: str) -> float:
    target_y = 5120.0 if team == "Blue" else -5120.0
    return float(np.hypot(ball_x, target_y - ball_y))


def _event_base(match_id: str, event_id: str, frame: int, event_type: EventType, player_id=None, player_name=None, team=None) -> Dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "match_id": str(match_id),
        "event_id": event_id,
        "frame": int(frame),
        "time_seconds": float(frame) / float(REPLAY_FPS),
        "event_type": event_type.serialize(),
        "player_id": player_id,
        "player_name": player_name,
        "team": team,
        "x": np.nan,
        "y": np.nan,
        "z": np.nan,
        "metric_value": np.nan,
        "is_key_play": False,
        "outcome_type": None,
        "is_on_target": False,
        "is_big_chance": False,
        "speed": np.nan,
        "pressure_context": "unknown",
        "nearest_defender_distance": np.nan,
        "shot_angle": np.nan,
        "shooter_boost": np.nan,
        "distance_to_goal": np.nan,
        "xg_pre": np.nan,
        "xg_post": np.nan,
        "xg_model_version": None,
        "xg_calibration_version": None,
        "xg": np.nan,
        "xa": np.nan,
        "segment_id": None,
        "chain_id": None,
        "touches_in_chain": np.nan,
        "chain_duration": np.nan,
        "avg_ball_speed": np.nan,
        "final_third_entries": np.nan,
        "turnovers_forced": np.nan,
    }


def _build_possession_segments(frame_state_df: pd.DataFrame, match_id: str) -> pd.DataFrame:
    poss = frame_state_df[["frame", "time_seconds", "possessing_player", "possessing_team"]].dropna(subset=["possessing_player"])
    segments: List[Dict] = []
    if poss.empty:
        return pd.DataFrame(columns=list(POSSESSION_SEGMENT_CONTRACT.columns.keys()))

    seg_start = poss.iloc[0]
    prev = poss.iloc[0]
    segment_idx = 0
    for _, row in poss.iloc[1:].iterrows():
        changed = int(row["frame"]) != int(prev["frame"]) + 1 or row["possessing_player"] != prev["possessing_player"]
        if changed:
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
    return pd.DataFrame(segments)


def _build_chain_context(
    match_id: str,
    frame_state_df: pd.DataFrame,
    event_df: pd.DataFrame,
    possession_segment_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, int], str], dict[str, dict[str, float]]]:
    if frame_state_df.empty or possession_segment_df.empty:
        return possession_segment_df.copy(), {}, {}

    seg = possession_segment_df.copy().sort_values("start_frame").reset_index(drop=True)
    seg["next_team"] = seg["team"].shift(-1)
    seg["is_turnover_next"] = (seg["team"].notna() & seg["next_team"].notna() & (seg["team"] != seg["next_team"]))
    chain_ids: list[str] = []
    chain_index = -1
    prior_team = None
    for row in seg.itertuples(index=False):
        if prior_team is None or row.team != prior_team:
            chain_index += 1
        chain_ids.append(f"chain-{chain_index}")
        prior_team = row.team
    seg["chain_id"] = chain_ids

    frame_to_chain: dict[int, str] = {}
    for row in seg.itertuples(index=False):
        for frame in range(int(row.start_frame), int(row.end_frame) + 1):
            frame_to_chain[int(frame)] = str(row.chain_id)

    touch_events = event_df[event_df["event_type"] == EventType.TOUCH.value].copy() if not event_df.empty else pd.DataFrame()
    touch_by_chain: dict[str, int] = {}
    if not touch_events.empty:
        for frame in pd.to_numeric(touch_events["frame"], errors="coerce").dropna().astype(int).tolist():
            chain_id = frame_to_chain.get(frame)
            if chain_id is not None:
                touch_by_chain[chain_id] = touch_by_chain.get(chain_id, 0) + 1

    frame_ctx = frame_state_df[["frame", "ball_vx", "ball_vy", "ball_vz", "ball_y"]].copy()
    frame_ctx["ball_speed"] = np.linalg.norm(frame_ctx[["ball_vx", "ball_vy", "ball_vz"]].fillna(0.0).values, axis=1)

    chain_summary: dict[str, dict[str, float]] = {}
    for chain_id, group in seg.groupby("chain_id", sort=False):
        start_frame = int(group["start_frame"].min())
        end_frame = int(group["end_frame"].max())
        team = group["team"].dropna().iloc[0] if not group["team"].dropna().empty else None
        window = frame_ctx[(frame_ctx["frame"] >= start_frame) & (frame_ctx["frame"] <= end_frame)].copy()
        avg_ball_speed = float(window["ball_speed"].mean()) if not window.empty else np.nan

        final_threshold = 1700.0
        if team == "Blue":
            in_final = window["ball_y"] >= final_threshold
        elif team == "Orange":
            in_final = window["ball_y"] <= -final_threshold
        else:
            in_final = pd.Series(False, index=window.index)
        entries = int((in_final.astype(int).diff().fillna(in_final.astype(int)) > 0).sum()) if not window.empty else 0

        turnovers_forced = int(group["is_turnover_next"].sum())
        chain_summary[chain_id] = {
            "touches_in_chain": float(touch_by_chain.get(chain_id, 0)),
            "chain_duration": float((end_frame - start_frame) / float(REPLAY_FPS)),
            "avg_ball_speed": avg_ball_speed,
            "final_third_entries": float(entries),
            "turnovers_forced": float(turnovers_forced),
        }

    for col in ["touches_in_chain", "chain_duration", "avg_ball_speed", "final_third_entries", "turnovers_forced"]:
        seg[col] = seg["chain_id"].map(lambda cid: chain_summary.get(cid, {}).get(col, np.nan))

    frame_to_segment: dict[tuple[str, int], str] = {}
    for row in seg.itertuples(index=False):
        for frame in range(int(row.start_frame), int(row.end_frame) + 1):
            frame_to_segment[(str(match_id), frame)] = str(row.segment_id)

    return seg.drop(columns=["next_team", "is_turnover_next"]), frame_to_segment, chain_summary


def _attach_chain_context_to_events(
    event_df: pd.DataFrame,
    match_id: str,
    frame_to_segment: dict[tuple[str, int], str],
    possession_segment_df: pd.DataFrame,
    chain_summary: dict[str, dict[str, float]],
) -> pd.DataFrame:
    if event_df.empty:
        return event_df

    out = event_df.copy()
    seg_to_chain = possession_segment_df.set_index("segment_id")["chain_id"].to_dict() if not possession_segment_df.empty else {}

    out["segment_id"] = out.apply(lambda r: frame_to_segment.get((str(r.get("match_id", match_id)), int(r.get("frame", -1)))), axis=1)
    out["chain_id"] = out["segment_id"].map(seg_to_chain)

    for col in ["touches_in_chain", "chain_duration", "avg_ball_speed", "final_third_entries", "turnovers_forced"]:
        out[col] = out["chain_id"].map(lambda cid: chain_summary.get(cid, {}).get(col, np.nan))

    return out




def _coerce_table(df: pd.DataFrame, contract) -> pd.DataFrame:
    if df is None or df.empty:
        return contract.empty()
    out = df.copy()
    for col, dtype in contract.columns.items():
        if col not in out.columns:
            out[col] = pd.NA
        try:
            out[col] = out[col].astype(dtype)
        except (TypeError, ValueError):
            pass
    return out[list(contract.columns.keys())]

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

    hits = [hit for hit in getattr(proto.game_stats, "hits", []) if hit.player_id]
    last_hit_by_team: Dict[str, tuple[str, int, str]] = {}
    for idx, hit in enumerate(hits):
        frame = int(hit.frame_number)
        state = _safe_ball_state(ball_df, frame)
        pid = str(hit.player_id.id)
        team = pid_team.get(pid)
        player_name = pid_name.get(pid)
        speed = float(np.linalg.norm([state["ball_vx"], state["ball_vy"], state["ball_vz"]])) if not np.isnan(state["ball_vx"]) else np.nan

        touch_row = _event_base(match_id, f"touch-{idx}", frame, EventType.TOUCH, pid, player_name, team)
        touch_row.update({"x": state["ball_x"], "y": state["ball_y"], "z": state["ball_z"], "speed": speed})
        event_rows.append(touch_row)

        # Pass completion from chained same-team touches.
        prior = last_hit_by_team.get(team)
        if prior is not None:
            prev_pid, prev_frame, prev_name = prior
            if prev_pid != pid and frame - prev_frame <= int(2.0 * REPLAY_FPS):
                pass_row = _event_base(match_id, f"pass-{prev_pid}-{pid}-{frame}", frame, EventType.PASS_COMPLETED, pid, player_name, team)
                pass_row.update({
                    "x": state["ball_x"],
                    "y": state["ball_y"],
                    "z": state["ball_z"],
                    "xa": np.nan,
                    "outcome_type": f"{prev_name}->{player_name}",
                })
                event_rows.append(pass_row)
        last_hit_by_team[team] = (pid, frame, player_name)

        # Challenge events from quick opponent exchanges.
        for opp_team, opp_info in last_hit_by_team.items():
            if opp_team == team or opp_info is None:
                continue
            opp_pid, opp_frame, opp_name = opp_info
            if frame - opp_frame <= int(1.5 * REPLAY_FPS):
                win_row = _event_base(match_id, f"challenge-win-{pid}-{frame}", frame, EventType.CHALLENGE_WIN, pid, player_name, team)
                loss_row = _event_base(match_id, f"challenge-loss-{opp_pid}-{frame}", frame, EventType.CHALLENGE_LOSS, opp_pid, opp_name, opp_team)
                win_row.update({"x": state["ball_x"], "y": state["ball_y"], "z": state["ball_z"]})
                loss_row.update({"x": state["ball_x"], "y": state["ball_y"], "z": state["ball_z"]})
                event_rows.extend([win_row, loss_row])
                break

        # Clear detection: touch from own defensive third moving out.
        if team in {"Blue", "Orange"} and not np.isnan(state["ball_y"]):
            in_def_third = state["ball_y"] < -1700 if team == "Blue" else state["ball_y"] > 1700
            exiting = state["ball_vy"] > 500 if team == "Blue" else state["ball_vy"] < -500
            if in_def_third and exiting:
                clear_row = _event_base(match_id, f"clear-{pid}-{frame}", frame, EventType.CLEAR, pid, player_name, team)
                clear_row.update({"x": state["ball_x"], "y": state["ball_y"], "z": state["ball_z"], "speed": speed})
                event_rows.append(clear_row)

    if not shot_df.empty:
        for idx, row in shot_df.iterrows():
            frame = int(row.get("Frame", 0))
            team = row.get("Team")
            player = row.get("Player")
            x = float(row.get("X", np.nan))
            y = float(row.get("Y", np.nan))
            z = float(_safe_ball_state(ball_df, frame).get("ball_z", np.nan))
            result = str(row.get("Result", "Shot"))
            speed = float(pd.to_numeric(row.get("Speed", np.nan), errors="coerce"))
            xg_pre = float(pd.to_numeric(row.get("xG_pre", row.get("xG", 0.0)), errors="coerce"))
            xg_post = float(pd.to_numeric(row.get("xG_post", row.get("xG", 0.0)), errors="coerce"))
            xg = xg_pre
            nearest_defender_distance = _nearest_defender_distance(game_df, proto, frame, team, x, y)
            shot_angle = _shot_angle(x, y, team) if team in {"Blue", "Orange"} else np.nan
            shooter_boost = _shooter_boost(game_df, player, frame)
            distance_to_goal = _distance_to_goal(x, y, team) if team in {"Blue", "Orange"} else np.nan
            pressure_context = classify_pressure_context(nearest_defender_distance)
            is_big_chance = bool(row.get("BigChance", False))
            shot_chain_id = row.get("ChainID")
            touches_in_chain = float(pd.to_numeric(row.get("TouchesInChain", np.nan), errors="coerce"))
            chain_duration = float(pd.to_numeric(row.get("ChainDuration", np.nan), errors="coerce"))
            avg_ball_speed = float(pd.to_numeric(row.get("AvgBallSpeed", np.nan), errors="coerce"))
            final_third_entries = float(pd.to_numeric(row.get("FinalThirdEntries", np.nan), errors="coerce"))
            turnovers_forced = float(pd.to_numeric(row.get("TurnoversForced", np.nan), errors="coerce"))

            shot_row = _event_base(match_id, f"shot-taken-{idx}", frame, EventType.SHOT_TAKEN, None, player, team)
            shot_row.update({
                "x": x,
                "y": y,
                "z": z,
                "metric_value": xg,
                "is_key_play": is_big_chance,
                "outcome_type": result.lower(),
                "is_on_target": True,
                "is_big_chance": is_big_chance,
                "speed": speed,
                "pressure_context": pressure_context,
                "nearest_defender_distance": nearest_defender_distance,
                "shot_angle": shot_angle,
                "shooter_boost": shooter_boost,
                "distance_to_goal": distance_to_goal,
                "xg_pre": xg_pre,
                "xg_post": xg_post,
                "xg_model_version": str(row.get("xGModelVersion", "legacy")),
                "xg_calibration_version": str(row.get("xGCalibrationVersion", "legacy")),
                "xg": xg,
                "chain_id": shot_chain_id,
                "touches_in_chain": touches_in_chain,
                "chain_duration": chain_duration,
                "avg_ball_speed": avg_ball_speed,
                "final_third_entries": final_third_entries,
                "turnovers_forced": turnovers_forced,
            })
            event_rows.append(shot_row)

            on_target_row = _event_base(match_id, f"shot-on-target-{idx}", frame, EventType.SHOT_ON_TARGET, None, player, team)
            on_target_row.update(shot_row)
            on_target_row["event_id"] = f"shot-on-target-{idx}"
            on_target_row["event_type"] = EventType.SHOT_ON_TARGET.value
            event_rows.append(on_target_row)

            if result == "Goal":
                goal_row = _event_base(match_id, f"goal-{idx}", frame, EventType.GOAL, None, player, team)
                goal_row.update(shot_row)
                goal_row["event_id"] = f"goal-{idx}"
                goal_row["event_type"] = EventType.GOAL.value
                event_rows.append(goal_row)
            else:
                save_row = _event_base(match_id, f"save-{idx}", frame, EventType.SAVE, None, player, team)
                save_row.update(shot_row)
                save_row["event_id"] = f"save-{idx}"
                save_row["event_type"] = EventType.SAVE.value
                event_rows.append(save_row)
                if pressure_context == "high":
                    block_row = _event_base(match_id, f"block-{idx}", frame, EventType.BLOCK, None, player, team)
                    block_row.update(shot_row)
                    block_row["event_id"] = f"block-{idx}"
                    block_row["event_type"] = EventType.BLOCK.value
                    event_rows.append(block_row)

    for p in proto.players:
        name, pid, team = p.name, str(p.id.id), ("Orange" if p.is_orange else "Blue")
        if name not in game_df or "boost" not in game_df[name].columns:
            continue
        boosts = game_df[name]["boost"].ffill().fillna(0)
        pickup_frames = boosts.diff().fillna(0)
        for frame in pickup_frames[pickup_frames > 12].index.tolist():
            prow = game_df[name].loc[frame]
            boost_row = _event_base(match_id, f"boost-{pid}-{int(frame)}", int(frame), EventType.BOOST_PICKUP, pid, name, team)
            boost_row.update({
                "x": float(prow.get("pos_x", np.nan)),
                "y": float(prow.get("pos_y", np.nan)),
                "z": float(prow.get("pos_z", np.nan)),
            })
            event_rows.append(boost_row)

    for idx, frame in enumerate(getattr(game, "kickoff_frames", [])):
        kickoff_row = _event_base(match_id, f"kickoff-{idx}", int(frame), EventType.KICKOFF)
        kickoff_row.update({"x": 0.0, "y": 0.0, "z": 0.0})
        event_rows.append(kickoff_row)

    event_df = pd.DataFrame(event_rows)

    base_segments = _build_possession_segments(frame_state_df, match_id=str(match_id))
    event_df = _coerce_table(event_df, EVENT_CONTRACT)
    base_segments = _coerce_table(base_segments, POSSESSION_SEGMENT_CONTRACT)

    enriched_segments, frame_to_segment, chain_summary = _build_chain_context(
        match_id=str(match_id),
        frame_state_df=frame_state_df,
        event_df=event_df,
        possession_segment_df=base_segments,
    )
    event_df = _attach_chain_context_to_events(
        event_df=event_df,
        match_id=str(match_id),
        frame_to_segment=frame_to_segment,
        possession_segment_df=enriched_segments,
        chain_summary=chain_summary,
    )

    return normalize_tables(AnalyticsTables(
        match=match_df,
        frame_state=frame_state_df,
        event=event_df,
        possession_segment=enriched_segments,
    ))


def event_table_to_shot_df(event_df: pd.DataFrame) -> pd.DataFrame:
    typed_columns = [
        "Player", "Team", "Frame", "Time", "Result", "OutcomeType", "OnTarget",
        "xG", "xG_pre", "xG_post", "XGModelVersion", "XGCalibrationVersion", "XA", "Speed", "PressureContext", "NearestDefenderDistance", "ShotAngle",
        "ShooterBoost", "DistanceToGoal", "BigChance", "X", "Y", "Z",
        "SegmentID", "ChainID", "TouchesInChain", "ChainDuration", "AvgBallSpeed", "FinalThirdEntries", "TurnoversForced",
    ]
    if event_df.empty:
        return pd.DataFrame(columns=typed_columns)

    subset = event_df[event_df["event_type"] == EventType.SHOT_TAKEN.value].copy()
    if subset.empty:
        return pd.DataFrame(columns=typed_columns)

    subset = subset.rename(columns={
        "player_name": "Player",
        "team": "Team",
        "frame": "Frame",
        "time_seconds": "Time",
        "outcome_type": "OutcomeType",
        "is_on_target": "OnTarget",
        "xg": "xG",
        "xg_pre": "xG_pre",
        "xg_post": "xG_post",
        "xg_model_version": "XGModelVersion",
        "xg_calibration_version": "XGCalibrationVersion",
        "xa": "XA",
        "speed": "Speed",
        "pressure_context": "PressureContext",
        "nearest_defender_distance": "NearestDefenderDistance",
        "shot_angle": "ShotAngle",
        "shooter_boost": "ShooterBoost",
        "distance_to_goal": "DistanceToGoal",
        "is_big_chance": "BigChance",
        "x": "X",
        "y": "Y",
        "z": "Z",
        "segment_id": "SegmentID",
        "chain_id": "ChainID",
        "touches_in_chain": "TouchesInChain",
        "chain_duration": "ChainDuration",
        "avg_ball_speed": "AvgBallSpeed",
        "final_third_entries": "FinalThirdEntries",
        "turnovers_forced": "TurnoversForced",
    })
    subset["Result"] = np.where(subset["OutcomeType"].str.lower() == "goal", "Goal", "Shot")

    for col in typed_columns:
        if col not in subset.columns:
            subset[col] = pd.NA
    return subset[typed_columns].sort_values("Frame")


def event_table_to_kickoff_df(event_df: pd.DataFrame, match_id: str = "") -> pd.DataFrame:
    cols = ["MatchID", "Frame", "Player", "Team", "BoostUsed", "Result", "Goal (5s)", "End_X", "End_Y"]
    if event_df.empty:
        return pd.DataFrame(columns=cols)
    kickoff = event_df[event_df["event_type"] == EventType.KICKOFF.value].copy()
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
