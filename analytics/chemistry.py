"""Chemistry modeling for pair/trio contribution beyond individual baselines."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from analytics.stats_uncertainty import bootstrap_mean_interval, deterministic_seed, reliability_from_sample_size

CHEMISTRY_COMPONENT_COLUMNS = [
    "ExpectedValueGain",
    "RotationalComplementarity",
    "PossessionHandoffEfficiency",
    "PressureReleaseReliability",
]


@dataclass(frozen=True)
class ChemistryShrinkageConfig:
    prior_sample_size: float = 12.0
    confidence: float = 0.95
    bootstrap_iterations: int = 1000


def _get_col(df: pd.DataFrame, candidates: Sequence[str], default: str | None = None) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return default


def _normalize_presence_frames(frames_df: pd.DataFrame) -> pd.DataFrame:
    frame_col = _get_col(frames_df, ["Frame", "frame", "frame_number"])
    player_col = _get_col(frames_df, ["Player", "Name", "player", "player_name"])
    team_col = _get_col(frames_df, ["Team", "team"])
    rotation_col = _get_col(frames_df, ["RotationRole", "Zone", "role", "position_role"], default=None)
    possession_col = _get_col(frames_df, ["PossessionTeam", "possession_team"], default=None)
    pressure_col = _get_col(frames_df, ["UnderPressure", "IsPressured", "under_pressure"], default=None)
    if frame_col is None or player_col is None or team_col is None:
        raise ValueError("frames_df requires frame/player/team columns")

    norm = pd.DataFrame({
        "Frame": pd.to_numeric(frames_df[frame_col], errors="coerce"),
        "Player": frames_df[player_col].astype(str),
        "Team": frames_df[team_col].astype(str),
        "RotationRole": frames_df[rotation_col].astype(str) if rotation_col else "Unknown",
        "PossessionTeam": frames_df[possession_col].astype(str) if possession_col else frames_df[team_col].astype(str),
        "UnderPressure": pd.to_numeric(frames_df[pressure_col], errors="coerce").fillna(0).astype(int) if pressure_col else 0,
    })
    return norm.dropna(subset=["Frame", "Player", "Team"])


def _normalize_event_stream(events_df: pd.DataFrame) -> pd.DataFrame:
    frame_col = _get_col(events_df, ["Frame", "frame", "frame_number"])
    team_col = _get_col(events_df, ["Team", "team"], default=None)
    value_col = _get_col(events_df, ["ExpectedValue", "xGDelta", "EventValue", "value"], default=None)
    event_type_col = _get_col(events_df, ["EventType", "Type", "event_type"], default=None)
    success_col = _get_col(events_df, ["Success", "Successful", "is_success"], default=None)
    from_col = _get_col(events_df, ["FromPlayer", "Passer", "BallCarrier", "from_player"], default=None)
    to_col = _get_col(events_df, ["ToPlayer", "Receiver", "to_player"], default=None)
    pressure_release_col = _get_col(events_df, ["PressureRelease", "ReleaseSuccess", "pressure_release"], default=None)

    if frame_col is None:
        raise ValueError("events_df requires frame column")

    players = []
    if "Players" in events_df.columns:
        players = events_df["Players"].apply(lambda x: list(x) if isinstance(x, (list, tuple, set)) else [str(x)] if pd.notna(x) else [])
    elif "Player" in events_df.columns:
        players = events_df["Player"].astype(str).apply(lambda x: [x])
    else:
        players = pd.Series([[] for _ in range(len(events_df))])

    if from_col:
        players = [sorted(set((p if isinstance(p, list) else list(p)) + ([str(events_df.iloc[i][from_col])] if pd.notna(events_df.iloc[i][from_col]) else []) + ([str(events_df.iloc[i][to_col])] if to_col and pd.notna(events_df.iloc[i][to_col]) else []))) for i, p in enumerate(players)]

    return pd.DataFrame({
        "Frame": pd.to_numeric(events_df[frame_col], errors="coerce"),
        "Team": events_df[team_col].astype(str) if team_col else "Unknown",
        "EventValue": pd.to_numeric(events_df[value_col], errors="coerce").fillna(0.0) if value_col else 0.0,
        "EventType": events_df[event_type_col].astype(str) if event_type_col else "generic",
        "Success": pd.to_numeric(events_df[success_col], errors="coerce").fillna(0).astype(int) if success_col else 0,
        "FromPlayer": events_df[from_col].astype(str) if from_col else "",
        "ToPlayer": events_df[to_col].astype(str) if to_col else "",
        "PressureRelease": pd.to_numeric(events_df[pressure_release_col], errors="coerce").fillna(0).astype(int) if pressure_release_col else 0,
        "Players": players,
    }).dropna(subset=["Frame"])


def _shrink(value: float, n: int, global_mean: float, cfg: ChemistryShrinkageConfig) -> float:
    weight = n / (n + max(1e-9, cfg.prior_sample_size))
    return float((weight * value) + ((1.0 - weight) * global_mean))


def _ci(values: Iterable[float], seed_parts: Sequence[object], cfg: ChemistryShrinkageConfig) -> tuple[float, float, float]:
    return bootstrap_mean_interval(values, confidence=cfg.confidence, iterations=cfg.bootstrap_iterations, seed=deterministic_seed(*seed_parts))


def build_pairwise_feature_matrix(frames_df: pd.DataFrame, events_df: pd.DataFrame, *, config: ChemistryShrinkageConfig | None = None) -> pd.DataFrame:
    cfg = config or ChemistryShrinkageConfig()
    frames = _normalize_presence_frames(frames_df)
    events = _normalize_event_stream(events_df)

    frame_group = frames.groupby(["Frame", "Team"], sort=False)
    baseline_by_player = events.explode("Players").groupby("Players")["EventValue"].mean().to_dict()

    rows: list[dict[str, object]] = []
    for (frame, team), grp in frame_group:
        players = sorted(grp["Player"].unique())
        if len(players) < 2:
            continue
        ev_frame = events[(events["Frame"] == frame) & (events["Team"] == team)]
        for p1, p2 in combinations(players, 2):
            base = float(baseline_by_player.get(p1, 0.0) + baseline_by_player.get(p2, 0.0)) / 2.0
            joint_events = ev_frame[ev_frame["Players"].apply(lambda x: p1 in x and p2 in x)]
            event_gain = float(joint_events["EventValue"].mean()) - base if not joint_events.empty else -base

            roles = grp.set_index("Player")["RotationRole"].to_dict()
            role_comp = 1.0 if roles.get(p1) != roles.get(p2) else 0.2

            handoffs = ev_frame[(ev_frame["FromPlayer"] == p1) & (ev_frame["ToPlayer"] == p2) | (ev_frame["FromPlayer"] == p2) & (ev_frame["ToPlayer"] == p1)]
            handoff_eff = float(handoffs["Success"].mean()) if not handoffs.empty else 0.0

            pressured = int(grp["UnderPressure"].mean() > 0)
            pr_rel = float(ev_frame["PressureRelease"].mean()) if pressured and not ev_frame.empty else 0.0

            rows.append({
                "Team": team,
                "Player1": p1,
                "Player2": p2,
                "Frame": frame,
                "ExpectedValueGain": event_gain,
                "RotationalComplementarity": role_comp,
                "PossessionHandoffEfficiency": handoff_eff,
                "PressureReleaseReliability": pr_rel,
            })

    granular = pd.DataFrame(rows)
    if granular.empty:
        return pd.DataFrame(columns=["Team", "Player1", "Player2", "Samples", *CHEMISTRY_COMPONENT_COLUMNS, "ChemistryScore", "ChemistryScore_Shrunk", "CI_Low", "CI_High", "Reliability"])

    granular["ChemistryScore"] = granular[CHEMISTRY_COMPONENT_COLUMNS].mean(axis=1)
    global_means = {col: float(pd.to_numeric(granular[col], errors="coerce").fillna(0).mean()) for col in (CHEMISTRY_COMPONENT_COLUMNS + ["ChemistryScore"])}
    summary = granular.groupby(["Team", "Player1", "Player2"], as_index=False).agg(
        Samples=("Frame", "nunique"),
        ExpectedValueGain=("ExpectedValueGain", "mean"),
        RotationalComplementarity=("RotationalComplementarity", "mean"),
        PossessionHandoffEfficiency=("PossessionHandoffEfficiency", "mean"),
        PressureReleaseReliability=("PressureReleaseReliability", "mean"),
    )
    summary["ChemistryScore"] = summary[CHEMISTRY_COMPONENT_COLUMNS].mean(axis=1)

    for col in CHEMISTRY_COMPONENT_COLUMNS + ["ChemistryScore"]:
        gmean = global_means.get(col, float(summary[col].mean()))
        summary[f"{col}_Shrunk"] = [
            _shrink(float(v), int(n), gmean, cfg) for v, n in zip(summary[col], summary["Samples"])
        ]

    cis = []
    for _, row in summary.iterrows():
        key = (row["Team"], row["Player1"], row["Player2"])
        vals = granular[(granular["Team"] == row["Team"]) & (granular["Player1"] == row["Player1"]) & (granular["Player2"] == row["Player2"])]["ChemistryScore"]
        _, lo, hi = _ci(vals, key, cfg)
        cis.append((lo, hi))
    summary["CI_Low"] = [c[0] for c in cis]
    summary["CI_High"] = [c[1] for c in cis]
    summary["Reliability"] = summary["Samples"].map(lambda n: reliability_from_sample_size(int(n)))
    summary["ChemistryScore_Shrunk"] = summary["ChemistryScore_Shrunk"].astype(float)
    return summary.sort_values("ChemistryScore_Shrunk", ascending=False).reset_index(drop=True)


def build_trio_feature_matrix(frames_df: pd.DataFrame, events_df: pd.DataFrame, *, config: ChemistryShrinkageConfig | None = None) -> pd.DataFrame:
    cfg = config or ChemistryShrinkageConfig()
    frames = _normalize_presence_frames(frames_df)
    events = _normalize_event_stream(events_df)
    rows = []
    for (frame, team), grp in frames.groupby(["Frame", "Team"], sort=False):
        players = sorted(grp["Player"].unique())
        if len(players) < 3:
            continue
        ev_frame = events[(events["Frame"] == frame) & (events["Team"] == team)]
        for trio in combinations(players, 3):
            joint = ev_frame[ev_frame["Players"].apply(lambda p: all(x in p for x in trio))]
            val = float(joint["EventValue"].mean()) if not joint.empty else 0.0
            rows.append({"Team": team, "Player1": trio[0], "Player2": trio[1], "Player3": trio[2], "Frame": frame, "ChemistryScore": val})
    tri = pd.DataFrame(rows)
    if tri.empty:
        return pd.DataFrame(columns=["Team", "Player1", "Player2", "Player3", "Samples", "ChemistryScore", "ChemistryScore_Shrunk", "CI_Low", "CI_High", "Reliability"])
    global_mean = float(tri["ChemistryScore"].mean())
    out = tri.groupby(["Team", "Player1", "Player2", "Player3"], as_index=False).agg(Samples=("Frame", "nunique"), ChemistryScore=("ChemistryScore", "mean"))
    out["ChemistryScore_Shrunk"] = [_shrink(v, int(n), global_mean, cfg) for v, n in zip(out["ChemistryScore"], out["Samples"])]
    ci_rows = []
    for _, row in out.iterrows():
        vals = tri[(tri["Team"] == row["Team"]) & (tri["Player1"] == row["Player1"]) & (tri["Player2"] == row["Player2"]) & (tri["Player3"] == row["Player3"])]["ChemistryScore"]
        _, lo, hi = _ci(vals, (row["Team"], row["Player1"], row["Player2"], row["Player3"]), cfg)
        ci_rows.append((lo, hi))
    out["CI_Low"] = [x[0] for x in ci_rows]
    out["CI_High"] = [x[1] for x in ci_rows]
    out["Reliability"] = out["Samples"].map(lambda n: reliability_from_sample_size(int(n)))
    return out.sort_values("ChemistryScore_Shrunk", ascending=False).reset_index(drop=True)


def build_streams_from_match_stats(season_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create synchronized pseudo frame/event streams from per-match player records."""
    if season_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    base = season_df.copy()
    for col in ["MatchID", "Team", "Name"]:
        if col not in base.columns:
            raise ValueError(f"season_df missing required column: {col}")
    frame_rows = []
    event_rows = []
    for mid, match in base.groupby("MatchID", sort=False):
        for team, team_rows in match.groupby("Team", sort=False):
            frame_id = len(frame_rows) + 1
            avg_pos = float(pd.to_numeric(team_rows.get("Possession", 0), errors="coerce").fillna(0).mean())
            team_pressure = pd.to_numeric(team_rows.get("Pressure Time (s)", 0), errors="coerce").fillna(0).mean()
            for _, row in team_rows.iterrows():
                frame_rows.append({
                    "Frame": frame_id,
                    "Team": team,
                    "Player": row["Name"],
                    "RotationRole": "Defense" if float(row.get("Time_1st%", 0)) > 38 else "Support" if float(row.get("Time_2nd%", 0)) > 32 else "Attack",
                    "PossessionTeam": team if avg_pos >= 50 else "Opponent",
                    "UnderPressure": 1 if float(team_pressure) > 20 else 0,
                })
            for p1, p2 in combinations(team_rows["Name"].astype(str).tolist(), 2):
                pair_slice = team_rows[team_rows["Name"].isin([p1, p2])]
                event_rows.append({
                    "Frame": frame_id,
                    "Team": team,
                    "EventType": "handoff",
                    "FromPlayer": p1,
                    "ToPlayer": p2,
                    "Success": int(pair_slice.get("Assists", pd.Series([0, 0])).sum() > 0),
                    "PressureRelease": int(pair_slice.get("Avg Recovery (s)", pd.Series([0, 0])).mean() < 1.4),
                    "ExpectedValue": float(pd.to_numeric(pair_slice.get("xG", 0), errors="coerce").fillna(0).sum() - pd.to_numeric(pair_slice.get("xGA", 0), errors="coerce").fillna(0).sum()),
                    "Players": [p1, p2],
                })
    return pd.DataFrame(frame_rows), pd.DataFrame(event_rows)


def build_season_chemistry_tables(season_df: pd.DataFrame, *, config: ChemistryShrinkageConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames_df, events_df = build_streams_from_match_stats(season_df)
    return build_pairwise_feature_matrix(frames_df, events_df, config=config), build_trio_feature_matrix(frames_df, events_df, config=config)
