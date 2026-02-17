"""Chemistry modeling for pair/trio contribution beyond individual baselines."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import pandas as pd

from analytics.partnership_contracts import apply_partnership_contract
from analytics.stats_uncertainty import bootstrap_mean_interval, deterministic_seed, reliability_from_sample_size

CHEMISTRY_COMPONENT_COLUMNS = [
    "ExpectedValueGain",
    "RotationalComplementarity",
    "PossessionHandoffEfficiency",
    "PressureReleaseReliability",
]

COMPONENT_LABELS = {
    "ExpectedValueGain": "Chance Creation",
    "RotationalComplementarity": "Rotation Balance",
    "PossessionHandoffEfficiency": "Possession Linking",
    "PressureReleaseReliability": "Pressure Release",
}

CONTEXT_LABELS = {
    "context_score_leading": "Leading game states",
    "context_score_tied": "Tied game states",
    "context_score_trailing": "Trailing game states",
    "context_score_defensive_third": "Defensive third",
    "context_score_offensive_third": "Offensive third",
    "context_score_high_pressure": "High-pressure phases",
}

CONTEXT_PRIORITY = list(CONTEXT_LABELS.keys())

CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}

LEGACY_ALIAS_MAP = {
    "Partnership Index": "ChemistryScore_Shrunk",
    "confidence_level": "Reliability",
    "sample_count": "Samples",
    "ci_low": "CI_Low",
    "ci_high": "CI_High",
}


@dataclass(frozen=True)
class ExplanationThresholds:
    strong_sample_count: int = 14
    medium_sample_count: int = 8
    tight_ci_width: float = 10.0
    wide_ci_width: float = 22.0


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
        "ScoreDelta": pd.to_numeric(frames_df[_get_col(frames_df, ["ScoreDelta", "score_delta", "GoalDiff", "goal_diff"], default=None)], errors="coerce").fillna(0.0) if _get_col(frames_df, ["ScoreDelta", "score_delta", "GoalDiff", "goal_diff"], default=None) else 0.0,
        "FieldThird": frames_df[_get_col(frames_df, ["FieldThird", "PitchThird", "ZoneThird", "field_third"], default=None)].astype(str) if _get_col(frames_df, ["FieldThird", "PitchThird", "ZoneThird", "field_third"], default=None) else frames_df[rotation_col].astype(str) if rotation_col else "Unknown",
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


def _classify_confidence_tier(row: pd.Series, thresholds: ExplanationThresholds) -> str:
    sample_count = int(pd.to_numeric(row.get("sample_count", row.get("Samples", 0)), errors="coerce") or 0)
    ci_low = float(pd.to_numeric(row.get("ci_low", row.get("CI_Low", 0.0)), errors="coerce") or 0.0)
    ci_high = float(pd.to_numeric(row.get("ci_high", row.get("CI_High", 0.0)), errors="coerce") or 0.0)
    ci_width = max(0.0, ci_high - ci_low)
    if sample_count >= thresholds.strong_sample_count and ci_width <= thresholds.tight_ci_width:
        return "high"
    if sample_count >= thresholds.medium_sample_count and ci_width <= thresholds.wide_ci_width:
        return "medium"
    return "low"


def _context_mean(granular: pd.DataFrame, group_keys: Sequence[str], flag_col: str) -> pd.Series:
    selected = granular[granular[flag_col] > 0]
    if selected.empty:
        return pd.Series(dtype=float)
    return selected.groupby(list(group_keys))["ChemistryScore"].mean()


def _build_primary_secondary_driver_labels(summary: pd.DataFrame) -> pd.DataFrame:
    pct_cols = [f"{col}_ContributionPct" for col in CHEMISTRY_COMPONENT_COLUMNS]
    labels = []
    secondary_labels = []
    for _, row in summary.iterrows():
        sorted_cols = sorted(
            pct_cols,
            key=lambda c: float(pd.to_numeric(row.get(c, 0.0), errors="coerce") or 0.0),
            reverse=True,
        )
        primary = sorted_cols[0].replace("_ContributionPct", "") if sorted_cols else CHEMISTRY_COMPONENT_COLUMNS[0]
        secondary = sorted_cols[1].replace("_ContributionPct", "") if len(sorted_cols) > 1 else primary
        labels.append(COMPONENT_LABELS.get(primary, primary))
        secondary_labels.append(COMPONENT_LABELS.get(secondary, secondary))
    summary["primary_driver_label"] = labels
    summary["secondary_driver_label"] = secondary_labels
    return summary


def _build_context_driver_labels(summary: pd.DataFrame) -> pd.DataFrame:
    best_labels = []
    risk_labels = []
    for _, row in summary.iterrows():
        context_scores = {k: float(pd.to_numeric(row.get(k, 0.0), errors="coerce") or 0.0) for k in CONTEXT_PRIORITY}
        best_key = max(CONTEXT_PRIORITY, key=lambda k: context_scores[k])
        risk_key = min(CONTEXT_PRIORITY, key=lambda k: context_scores[k])
        best_labels.append(CONTEXT_LABELS[best_key])
        risk_labels.append(CONTEXT_LABELS[risk_key])
    summary["best_context_label"] = best_labels
    summary["risk_context_label"] = risk_labels
    return summary


def _driver_phrase(label: str) -> str:
    phrases = {
        "Pressure Release": "absorbing pressure and exiting cleanly",
        "Chance Creation": "turning possession into chance creation",
        "Rotation Balance": "staying connected through rotation cycles",
        "Possession Linking": "sustaining possession through clean handoffs",
    }
    return phrases.get(label, "building stable two-player sequences")


def _certainty_prefix(tier: str) -> str:
    if tier == "high":
        return "Strongest when"
    if tier == "medium":
        return "Often strongest when"
    return "Shows signs of being strongest when"


def _certainty_verb(tier: str) -> str:
    if tier == "high":
        return "Best used in"
    if tier == "medium":
        return "Often useful in"
    return "May be best used in"


def _context_badge(label: str, kind: str) -> str:
    if kind == "best":
        return f"🟢 Best context: {label}"
    return f"🟠 Risk context: {label}"


def _passes_confidence_gate(row: pd.Series, minimum_tier: str = "medium") -> bool:
    current = str(row.get("confidence_level", row.get("Reliability", "low"))).strip().lower()
    return CONFIDENCE_RANK.get(current, 0) >= CONFIDENCE_RANK.get(minimum_tier.lower(), 1)


def apply_chemistry_compatibility_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Keep legacy chemistry naming available while Partnership Intelligence ships."""
    if df.empty:
        return df.copy()
    out = df.copy()

    for new_col, legacy_col in LEGACY_ALIAS_MAP.items():
        if legacy_col not in out.columns and new_col in out.columns:
            out[legacy_col] = out[new_col]
        if new_col not in out.columns and legacy_col in out.columns:
            out[new_col] = out[legacy_col]

    if "ChemistryScore_Shrunk" not in out.columns and "Partnership Index" in out.columns:
        out["ChemistryScore_Shrunk"] = out["Partnership Index"]
    if "Partnership Index" not in out.columns and "ChemistryScore_Shrunk" in out.columns:
        out["Partnership Index"] = out["ChemistryScore_Shrunk"]

    return out


def add_chemistry_explanations(df: pd.DataFrame, *, thresholds: ExplanationThresholds | None = None) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    t = thresholds or ExplanationThresholds()
    out = df.copy()
    primary_explanations = []
    context_explanations = []
    best_context_badges = []
    risk_context_badges = []
    for _, row in out.iterrows():
        tier = _classify_confidence_tier(row, t)
        primary = str(row.get("primary_driver_label", "Chemistry"))
        best_context_label = str(row.get("best_context_label", "neutral contexts"))
        risk_context_label = str(row.get("risk_context_label", "neutral contexts"))
        best_context = best_context_label.lower()
        driver_phrase = _driver_phrase(primary)
        primary_explanations.append(f"{_certainty_prefix(tier)} {driver_phrase}.")
        if _passes_confidence_gate(row, minimum_tier="medium"):
            context_explanations.append(f"{_certainty_verb(tier)} {best_context} for {primary.lower()}.")
            best_context_badges.append(_context_badge(best_context_label, "best"))
            risk_context_badges.append(_context_badge(risk_context_label, "risk"))
        else:
            context_explanations.append("Context usage signal is still stabilizing; collect more shared samples.")
            best_context_badges.append("⚪ Context signal pending")
            risk_context_badges.append("⚪ Context signal pending")
    out["primary_driver_explanation"] = primary_explanations
    out["context_usage_explanation"] = context_explanations
    out["best_context_badge"] = best_context_badges
    out["risk_context_badge"] = risk_context_badges
    return out


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
        score_delta = float(pd.to_numeric(grp.get("ScoreDelta", 0.0), errors="coerce").fillna(0.0).mean())
        state_leading = int(score_delta > 0)
        state_tied = int(score_delta == 0)
        state_trailing = int(score_delta < 0)

        field_values = grp.get("FieldThird", pd.Series(["Unknown"] * len(grp), index=grp.index)).astype(str).str.lower()
        state_def_third = int(field_values.str.contains("def|back").mean() >= 0.5)
        state_off_third = int(field_values.str.contains("off|att|front").mean() >= 0.5)
        high_pressure = int(pd.to_numeric(grp["UnderPressure"], errors="coerce").fillna(0).mean() > 0)

        for p1, p2 in combinations(players, 2):
            base = float(baseline_by_player.get(p1, 0.0) + baseline_by_player.get(p2, 0.0)) / 2.0
            joint_events = ev_frame[ev_frame["Players"].apply(lambda x: p1 in x and p2 in x)]
            event_gain = float(joint_events["EventValue"].mean()) - base if not joint_events.empty else -base

            roles = grp.set_index("Player")["RotationRole"].to_dict()
            role_comp = 1.0 if roles.get(p1) != roles.get(p2) else 0.2

            handoffs = ev_frame[((ev_frame["FromPlayer"] == p1) & (ev_frame["ToPlayer"] == p2)) | ((ev_frame["FromPlayer"] == p2) & (ev_frame["ToPlayer"] == p1))]
            handoff_eff = float(handoffs["Success"].mean()) if not handoffs.empty else 0.0

            pr_rel = float(ev_frame["PressureRelease"].mean()) if high_pressure and not ev_frame.empty else 0.0

            rows.append({
                "Team": team,
                "Player1": p1,
                "Player2": p2,
                "Frame": frame,
                "ExpectedValueGain": event_gain,
                "RotationalComplementarity": role_comp,
                "PossessionHandoffEfficiency": handoff_eff,
                "PressureReleaseReliability": pr_rel,
                "state_leading": state_leading,
                "state_tied": state_tied,
                "state_trailing": state_trailing,
                "state_defensive_third": state_def_third,
                "state_offensive_third": state_off_third,
                "state_high_pressure": high_pressure,
            })

    granular = pd.DataFrame(rows)
    if granular.empty:
        return pd.DataFrame(columns=[
            "Team", "Player1", "Player2", "Samples", *CHEMISTRY_COMPONENT_COLUMNS,
            "ChemistryScore", "ChemistryScore_Shrunk", "CI_Low", "CI_High", "Reliability",
            "Partnership Index", "Value Lift", "Rotation Fit", "Handoff Quality", "Pressure Escape",
            "confidence_level", "ci_low", "ci_high", "sample_count", "expected_xgd_lift_per_match",
            "win_rate_lift_points", "PartnershipIndex", "ConfidenceLevel", "SampleCount", "CI_Low_Index", "CI_High_Index",
            *[f"{col}_ContributionPct" for col in CHEMISTRY_COMPONENT_COLUMNS],
            *CONTEXT_PRIORITY,
            "primary_driver_label", "secondary_driver_label", "best_context_label", "risk_context_label",
            "primary_driver_explanation", "context_usage_explanation",
        ])

    granular["ChemistryScore"] = granular[CHEMISTRY_COMPONENT_COLUMNS].mean(axis=1)
    global_means = {col: float(pd.to_numeric(granular[col], errors="coerce").fillna(0).mean()) for col in (CHEMISTRY_COMPONENT_COLUMNS + ["ChemistryScore"])}
    group_keys = ["Team", "Player1", "Player2"]
    summary = granular.groupby(group_keys, as_index=False).agg(
        Samples=("Frame", "nunique"),
        ExpectedValueGain=("ExpectedValueGain", "mean"),
        RotationalComplementarity=("RotationalComplementarity", "mean"),
        PossessionHandoffEfficiency=("PossessionHandoffEfficiency", "mean"),
        PressureReleaseReliability=("PressureReleaseReliability", "mean"),
    )
    summary["ChemistryScore"] = summary[CHEMISTRY_COMPONENT_COLUMNS].mean(axis=1)

    for col in CHEMISTRY_COMPONENT_COLUMNS + ["ChemistryScore"]:
        gmean = global_means.get(col, float(summary[col].mean()))
        summary[f"{col}_Shrunk"] = [_shrink(float(v), int(n), gmean, cfg) for v, n in zip(summary[col], summary["Samples"])]

    for context_col, state_col in {
        "context_score_leading": "state_leading",
        "context_score_tied": "state_tied",
        "context_score_trailing": "state_trailing",
        "context_score_defensive_third": "state_defensive_third",
        "context_score_offensive_third": "state_offensive_third",
        "context_score_high_pressure": "state_high_pressure",
    }.items():
        context_series = _context_mean(granular, group_keys, state_col)
        if context_series.empty:
            summary[context_col] = summary["ChemistryScore"]
        else:
            context_df = context_series.rename(context_col).reset_index()
            summary = summary.merge(context_df, on=group_keys, how="left")
            summary[context_col] = summary[context_col].fillna(summary["ChemistryScore"])

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

    shrunk_component_cols = [f"{col}_Shrunk" for col in CHEMISTRY_COMPONENT_COLUMNS]
    positive_mass = summary[shrunk_component_cols].clip(lower=0.0).sum(axis=1)
    for raw_col, shrunk_col in zip(CHEMISTRY_COMPONENT_COLUMNS, shrunk_component_cols):
        pct_col = f"{raw_col}_ContributionPct"
        summary[pct_col] = (
            summary[shrunk_col].clip(lower=0.0) / positive_mass.where(positive_mass > 0, 1.0) * 100.0
        ).fillna(25.0)

    summary = apply_partnership_contract(summary)
    summary = _build_primary_secondary_driver_labels(summary)
    summary = _build_context_driver_labels(summary)
    summary = add_chemistry_explanations(summary)

    summary = apply_chemistry_compatibility_aliases(summary)
    summary["PartnershipIndex"] = summary["Partnership Index"]
    summary["ConfidenceLevel"] = summary["confidence_level"]
    summary["SampleCount"] = summary["sample_count"]
    summary["CI_Low_Index"] = summary["ci_low"]
    summary["CI_High_Index"] = summary["ci_high"]

    return summary.sort_values("Partnership Index", ascending=False).reset_index(drop=True)


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
