"""Director Mode orchestration for timeline-first single-match analysis."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class DirectorEvent:
    event_id: str
    source: str
    event_type: str
    time: float
    team: str
    impact_score: float
    confidence: float
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_time_series(df: pd.DataFrame, *, frame_col: str = "Frame", time_col: str = "Time", fps: float = 30.0) -> pd.Series:
    if time_col in df.columns:
        return pd.to_numeric(df[time_col], errors="coerce").fillna(0.0)
    if frame_col in df.columns:
        return pd.to_numeric(df[frame_col], errors="coerce").fillna(0.0) / float(fps)
    return pd.Series([0.0] * len(df), index=df.index, dtype=float)


def _team_col(df: pd.DataFrame) -> pd.Series:
    for col in ("Team", "team"):
        if col in df.columns:
            return df[col].fillna("Neutral").astype(str)
    return pd.Series(["Neutral"] * len(df), index=df.index, dtype=object)


def _build_event_rows(win_prob_df: pd.DataFrame, shot_df: pd.DataFrame, kickoff_df: pd.DataFrame, vaep_df: pd.DataFrame, save_events_df: pd.DataFrame) -> list[DirectorEvent]:
    events: list[DirectorEvent] = []

    if win_prob_df is not None and not win_prob_df.empty and "WinProb" in win_prob_df.columns:
        wp = win_prob_df.copy()
        wp["Time"] = _safe_time_series(wp)
        wp["WinProb"] = pd.to_numeric(wp["WinProb"], errors="coerce").fillna(50.0)
        wp["delta"] = wp["WinProb"].diff().abs().fillna(0.0)
        for i, row in wp.nlargest(8, "delta").iterrows():
            impact = float(row["delta"]) / 100.0
            events.append(DirectorEvent(
                event_id=f"wp_{int(i)}",
                source="win_prob_df",
                event_type="win_probability_swing",
                time=float(row["Time"]),
                team="Blue" if float(row["WinProb"]) >= 50.0 else "Orange",
                impact_score=impact,
                confidence=min(0.98, 0.45 + impact),
                summary=f"Win probability swing of {float(row['delta']):.1f} pts",
            ))

    if shot_df is not None and not shot_df.empty:
        shots = shot_df.copy()
        shots["Time"] = _safe_time_series(shots)
        shots["xG"] = pd.to_numeric(shots.get("xG", 0.0), errors="coerce").fillna(0.0)
        shots["Team"] = _team_col(shots)
        for i, row in shots.nlargest(10, "xG").iterrows():
            impact = float(row["xG"])
            events.append(DirectorEvent(
                event_id=f"shot_{int(i)}",
                source="shot_df",
                event_type="shot_chance",
                time=float(row["Time"]),
                team=str(row["Team"]),
                impact_score=impact,
                confidence=min(0.95, 0.55 + impact),
                summary=f"{str(row.get('Result', 'Shot'))} chance xG={impact:.2f}",
            ))

    if kickoff_df is not None and not kickoff_df.empty:
        kos = kickoff_df.copy()
        kos["Time"] = _safe_time_series(kos)
        kos["Team"] = _team_col(kos)
        for i, row in kos.iterrows():
            result = str(row.get("Result", "Neutral"))
            impact = 0.22 if result.lower() == "win" else 0.12
            events.append(DirectorEvent(
                event_id=f"ko_{int(i)}",
                source="kickoff_df",
                event_type="kickoff",
                time=float(row["Time"]),
                team=str(row["Team"]),
                impact_score=impact,
                confidence=0.65,
                summary=f"Kickoff {result}",
            ))

    if vaep_df is not None and not vaep_df.empty and "VAEP" in vaep_df.columns:
        vaep = vaep_df.copy()
        vaep["Time"] = _safe_time_series(vaep)
        vaep["VAEP"] = pd.to_numeric(vaep["VAEP"], errors="coerce").fillna(0.0)
        vaep["Team"] = _team_col(vaep)
        for i, row in vaep.reindex(vaep["VAEP"].abs().nlargest(10).index).iterrows():
            impact = abs(float(row["VAEP"]))
            events.append(DirectorEvent(
                event_id=f"vaep_{int(i)}",
                source="vaep_df",
                event_type="value_swing",
                time=float(row["Time"]),
                team=str(row["Team"]),
                impact_score=impact,
                confidence=min(0.95, 0.5 + impact),
                summary=f"VAEP swing {float(row['VAEP']):+.3f}",
            ))

    if save_events_df is not None and not save_events_df.empty:
        saves = save_events_df.copy()
        saves["Time"] = _safe_time_series(saves)
        saves["SaveImpact"] = pd.to_numeric(saves.get("SaveImpact", 0.0), errors="coerce").fillna(0.0)
        saves["Team"] = _team_col(saves)
        for i, row in saves.reindex(saves["SaveImpact"].abs().nlargest(6).index).iterrows():
            impact = abs(float(row["SaveImpact"]))
            events.append(DirectorEvent(
                event_id=f"save_{int(i)}",
                source="save_events_df",
                event_type="save",
                time=float(row["Time"]),
                team=str(row["Team"]),
                impact_score=impact,
                confidence=min(0.92, 0.52 + impact),
                summary=f"Save impact {float(row['SaveImpact']):+.2f}",
            ))

    return events


def build_director_event_queue(
    *,
    win_prob_df: pd.DataFrame,
    shot_df: pd.DataFrame,
    kickoff_df: pd.DataFrame,
    vaep_df: pd.DataFrame,
    save_events_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a deterministic ranked queue of swing moments for Director Mode."""
    events = _build_event_rows(win_prob_df, shot_df, kickoff_df, vaep_df, save_events_df)
    if not events:
        return pd.DataFrame(columns=["event_id", "source", "event_type", "time", "team", "impact_score", "confidence", "summary", "rank_score"])

    queue = pd.DataFrame([e.to_dict() for e in events])
    queue["rank_score"] = pd.to_numeric(queue["impact_score"], errors="coerce").fillna(0.0) * 0.7 + pd.to_numeric(queue["confidence"], errors="coerce").fillna(0.0) * 0.3
    queue = queue.sort_values(["rank_score", "time", "event_id"], ascending=[False, True, True], kind="mergesort").reset_index(drop=True)
    return queue



def synchronize_track_times(*, win_prob_df: pd.DataFrame, shot_df: pd.DataFrame, vaep_df: pd.DataFrame, momentum_series: pd.Series) -> pd.DataFrame:
    """Return a canonical time-indexed frame used by all track renderers."""
    times = pd.Series(dtype=float)
    if win_prob_df is not None and not win_prob_df.empty and "Time" in win_prob_df.columns:
        times = pd.concat([times, pd.to_numeric(win_prob_df["Time"], errors="coerce")], ignore_index=True)
    if shot_df is not None and not shot_df.empty:
        t = pd.to_numeric(shot_df.get("Time", shot_df.get("Frame", 0)), errors="coerce")
        if "Frame" in shot_df.columns and "Time" not in shot_df.columns:
            t = t / 30.0
        times = pd.concat([times, t], ignore_index=True)
    if vaep_df is not None and not vaep_df.empty:
        t = pd.to_numeric(vaep_df.get("Time", vaep_df.get("Frame", 0)), errors="coerce")
        if "Frame" in vaep_df.columns and "Time" not in vaep_df.columns:
            t = t / 30.0
        times = pd.concat([times, t], ignore_index=True)
    if momentum_series is not None and not momentum_series.empty:
        times = pd.concat([times, pd.to_numeric(pd.Series(momentum_series.index), errors="coerce")], ignore_index=True)

    grid = pd.DataFrame({"time": sorted(times.dropna().astype(float).unique().tolist())})
    if grid.empty:
        return grid

    if win_prob_df is not None and not win_prob_df.empty and "Time" in win_prob_df.columns:
        wp = win_prob_df[["Time", "WinProb"]].copy()
        wp["Time"] = pd.to_numeric(wp["Time"], errors="coerce")
        grid = grid.merge(wp.rename(columns={"Time": "time", "WinProb": "win_prob"}), on="time", how="left")
    if shot_df is not None and not shot_df.empty and "xG" in shot_df.columns:
        s = shot_df.copy()
        s["time"] = pd.to_numeric(s.get("Time", s.get("Frame", 0)), errors="coerce")
        if "Frame" in s.columns and "Time" not in s.columns:
            s["time"] = s["time"] / 30.0
        grid = grid.merge(s[["time", "xG"]].rename(columns={"xG": "xg"}), on="time", how="left")
    if vaep_df is not None and not vaep_df.empty and "VAEP" in vaep_df.columns:
        v = vaep_df.copy()
        v["time"] = pd.to_numeric(v.get("Time", v.get("Frame", 0)), errors="coerce")
        if "Frame" in v.columns and "Time" not in v.columns:
            v["time"] = v["time"] / 30.0
        grid = grid.merge(v[["time", "VAEP"]].rename(columns={"VAEP": "vaep"}), on="time", how="left")
    if momentum_series is not None and not momentum_series.empty:
        m = pd.DataFrame({"time": pd.to_numeric(pd.Series(momentum_series.index), errors="coerce"), "pressure": pd.to_numeric(momentum_series.values, errors="coerce")})
        grid = grid.merge(m, on="time", how="left")

    return grid.sort_values("time", kind="mergesort").reset_index(drop=True)
