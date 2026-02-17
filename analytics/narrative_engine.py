"""Narrative Studio engine with evidence-linked claims and exportable reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Literal

import pandas as pd


NarrativePhase = Literal["kickoff", "transition", "offensive_zone", "defensive_zone", "clutch_minute"]
NarrativeTone = Literal["balanced", "hype", "coach"]
RoleTarget = Literal["solo_queue", "team_scrim", "coaching_review"]
VerbosityLevel = Literal["brief", "standard", "deep"]


@dataclass(slots=True)
class EvidenceRef:
    source: str
    row_index: int | None = None
    frame: int | None = None
    detail: str = ""


@dataclass(slots=True)
class NarrativeClaim:
    phase: NarrativePhase
    tone: NarrativeTone
    text: str
    confidence: float
    confidence_language: str
    evidence: list[EvidenceRef]


@dataclass(slots=True)
class NarrativeReport:
    tone: NarrativeTone
    verbosity: VerbosityLevel
    role_target: RoleTarget
    claims: list[NarrativeClaim]
    recommendations: list[str]

    def to_json_dict(self) -> dict:
        return {
            "tone": self.tone,
            "verbosity": self.verbosity,
            "role_target": self.role_target,
            "claims": [
                {
                    "phase": c.phase,
                    "tone": c.tone,
                    "text": c.text,
                    "confidence": round(c.confidence, 3),
                    "confidence_language": c.confidence_language,
                    "evidence": [asdict(e) for e in c.evidence],
                }
                for c in self.claims
            ],
            "recommendations": self.recommendations,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), indent=2)

    def to_markdown(self) -> str:
        lines = [
            "# Narrative Studio Report",
            f"- Tone: `{self.tone}`",
            f"- Verbosity: `{self.verbosity}`",
            f"- Role target: `{self.role_target}`",
            "",
            "## Claims",
        ]
        for claim in self.claims:
            lines.append(f"### {claim.phase.replace('_', ' ').title()}")
            lines.append(f"- {claim.text}")
            lines.append(f"- Confidence: {claim.confidence_language} ({claim.confidence:.2f})")
            lines.append("- Evidence:")
            for ev in claim.evidence:
                suffix = []
                if ev.row_index is not None:
                    suffix.append(f"row={ev.row_index}")
                if ev.frame is not None:
                    suffix.append(f"frame={ev.frame}")
                suffix_str = f" ({', '.join(suffix)})" if suffix else ""
                lines.append(f"  - `{ev.source}`{suffix_str}: {ev.detail}")
            lines.append("")
        if self.recommendations:
            lines.append("## Role-Targeted Recommendations")
            for rec in self.recommendations:
                lines.append(f"- {rec}")
        return "\n".join(lines)


_TEMPLATES: dict[NarrativeTone, dict[NarrativePhase, str]] = {
    "balanced": {
        "kickoff": "Kickoff control sat at {kickoff_win_rate:.1f}% with {kickoff_samples} tracked kickoffs.",
        "transition": "Transition swings averaged {transition_swing:+.3f} VAEP across sampled sequences.",
        "offensive_zone": "Offensive pressure produced {offensive_xg:.2f} expected goals on {offensive_shots} shots.",
        "defensive_zone": "Defensive resilience posted {save_impact:+.2f} save impact with rotation OS {rotation_os:.1f}.",
        "clutch_minute": "Clutch minute output was {clutch_goals} goals and {clutch_saves} saves in late-game windows.",
    },
    "hype": {
        "kickoff": "Kickoff battles were explosive at {kickoff_win_rate:.1f}% control over {kickoff_samples} launches.",
        "transition": "Transitions flipped fast, with a {transition_swing:+.3f} VAEP surge through midfield chaos.",
        "offensive_zone": "Attack mode generated {offensive_xg:.2f} xG from {offensive_shots} pressure shots.",
        "defensive_zone": "Backline heroics delivered {save_impact:+.2f} save impact while rotation OS held {rotation_os:.1f}.",
        "clutch_minute": "Final-minute clutch: {clutch_goals} goals and {clutch_saves} saves under pressure.",
    },
    "coach": {
        "kickoff": "Kickoff process returned {kickoff_win_rate:.1f}% wins ({kickoff_samples} samples); review first-touch consistency.",
        "transition": "Transition value settled at {transition_swing:+.3f} VAEP; spacing decisions drove the delta.",
        "offensive_zone": "Offense created {offensive_xg:.2f} xG on {offensive_shots} attempts; prioritize shot quality over volume.",
        "defensive_zone": "Defense delivered {save_impact:+.2f} save impact with rotation OS {rotation_os:.1f}; {defensive_depth_label}.",
        "clutch_minute": "Clutch profile: {clutch_goals} goals, {clutch_saves} saves in the final minute states.",
    },
}


def _confidence_language(score: float) -> str:
    if score >= 0.80:
        return "high confidence"
    if score >= 0.55:
        return "moderate confidence"
    return "low confidence"


def _signal_metrics(
    momentum_series: pd.Series,
    possession_value_df: pd.DataFrame,
    rotation_summary: pd.DataFrame,
    shot_df: pd.DataFrame,
    save_events_df: pd.DataFrame,
    situational_df: pd.DataFrame,
    kickoff_df: pd.DataFrame,
) -> dict[str, float]:
    return {
        "kickoff_win_rate": float((kickoff_df.get("Result", pd.Series(dtype=object)) == "Win").mean() * 100.0) if not kickoff_df.empty else 0.0,
        "kickoff_samples": float(len(kickoff_df)),
        "transition_swing": float(pd.to_numeric(possession_value_df.get("VAEP", pd.Series(dtype=float)), errors="coerce").fillna(0.0).mean()) if not possession_value_df.empty else 0.0,
        "offensive_xg": float(pd.to_numeric(shot_df.get("xG", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) if not shot_df.empty else 0.0,
        "offensive_shots": float(len(shot_df)),
        "save_impact": float(pd.to_numeric(save_events_df.get("SaveImpact", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) if not save_events_df.empty else 0.0,
        "rotation_os": float(pd.to_numeric(rotation_summary.get("Time_1st%", pd.Series(dtype=float)), errors="coerce").fillna(0.0).mean()) if not rotation_summary.empty else 0.0,
        "clutch_goals": float(pd.to_numeric(situational_df.get("Goals_Last_Min", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) if not situational_df.empty else 0.0,
        "clutch_saves": float(pd.to_numeric(situational_df.get("Saves_Last_Min", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) if not situational_df.empty else 0.0,
        "momentum_consistency": float(1.0 / (1.0 + pd.to_numeric(momentum_series, errors="coerce").fillna(0.0).std())) if not momentum_series.empty else 0.3,
    }


def _evidence_for_phase(
    phase: NarrativePhase,
    momentum_series: pd.Series,
    possession_value_df: pd.DataFrame,
    rotation_summary: pd.DataFrame,
    shot_df: pd.DataFrame,
    save_events_df: pd.DataFrame,
    situational_df: pd.DataFrame,
    kickoff_df: pd.DataFrame,
) -> list[EvidenceRef]:
    evidence: list[EvidenceRef] = []
    if phase == "kickoff" and not kickoff_df.empty:
        first = kickoff_df.iloc[0]
        evidence.append(EvidenceRef("kickoff_df", row_index=int(kickoff_df.index[0]), detail=f"Result={first.get('Result', 'n/a')}, Player={first.get('Player', 'n/a')}"))
    elif phase == "transition":
        if not possession_value_df.empty:
            row = possession_value_df.iloc[0]
            evidence.append(EvidenceRef("possession_value_df", row_index=int(possession_value_df.index[0]), frame=int(row.get("Frame", 0)) if pd.notna(row.get("Frame", None)) else None, detail=f"VAEP={float(pd.to_numeric(row.get('VAEP', 0.0), errors='coerce') or 0.0):+.3f}"))
        if not momentum_series.empty:
            idx = momentum_series.index[0]
            evidence.append(EvidenceRef("momentum_series", frame=int(float(idx) * 30), detail=f"momentum={float(momentum_series.iloc[0]):+.3f}"))
    elif phase == "offensive_zone" and not shot_df.empty:
        row = shot_df.sort_values("xG", ascending=False).iloc[0] if "xG" in shot_df.columns else shot_df.iloc[0]
        ridx = int(row.name) if isinstance(row.name, (int, float)) else 0
        evidence.append(EvidenceRef("shot_df", row_index=ridx, frame=int(row.get("Frame", 0)) if pd.notna(row.get("Frame", None)) else None, detail=f"xG={float(pd.to_numeric(row.get('xG', 0.0), errors='coerce') or 0.0):.3f}, Result={row.get('Result', 'n/a')}"))
    elif phase == "defensive_zone":
        if not save_events_df.empty:
            row = save_events_df.sort_values("SaveImpact", ascending=False).iloc[0] if "SaveImpact" in save_events_df.columns else save_events_df.iloc[0]
            ridx = int(row.name) if isinstance(row.name, (int, float)) else 0
            evidence.append(EvidenceRef("save_events_df", row_index=ridx, frame=int(row.get("Frame", 0)) if pd.notna(row.get("Frame", None)) else None, detail=f"SaveImpact={float(pd.to_numeric(row.get('SaveImpact', 0.0), errors='coerce') or 0.0):+.3f}"))
        if not rotation_summary.empty:
            row = rotation_summary.iloc[0]
            ridx = int(rotation_summary.index[0]) if isinstance(rotation_summary.index[0], (int, float)) else 0
            evidence.append(EvidenceRef("rotation_summary", row_index=ridx, detail=f"Time_1st%={float(pd.to_numeric(row.get('Time_1st%', 0.0), errors='coerce') or 0.0):.1f}"))
    elif phase == "clutch_minute" and not situational_df.empty:
        row = situational_df.iloc[0]
        ridx = int(situational_df.index[0]) if isinstance(situational_df.index[0], (int, float)) else 0
        evidence.append(EvidenceRef("situational_df", row_index=ridx, detail=f"Goals_Last_Min={int(pd.to_numeric(row.get('Goals_Last_Min', 0), errors='coerce') or 0)}, Saves_Last_Min={int(pd.to_numeric(row.get('Saves_Last_Min', 0), errors='coerce') or 0)}"))
    return evidence


def _recommendations_for_role(role_target: RoleTarget, players_per_team: int | None = None) -> list[str]:
    if role_target == "solo_queue":
        return [
            "Favor low-commit transition options when momentum confidence is moderate or lower.",
            "Use kickoff setups with highest observed win evidence before improvising.",
        ]
    if role_target == "team_scrim":
        return [
            "Run scripted transition drills around the lowest-evidence rotation OS moments.",
            "Audit offensive-zone shot selection against xG-rich evidence clips.",
        ]
    support_depth_callout = "Align third-man depth." if (players_per_team or 3) >= 3 else "Align second-player support depth."
    return [
        "Prioritize VOD clips where confidence is low but outcome impact is high.",
        f"Track defensive-zone claims against save model evidence for player feedback loops. {support_depth_callout}",
    ]


def generate_narrative_report(
    *,
    momentum_series: pd.Series,
    possession_value_df: pd.DataFrame,
    rotation_summary: pd.DataFrame,
    shot_df: pd.DataFrame,
    save_events_df: pd.DataFrame,
    situational_df: pd.DataFrame,
    kickoff_df: pd.DataFrame,
    tone: NarrativeTone = "balanced",
    verbosity: VerbosityLevel = "standard",
    role_target: RoleTarget = "coaching_review",
    players_per_team: int | None = None,
) -> NarrativeReport:
    metrics = _signal_metrics(
        momentum_series,
        possession_value_df,
        rotation_summary,
        shot_df,
        save_events_df,
        situational_df,
        kickoff_df,
    )
    defensive_depth_label = "align third-man depth" if (players_per_team or 3) >= 3 else "align second-player support depth"
    phases: list[NarrativePhase] = ["kickoff", "transition", "offensive_zone", "defensive_zone", "clutch_minute"]
    if verbosity == "brief":
        phases = ["transition", "offensive_zone", "clutch_minute"]
    claims: list[NarrativeClaim] = []
    for phase in phases:
        evidence = _evidence_for_phase(
            phase,
            momentum_series,
            possession_value_df,
            rotation_summary,
            shot_df,
            save_events_df,
            situational_df,
            kickoff_df,
        )
        if not evidence:
            continue
        confidence = min(0.95, 0.35 + 0.1 * len(evidence) + 0.4 * metrics.get("momentum_consistency", 0.3))
        template = _TEMPLATES[tone][phase]
        text = template.format(**metrics, defensive_depth_label=defensive_depth_label)
        if verbosity == "deep":
            text = f"{text} Evidence-backed phase summary built from canonical model outputs."
        claims.append(
            NarrativeClaim(
                phase=phase,
                tone=tone,
                text=text,
                confidence=confidence,
                confidence_language=_confidence_language(confidence),
                evidence=evidence,
            )
        )

    # Hard requirement: every claim must include explicit evidence references.
    claims = [c for c in claims if c.evidence]
    return NarrativeReport(
        tone=tone,
        verbosity=verbosity,
        role_target=role_target,
        claims=claims,
        recommendations=_recommendations_for_role(role_target, players_per_team=players_per_team),
    )
