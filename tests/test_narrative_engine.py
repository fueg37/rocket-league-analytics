import pandas as pd

from analytics.narrative_engine import generate_narrative_report


def _sample_inputs():
    momentum_series = pd.Series([0.1, -0.05, 0.2], index=[1.0, 2.0, 3.0])
    possession_value_df = pd.DataFrame([
        {"Frame": 90, "VAEP": 0.12},
        {"Frame": 120, "VAEP": -0.03},
    ])
    rotation_summary = pd.DataFrame([{"Time_1st%": 42.5}])
    shot_df = pd.DataFrame([
        {"Frame": 200, "xG": 0.35, "Result": "Goal"},
        {"Frame": 260, "xG": 0.12, "Result": "Saved"},
    ])
    save_events_df = pd.DataFrame([
        {"Frame": 210, "SaveImpact": 0.22},
        {"Frame": 270, "SaveImpact": -0.05},
    ])
    situational_df = pd.DataFrame([
        {"Goals_Last_Min": 1, "Saves_Last_Min": 2},
    ])
    kickoff_df = pd.DataFrame([
        {"Player": "A", "Result": "Win"},
        {"Player": "B", "Result": "Loss"},
    ])
    return {
        "momentum_series": momentum_series,
        "possession_value_df": possession_value_df,
        "rotation_summary": rotation_summary,
        "shot_df": shot_df,
        "save_events_df": save_events_df,
        "situational_df": situational_df,
        "kickoff_df": kickoff_df,
    }


def test_generate_narrative_report_requires_evidence_per_claim():
    report = generate_narrative_report(**_sample_inputs(), tone="balanced", verbosity="standard", role_target="coaching_review", players_per_team=3)
    assert report.claims
    assert all(claim.evidence for claim in report.claims)


def test_generate_narrative_report_export_formats():
    report = generate_narrative_report(**_sample_inputs(), tone="coach", verbosity="deep", role_target="team_scrim", players_per_team=2)
    md = report.to_markdown()
    js = report.to_json()
    assert "# Narrative Studio Report" in md
    assert "coaching_review" not in js
    assert '"role_target": "team_scrim"' in js


def test_coaching_language_adapts_to_2v2_depth_model():
    report_2s = generate_narrative_report(**_sample_inputs(), tone="coach", verbosity="standard", role_target="coaching_review", players_per_team=2)
    defensive_claims = [c for c in report_2s.claims if c.phase == "defensive_zone"]
    assert defensive_claims
    assert "second-player support depth" in defensive_claims[0].text
    assert "third-man depth" not in defensive_claims[0].text

    rec_text = " ".join(report_2s.recommendations)
    assert "second-player support depth" in rec_text
    assert "third-man depth" not in rec_text
