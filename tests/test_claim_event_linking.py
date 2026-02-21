import pandas as pd

from analytics.narrative_engine import generate_narrative_report


def test_claims_link_to_director_event_ids():
    director_queue = pd.DataFrame([
        {"event_id": "ko_0", "event_type": "kickoff"},
        {"event_id": "vaep_1", "event_type": "value_swing"},
        {"event_id": "shot_2", "event_type": "shot_chance"},
        {"event_id": "save_3", "event_type": "save"},
        {"event_id": "wp_4", "event_type": "win_probability_swing"},
    ])
    report = generate_narrative_report(
        momentum_series=pd.Series([0.1, 0.2], index=[1.0, 2.0]),
        possession_value_df=pd.DataFrame([{"Frame": 30, "VAEP": 0.1}]),
        rotation_summary=pd.DataFrame([{"Time_1st%": 30.0}]),
        shot_df=pd.DataFrame([{"Frame": 40, "xG": 0.3, "Result": "Goal"}]),
        save_events_df=pd.DataFrame([{"Frame": 50, "SaveImpact": 0.2}]),
        situational_df=pd.DataFrame([{"Goals_Last_Min": 1, "Saves_Last_Min": 1}]),
        kickoff_df=pd.DataFrame([{"Player": "A", "Result": "Win"}]),
        director_event_queue=director_queue,
    )

    assert report.claims
    assert all(claim.canonical_event_id for claim in report.claims)
    ids = {c.canonical_event_id for c in report.claims}
    assert ids.issubset(set(director_queue["event_id"]))
