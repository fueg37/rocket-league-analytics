import pandas as pd

from analytics.director_mode import build_director_event_queue


def test_event_queue_determinism():
    win_prob_df = pd.DataFrame({"Time": [1, 2, 3], "WinProb": [50, 65, 40]})
    shot_df = pd.DataFrame({"Time": [1.5, 2.5], "xG": [0.2, 0.5], "Team": ["Blue", "Orange"], "Result": ["Shot", "Goal"]})
    kickoff_df = pd.DataFrame({"Time": [0.1], "Team": ["Blue"], "Result": ["Win"]})
    vaep_df = pd.DataFrame({"Time": [1.1, 2.2], "VAEP": [0.1, -0.3], "Team": ["Blue", "Orange"]})
    save_df = pd.DataFrame({"Time": [2.0], "SaveImpact": [0.25], "Team": ["Blue"]})

    q1 = build_director_event_queue(
        win_prob_df=win_prob_df,
        shot_df=shot_df,
        kickoff_df=kickoff_df,
        vaep_df=vaep_df,
        save_events_df=save_df,
    )
    q2 = build_director_event_queue(
        win_prob_df=win_prob_df,
        shot_df=shot_df,
        kickoff_df=kickoff_df,
        vaep_df=vaep_df,
        save_events_df=save_df,
    )
    assert q1["event_id"].tolist() == q2["event_id"].tolist()
    assert q1["rank_score"].round(6).tolist() == q2["rank_score"].round(6).tolist()
