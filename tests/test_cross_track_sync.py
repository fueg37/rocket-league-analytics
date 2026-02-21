import pandas as pd

from analytics.director_mode import synchronize_track_times


def test_cross_track_time_synchronization():
    win_prob_df = pd.DataFrame({"Time": [1.0, 2.0], "WinProb": [52, 48]})
    shot_df = pd.DataFrame({"Time": [1.5], "xG": [0.3]})
    vaep_df = pd.DataFrame({"Time": [1.0, 2.5], "VAEP": [0.1, -0.2]})
    momentum_series = pd.Series([0.2, -0.1], index=[0.5, 2.0])

    sync = synchronize_track_times(
        win_prob_df=win_prob_df,
        shot_df=shot_df,
        vaep_df=vaep_df,
        momentum_series=momentum_series,
    )

    assert sync["time"].is_monotonic_increasing
    assert {0.5, 1.0, 1.5, 2.0, 2.5}.issubset(set(sync["time"].tolist()))
    row = sync.loc[sync["time"] == 1.5].iloc[0]
    assert row["xg"] == 0.3
