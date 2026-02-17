import pandas as pd

from analytics import counterfactuals
from analytics.counterfactuals import (
    ACTION_LIBRARY,
    build_action_library,
    build_coach_report,
    derive_replay_priors,
    extract_decision_moments,
    score_candidate_actions,
)


def _sample_states():
    return pd.DataFrame(
        [
            {
                "MatchID": "m1",
                "Frame": 30,
                "BluePossessionBelief": 0.52,
                "OrangePossessionBelief": 0.48,
                "BallPosX": 50,
                "BallPosY": 800,
                "BallPosZ": 90,
                "BallVelX": 100,
                "BallVelY": 400,
                "BallVelZ": 0,
                "BallSpeedUUPerSec": 1200,
                "NearestBlueDist": 550,
                "NearestOrangeDist": 800,
                "TeamBoostAvgBlue": 34,
                "TeamBoostAvgOrange": 28,
                "PressureBlue": 0.6,
                "PressureOrange": 0.35,
                "BlueAttacking": 1,
                "OrangeAttacking": 0,
                "SecondsRemaining": 290,
            },
            {
                "MatchID": "m1",
                "Frame": 60,
                "BluePossessionBelief": 0.46,
                "OrangePossessionBelief": 0.54,
                "BallPosX": -200,
                "BallPosY": -500,
                "BallPosZ": 80,
                "BallVelX": 150,
                "BallVelY": -250,
                "BallVelZ": 0,
                "BallSpeedUUPerSec": 980,
                "NearestBlueDist": 900,
                "NearestOrangeDist": 600,
                "TeamBoostAvgBlue": 22,
                "TeamBoostAvgOrange": 31,
                "PressureBlue": 0.4,
                "PressureOrange": 0.5,
                "BlueAttacking": 0,
                "OrangeAttacking": 1,
                "SecondsRemaining": 289,
            },
        ]
    )


def test_action_library_filters_and_scores():
    states = _sample_states()
    priors = derive_replay_priors(states)
    snapshot = states.iloc[0]
    actions = build_action_library(snapshot, priors, candidate_actions=["challenge_now", "third_man_hold"])
    assert actions
    assert {a.name for a in actions}.issubset(ACTION_LIBRARY.keys())

    win_prob_df = pd.DataFrame({"Time": [1.0, 2.0, 3.0], "WinProb": [50.0, 51.0, 53.0]})
    scored = score_candidate_actions(snapshot, actions, team="Blue", value_model=None, win_prob_df=win_prob_df, reference_time=1.0)
    assert not scored.empty
    assert {"ExpectedSwing", "Confidence", "RoleTargets"}.issubset(scored.columns)


def test_extract_moments_and_build_report():
    states = _sample_states()
    momentum = pd.Series([5, -15, 20, -12, 18, 6], index=[0, 1, 2, 3, 4, 5])
    win_prob_df = pd.DataFrame({"Time": [0, 1, 2, 3, 4, 5], "WinProb": [50, 49, 52, 47, 53, 54]})
    moments = extract_decision_moments(momentum, win_prob_df, top_n=3)
    assert not moments.empty
    assert {"Frame", "Leverage", "WindowStartFrame", "WindowEndFrame"}.issubset(moments.columns)

    rotation_timeline = pd.DataFrame(
        {
            "Frame": [60, 60, 60],
            "Team": ["Blue", "Blue", "Blue"],
            "Role": ["1st", "2nd", "2nd"],
            "Player": ["A", "B", "C"],
        }
    )
    rotation_summary = pd.DataFrame({"Name": ["A"], "Team": ["Blue"], "Time_1st%": [45.0], "Time_2nd%": [40.0]})
    report = build_coach_report(states, momentum, win_prob_df, rotation_timeline, rotation_summary, team="Blue", top_n=5)
    assert not report.empty
    assert {"MissedSwing", "RecommendationText", "ClipKey", "RankScore", "ActionabilityFlag"}.issubset(report.columns)


def test_build_coach_report_actionability_dedup_and_rank_deterministic(monkeypatch):
    states = pd.DataFrame([{"Frame": f, "TeamBoostAvgBlue": 40, "PressureBlue": 0.5, "BallPosX": 0, "BallPosY": 0} for f in [100, 120, 140, 260]])
    momentum = pd.Series([2.0, 2.0, 2.0, 2.0], index=[100 / 30, 120 / 30, 140 / 30, 260 / 30])
    win_prob_df = pd.DataFrame({"Time": [0.0, 10.0], "WinProb": [50.0, 50.0]})
    rotation_timeline = pd.DataFrame({"Frame": [100, 120, 140, 260], "Team": ["Blue"] * 4, "Role": ["2nd"] * 4, "Player": ["A"] * 4})
    rotation_summary = pd.DataFrame({"Name": ["A"], "Team": ["Blue"], "Time_1st%": [25.0]})

    monkeypatch.setattr(counterfactuals, "build_action_library", lambda *_args, **_kwargs: [counterfactuals.ACTION_LIBRARY["challenge_now"]])

    monkeypatch.setattr(counterfactuals, "extract_decision_moments", lambda *_args, **_kwargs: pd.DataFrame([
        {"Frame": 100, "Time": 3.3, "Leverage": 3.0, "WindowStartFrame": 50, "WindowEndFrame": 150},
        {"Frame": 120, "Time": 4.0, "Leverage": 2.8, "WindowStartFrame": 70, "WindowEndFrame": 170},
        {"Frame": 140, "Time": 4.6, "Leverage": 2.4, "WindowStartFrame": 90, "WindowEndFrame": 190},
        {"Frame": 260, "Time": 8.6, "Leverage": 2.9, "WindowStartFrame": 210, "WindowEndFrame": 310},
    ]))

    sequence = [
        {"ExpectedSwing": 0.08, "Confidence": 0.90, "Action": "challenge_now", "ModelSource": "m1"},
        {"ExpectedSwing": 0.07, "Confidence": 0.86, "Action": "challenge_now", "ModelSource": "m1"},
        {"ExpectedSwing": 0.01, "Confidence": 0.30, "Action": "shadow_defend", "ModelSource": "m1"},
        {"ExpectedSwing": 0.06, "Confidence": 0.85, "Action": "fake_challenge", "ModelSource": "m2"},
    ]

    def fake_score(*_args, **_kwargs):
        row = sequence.pop(0)
        return pd.DataFrame([row])

    monkeypatch.setattr(counterfactuals, "score_candidate_actions", fake_score)

    report = build_coach_report(states, momentum, win_prob_df, rotation_timeline, rotation_summary, top_n=3)
    assert len(report) == 3
    assert report["ActionabilityFlag"].tolist() == [True, True, False]
    assert report["Frame"].tolist() == [100, 260, 140]
    assert report["RankScore"].is_monotonic_decreasing

    sequence[:] = [
        {"ExpectedSwing": 0.08, "Confidence": 0.90, "Action": "challenge_now", "ModelSource": "m1"},
        {"ExpectedSwing": 0.07, "Confidence": 0.86, "Action": "challenge_now", "ModelSource": "m1"},
        {"ExpectedSwing": 0.01, "Confidence": 0.30, "Action": "shadow_defend", "ModelSource": "m1"},
        {"ExpectedSwing": 0.06, "Confidence": 0.85, "Action": "fake_challenge", "ModelSource": "m2"},
    ]
    report_again = build_coach_report(states, momentum, win_prob_df, rotation_timeline, rotation_summary, top_n=3)
    pd.testing.assert_frame_equal(report.reset_index(drop=True), report_again.reset_index(drop=True))
