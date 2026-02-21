import pandas as pd

from analytics.possession_value import (
    CANONICAL_STATE_COLUMNS,
    compute_action_value_deltas,
    fit_value_model,
)


def test_fit_value_model_and_predict_columns():
    states = pd.DataFrame(
        [
            {"BluePossessionBelief": 0.6, "OrangePossessionBelief": 0.4, "BallPosY": 1000, "BallVelY": 300, "NearestBlueDist": 500, "NearestOrangeDist": 900, "TeamBoostAvgBlue": 40, "TeamBoostAvgOrange": 30, "PressureBlue": 0.7, "PressureOrange": 0.2, "BlueAttacking": 1, "OrangeAttacking": 0},
            {"BluePossessionBelief": 0.4, "OrangePossessionBelief": 0.6, "BallPosY": -800, "BallVelY": -250, "NearestBlueDist": 1000, "NearestOrangeDist": 450, "TeamBoostAvgBlue": 20, "TeamBoostAvgOrange": 45, "PressureBlue": 0.2, "PressureOrange": 0.6, "BlueAttacking": 0, "OrangeAttacking": 1},
        ]
    )
    outcomes = pd.DataFrame({"ExpectedGoalDiff_3s": [0.08, -0.05], "ExpectedGoalDiff_10s": [0.12, -0.09]})
    model = fit_value_model(states, outcomes)
    pred = model.predict_state_value(states.iloc[0])
    assert set(pred.keys()) == {"3s", "10s"}


def test_compute_action_value_deltas_adds_alias_vaep():
    states = pd.DataFrame(
        [
            {"MatchID": "m1", "Frame": 10, "BluePossessionBelief": 0.5, "OrangePossessionBelief": 0.5, "BallPosY": 0, "BallVelY": 0, "NearestBlueDist": 600, "NearestOrangeDist": 600, "TeamBoostAvgBlue": 30, "TeamBoostAvgOrange": 30, "PressureBlue": 0.3, "PressureOrange": 0.3, "BlueAttacking": 0, "OrangeAttacking": 0},
            {"MatchID": "m1", "Frame": 11, "BluePossessionBelief": 0.7, "OrangePossessionBelief": 0.3, "BallPosY": 250, "BallVelY": 500, "NearestBlueDist": 500, "NearestOrangeDist": 900, "TeamBoostAvgBlue": 35, "TeamBoostAvgOrange": 25, "PressureBlue": 0.5, "PressureOrange": 0.2, "BlueAttacking": 1, "OrangeAttacking": 0},
        ]
    )
    events = pd.DataFrame([{"MatchID": "m1", "Frame": 10, "PostFrame": 11, "Player": "A", "Team": "Blue", "EventType": "touch"}])
    out = compute_action_value_deltas(events, states)
    assert "VAEP" in out.columns
    assert "ValueDelta_3s" in out.columns


def test_canonical_columns_contract():
    assert "MatchID" in CANONICAL_STATE_COLUMNS
    assert "Frame" in CANONICAL_STATE_COLUMNS


def test_compute_action_value_deltas_uses_nearest_frame_when_exact_missing():
    states = pd.DataFrame(
        [
            {"MatchID": "m1", "Frame": 10, "BluePossessionBelief": 0.4, "OrangePossessionBelief": 0.6, "BallPosY": -100, "BallVelY": -100, "NearestBlueDist": 800, "NearestOrangeDist": 500, "TeamBoostAvgBlue": 20, "TeamBoostAvgOrange": 40, "PressureBlue": 0.2, "PressureOrange": 0.5, "BlueAttacking": 0, "OrangeAttacking": 1},
            {"MatchID": "m1", "Frame": 20, "BluePossessionBelief": 0.8, "OrangePossessionBelief": 0.2, "BallPosY": 300, "BallVelY": 450, "NearestBlueDist": 400, "NearestOrangeDist": 900, "TeamBoostAvgBlue": 50, "TeamBoostAvgOrange": 20, "PressureBlue": 0.7, "PressureOrange": 0.1, "BlueAttacking": 1, "OrangeAttacking": 0},
        ]
    )
    events = pd.DataFrame([{"MatchID": "m1", "Frame": 14, "PostFrame": 15, "Player": "A", "Team": "Blue", "EventType": "touch"}])
    out = compute_action_value_deltas(events, states)
    assert float(out.loc[0, "ValueDelta_3s"]) == 0.0
    assert float(out.loc[0, "ValueDelta_10s"]) == 0.0
