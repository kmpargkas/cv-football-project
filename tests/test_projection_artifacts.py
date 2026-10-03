"""Round trips for the projection CSV formats."""

import numpy as np

from football_tracker.projection.artifacts import (
    PossessionState,
    read_ball_positions_csv,
    read_positions_csv,
    read_possession_csv,
    write_ball_positions_csv,
    write_positions_csv,
    write_possession_csv,
)
from football_tracker.projection.to_pitch import BallPitchState, PitchPosition


def test_positions_round_trip(tmp_path):
    rows = [
        PitchPosition(0, 7, np.array([50.123, 30.456]), True),
        PitchPosition(1, 7, np.array([50.2, 30.5]), False),
    ]
    p = tmp_path / "positions.csv"
    write_positions_csv(p, rows)
    back = read_positions_csv(p)
    assert len(back) == 2
    assert (back[0].frame_idx, back[0].track_id, back[0].reliable) == (0, 7, True)
    np.testing.assert_allclose(back[0].xy_m, [50.123, 30.456], atol=1e-3)
    assert (back[1].frame_idx, back[1].track_id, back[1].reliable) == (1, 7, False)
    np.testing.assert_allclose(back[1].xy_m, [50.2, 30.5], atol=1e-3)


def test_ball_positions_round_trip_preserves_missing(tmp_path):
    rows = [
        BallPitchState(0, np.array([10.0, 20.0]), "detected", False, True),
        BallPitchState(1, None, "missing", False, False),
        BallPitchState(2, np.array([11.0, 20.0]), "interpolated", True, True),
    ]
    p = tmp_path / "ball_positions.csv"
    write_ball_positions_csv(p, rows)
    back = read_ball_positions_csv(p)
    assert back[1].xy_m is None and back[1].source == "missing"
    assert back[0].airborne is False and back[0].reliable is True
    assert back[1].airborne is False and back[1].reliable is False
    assert back[2].airborne is True and back[2].reliable is True
    np.testing.assert_allclose(back[0].xy_m, [10.0, 20.0], atol=1e-3)


def test_possession_round_trip(tmp_path):
    rows = [
        PossessionState(100, "unknown", -1),
        PossessionState(101, "team_a", 7),
    ]
    p = tmp_path / "possession.csv"
    write_possession_csv(p, rows)
    back = read_possession_csv(p)
    assert back == rows
