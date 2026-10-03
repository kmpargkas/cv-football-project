"""Pitch projection + smoothing over synthetic homographies (no data/weights)."""

import numpy as np
import pytest

from football_tracker.homography.record import FrameCalibration
from football_tracker.projection.to_pitch import (
    BallPitchState,
    PitchPosition,
    ball_speeds,
    project_ball_states,
    project_player_tracks,
    smooth_ball,
    smooth_positions,
)
from football_tracker.tracking.ball_trajectory import BallState
from football_tracker.tracking.tracker import TrackedObjects

# px -> m: divide by 10. Foot point (500, 300) -> (50, 30) m.
H_SCALE = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1.0]])


def _calib(frame_idx, H=H_SCALE, low_confidence=False):
    return FrameCalibration(
        frame_idx=frame_idx,
        H=H,
        source="solved",
        low_confidence=low_confidence,
        gmc=np.eye(2, 3),
    )


def _objs(boxes, ids):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    n = len(boxes)
    return TrackedObjects(
        xyxy=boxes,
        id=np.asarray(ids, dtype=np.int64),
        confidence=np.ones(n, dtype=np.float32),
        class_id=np.zeros(n, dtype=np.int64),
    )


def test_project_player_tracks_foot_point_times_h():
    tracks = {0: _objs([[480, 200, 520, 300]], [7])}  # foot point (500, 300)
    out = project_player_tracks(tracks, [_calib(0)])
    assert len(out) == 1
    p = out[0]
    assert (p.frame_idx, p.track_id, p.reliable) == (0, 7, True)
    np.testing.assert_allclose(p.xy_m, [50.0, 30.0], atol=1e-6)


def test_project_player_tracks_skips_uncalibrated_and_flags_low_confidence():
    tracks = {0: _objs([[0, 0, 10, 10]], [1]), 1: _objs([[0, 0, 10, 10]], [1])}
    calib = [_calib(0, H=None), _calib(1, low_confidence=True)]
    out = project_player_tracks(tracks, calib)
    assert [p.frame_idx for p in out] == [1]  # frame 0 had no H at all
    assert out[0].reliable is False


def test_project_ball_states_missing_stays_missing():
    states = {
        0: BallState(
            frame_idx=0,
            xy=np.array([500.0, 300.0]),
            source="detected",
            airborne=False,
            confidence=0.9,
        ),
        1: BallState(frame_idx=1, xy=None, source="missing", airborne=False, confidence=0.0),
    }
    out = project_ball_states(states, [_calib(0), _calib(1)])
    np.testing.assert_allclose(out[0].xy_m, [50.0, 30.0], atol=1e-6)
    assert out[0].reliable is True
    assert out[1].xy_m is None and out[1].reliable is False


def test_smooth_positions_reduces_noise_and_keeps_endpoints_sane():
    rng = np.random.default_rng(0)
    true = np.stack([np.linspace(0, 20, 61), np.full(61, 30.0)], axis=1)
    noisy = true + rng.normal(0, 0.3, true.shape)
    positions = [
        PitchPosition(frame_idx=i, track_id=1, xy_m=noisy[i], reliable=True) for i in range(61)
    ]
    smoothed = smooth_positions(positions, window=15)
    assert len(smoothed) == 61  # smoothing never drops rows
    err_raw = np.abs(np.array([p.xy_m for p in positions]) - true).mean()
    err_smooth = np.abs(np.array([p.xy_m for p in smoothed]) - true).mean()
    assert err_smooth < err_raw / 2


def test_smooth_positions_breaks_at_track_gaps():
    # Two runs of one track far apart; smoothing must not bleed across the gap.
    a = [PitchPosition(i, 1, np.array([0.0, 0.0]), True) for i in range(5)]
    b = [PitchPosition(i, 1, np.array([50.0, 50.0]), True) for i in range(100, 105)]
    smoothed = smooth_positions(a + b, window=15)
    np.testing.assert_allclose(smoothed[0].xy_m, [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(smoothed[-1].xy_m, [50.0, 50.0], atol=1e-9)


def test_smooth_ball_breaks_at_missing_frames():
    # An explicit missing frame sits between two runs at otherwise-contiguous
    # frame indices; smoothing must not blend across it into the gap.
    a = [BallPitchState(i, np.array([0.0, 0.0]), "detected", False, True) for i in range(5)]
    gap = [BallPitchState(5, None, "missing", False, False)]
    b = [BallPitchState(i, np.array([50.0, 50.0]), "detected", False, True) for i in range(6, 11)]
    smoothed = smooth_ball(a + gap + b, window=15)
    assert len(smoothed) == 11  # smoothing never drops rows, including the missing one
    assert smoothed[5].xy_m is None
    np.testing.assert_allclose(smoothed[0].xy_m, [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(smoothed[-1].xy_m, [50.0, 50.0], atol=1e-9)


def test_ball_speeds_consecutive_frames_only():
    ball = [
        BallPitchState(0, np.array([0.0, 0.0]), "detected", False, True),
        BallPitchState(1, np.array([0.5, 0.0]), "detected", False, True),  # 0.5 m/frame
        BallPitchState(5, np.array([10.0, 0.0]), "detected", False, True),  # after a gap
    ]
    speeds = ball_speeds(ball, fps=60.0)
    assert speeds[1] == pytest.approx(30.0)  # 0.5 m * 60 fps
    assert 5 not in speeds  # no speed across a gap; caller treats absent as inf
