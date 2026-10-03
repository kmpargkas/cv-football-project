"""When the radar must not draw a confident ball dot.

The airborne flag alone almost never fires, so the marker is keyed on every signal
already present in the projected CSVs.
"""

from __future__ import annotations

import numpy as np

from football_tracker.projection.to_pitch import BallPitchState
from football_tracker.projection.uncertainty import ball_uncertain, ball_verified


def _state(**kw):
    base = {
        "frame_idx": 10,
        "xy_m": np.array([50.0, 30.0]),
        "source": "detected",
        "airborne": False,
        "reliable": True,
    }
    base.update(kw)
    return BallPitchState(**base)


def test_a_clean_detected_ball_is_certain():
    uncertain, reason = ball_uncertain(_state(), speed_mps=4.0)
    assert uncertain is False
    assert reason == ""


def test_the_airborne_flag_marks_uncertain_when_it_does_fire():
    uncertain, reason = ball_uncertain(_state(airborne=True), speed_mps=4.0)
    assert uncertain is True
    assert "airborne" in reason


def test_an_implausible_projected_speed_marks_uncertain():
    """A flat-pitch H sweeps a launched ball at 50-118 m/s."""
    uncertain, reason = ball_uncertain(_state(), speed_mps=60.0, max_plausible_mps=25.0)
    assert uncertain is True
    assert "speed" in reason


def test_an_interpolated_ball_is_uncertain():
    uncertain, reason = ball_uncertain(_state(source="interpolated"), speed_mps=4.0)
    assert uncertain is True
    assert "interpolated" in reason


def test_an_unreliable_homography_marks_uncertain():
    uncertain, reason = ball_uncertain(_state(reliable=False), speed_mps=4.0)
    assert uncertain is True
    assert "homography" in reason


def test_a_missing_position_is_uncertain_without_dividing_by_none():
    uncertain, reason = ball_uncertain(_state(xy_m=None, source="missing"), speed_mps=None)
    assert uncertain is True


def test_an_unknown_speed_alone_does_not_mark_a_clean_ball_uncertain():
    """Speed is unavailable at clip edges; absence of evidence is not evidence."""
    uncertain, _ = ball_uncertain(_state(), speed_mps=None)
    assert uncertain is False


def test_a_speed_exactly_at_the_threshold_is_still_plausible():
    """The bound is exclusive: 25.0 m/s is the fastest speed we still trust."""
    uncertain, _ = ball_uncertain(_state(), speed_mps=25.0, max_plausible_mps=25.0)
    assert uncertain is False


def test_reasons_compose_so_the_render_can_explain_itself():
    uncertain, reason = ball_uncertain(_state(airborne=True, source="interpolated"), speed_mps=99.0)
    assert uncertain is True
    assert "airborne" in reason and "interpolated" in reason and "speed" in reason


# --- the deliverable draws the ball only when every check passes ----------------------


def test_verified_is_a_detected_position_that_passes_every_check():
    assert ball_verified(_state(), speed_mps=4.0) is True
    assert ball_verified(_state(), speed_mps=None) is True  # no delta is not a failure


def test_interpolated_and_missing_positions_are_never_verified():
    assert ball_verified(_state(source="interpolated"), speed_mps=4.0) is False
    assert ball_verified(_state(source="missing", xy_m=None), speed_mps=None) is False


def test_a_detection_that_fails_any_check_is_not_verified():
    assert ball_verified(_state(reliable=False), speed_mps=4.0) is False
    assert ball_verified(_state(airborne=True), speed_mps=4.0) is False
    assert ball_verified(_state(), speed_mps=40.0) is False
