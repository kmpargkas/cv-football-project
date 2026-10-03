"""Possession hysteresis over synthetic pitch positions.

The model is last-touch: a team keeps possession through passes in transit and short
ball dropouts, losing it only when the other side acquires or the ball is gone too long.
"""

import numpy as np

from football_tracker.config import PossessionConfig
from football_tracker.projection.possession import possession_summary, resolve_possession
from football_tracker.projection.to_pitch import BallPitchState, PitchPosition
from football_tracker.reid.roles import Role
from football_tracker.reid.teamid import TrackLabel

CFG = PossessionConfig(radius_m=2.0, acquire_frames=5, max_ball_speed_mps=10.0, max_unknown_s=0.5)
FPS = 10.0  # small numbers keep the synthetic scenarios readable

LABELS = {
    1: TrackLabel(role=Role.TEAM_A, team=0, confidence=1.0, n_samples=10, n_rejected=0),
    2: TrackLabel(role=Role.TEAM_B, team=1, confidence=1.0, n_samples=10, n_rejected=0),
    3: TrackLabel(role=Role.REFEREE, team=-1, confidence=1.0, n_samples=10, n_rejected=0),
    4: TrackLabel(role=Role.GOALKEEPER, team=0, confidence=1.0, n_samples=10, n_rejected=0),
}


def _still(track_id, xy, frames):
    return [PitchPosition(f, track_id, np.array(xy, dtype=float), True) for f in frames]


def _ball(frame, xy, source="detected"):
    xy_m = None if xy is None else np.array(xy, dtype=float)
    return BallPitchState(
        frame, xy_m, source if xy is not None else "missing", False, xy is not None
    )


def test_acquisition_needs_sustained_proximity():
    players = _still(1, (10.0, 10.0), range(20))
    ball = [_ball(f, (10.5, 10.0)) for f in range(20)]  # parked next to player 1
    states = resolve_possession(players, ball, LABELS, CFG, FPS)
    assert states[0].state == "unknown"  # not yet acquired
    assert states[3].state == "unknown"  # 4 frames < acquire_frames=5
    assert states[10].state == "team_a" and states[10].track_id == 1


def test_transit_keeps_last_touch_then_receiver_acquires():
    players = _still(1, (0.0, 0.0), range(60)) + _still(2, (30.0, 0.0), range(60))
    ball = [_ball(f, (0.5, 0.0)) for f in range(10)]  # at player 1
    # pass in transit: 3 m/frame = 30 m/s, far above max_ball_speed_mps
    ball += [_ball(10 + i, (0.5 + 3.0 * (i + 1), 0.0)) for i in range(9)]
    ball += [_ball(19 + i, (29.8, 0.0)) for i in range(41)]  # settles at player 2
    states = resolve_possession(players, ball, LABELS, CFG, FPS)
    by = {s.frame_idx: s for s in states}
    assert by[9].state == "team_a"
    assert by[15].state == "team_a"  # in transit: last touch holds
    assert by[40].state == "team_b"  # receiver acquired


def test_referee_and_unlabelled_tracks_never_acquire():
    players = _still(3, (10.0, 10.0), range(30)) + _still(99, (10.0, 12.0), range(30))
    ball = [_ball(f, (10.2, 10.0)) for f in range(30)]  # on the referee's toes
    states = resolve_possession(players, ball, LABELS, CFG, FPS)
    assert all(s.state == "unknown" for s in states)


def test_nearest_eligible_player_acquires_past_closer_ineligible_track():
    """Nearest *eligible* player, not nearest overall: a closer referee must not block
    acquisition."""
    players = (
        _still(3, (10.1, 10.0), range(20))  # referee: closest to the ball
        + _still(99, (10.0, 15.0), range(20))  # unlabeled: irrelevant, far away
        + _still(1, (10.5, 10.0), range(20))  # team_a: farther than the referee, but eligible
    )
    ball = [_ball(f, (10.0, 10.0)) for f in range(20)]
    states = resolve_possession(players, ball, LABELS, CFG, FPS)
    assert states[10].state == "team_a" and states[10].track_id == 1


def test_hysteresis_restarts_after_interruption_at_n_minus_one():
    """A candidate interrupted one frame short of acquiring restarts from zero, not
    from where it left off."""
    players = _still(1, (0.0, 0.0), range(20))
    ball = [_ball(f, (0.5, 0.0)) for f in range(5)]  # frames 0-4: builds count to 4
    ball += [_ball(5, (500.0, 0.0))]  # frame 5: huge jump -> fast -> interrupts
    ball += [_ball(6, (0.5, 0.0))]  # frame 6: jumps back -> still fast relative to frame 5
    ball += [_ball(f, (0.5, 0.0)) for f in range(7, 20)]  # frames 7-19: settled again
    states = resolve_possession(players, ball, LABELS, CFG, FPS)
    by = {s.frame_idx: s for s in states}
    assert by[4].state == "unknown"  # count=4 (one short of acquire_frames=5): not yet flipped
    assert by[7].state == "unknown"  # first good frame post-interruption: fresh count=1
    assert by[10].state == "unknown"  # count=4 again post-restart: still not flipped
    assert by[11].state == "team_a" and by[11].track_id == 1  # fresh count reaches 5


def test_unreliable_frame_counts_as_dropout_like_missing():
    """``reliable=False`` with ``xy_m`` present counts as a missing frame for the dropout run."""
    players = _still(1, (10.0, 10.0), range(40))
    ball = [_ball(f, (10.5, 10.0)) for f in range(10)]
    ball += [
        BallPitchState(10 + i, np.array([10.5, 10.0]), "detected", False, False) for i in range(30)
    ]  # xy_m present but unreliable, for 3s at FPS=10 -- must count as a dropout
    states = resolve_possession(players, ball, LABELS, CFG, FPS)
    by = {s.frame_idx: s for s in states}
    assert by[9].state == "team_a"
    assert by[12].state == "team_a"  # short dropout: hold, despite xy_m being present
    assert by[39].state == "unknown"  # long dropout: reset, despite xy_m being present


def test_long_ball_dropout_resets_to_unknown():
    players = _still(1, (10.0, 10.0), range(40))
    ball = [_ball(f, (10.5, 10.0)) for f in range(10)]
    ball += [_ball(10 + i, None) for i in range(30)]  # gone for 3 s at FPS=10
    states = resolve_possession(players, ball, LABELS, CFG, FPS)
    by = {s.frame_idx: s for s in states}
    assert by[9].state == "team_a"
    assert by[12].state == "team_a"  # short dropout: hold (max_unknown_s=0.5 -> 5 frames)
    assert by[39].state == "unknown"


def test_goalkeeper_nearest_is_not_attributed():
    """A keeper's team is not known, so a keeper on the ball is N/A, not a guess."""
    players = _still(4, (5.0, 34.0), range(20))
    ball = [_ball(f, (5.5, 34.0)) for f in range(20)]
    states = resolve_possession(players, ball, LABELS, CFG, FPS)
    assert states[-1].state == "keeper" and states[-1].track_id == 4


def test_outfielder_beside_the_keeper_does_not_take_the_frame():
    """The keeper competes as nearest like anyone else; a closing striker cannot claim it."""
    players = _still(4, (5.0, 34.0), range(20)) + _still(1, (6.8, 34.0), range(20))
    ball = [_ball(f, (5.5, 34.0)) for f in range(20)]
    states = resolve_possession(players, ball, LABELS, CFG, FPS)
    assert states[-1].state == "keeper"


def test_receiver_acquires_after_keeper_distribution():
    players = _still(4, (5.0, 34.0), range(30)) + _still(2, (30.0, 34.0), range(30))
    ball = [_ball(f, (5.5, 34.0)) for f in range(10)]
    ball += [_ball(f, (30.5, 34.0)) for f in range(10, 30)]
    states = resolve_possession(players, ball, LABELS, CFG, FPS)
    by = {s.frame_idx: s for s in states}
    assert by[9].state == "keeper"
    assert by[10].state == "keeper"  # in transit: last touch holds
    assert by[29].state == "team_b" and by[29].track_id == 2


def test_keeper_frames_leave_the_team_shares_untouched():
    from football_tracker.projection.artifacts import PossessionState

    states = [PossessionState(0, "team_a", 1)] * 6 + [PossessionState(0, "team_b", 2)] * 2
    states += [PossessionState(0, "keeper", 4)] * 2
    s = possession_summary(states)
    assert s["team_a"] == 0.75 and s["team_b"] == 0.25
    assert s["keeper_share"] == 0.2 and s["unknown_share"] == 0.0


def test_possession_summary_shares():
    from football_tracker.projection.artifacts import PossessionState

    states = [PossessionState(0, "team_a", 1)] * 6 + [PossessionState(0, "team_b", 2)] * 2
    states += [PossessionState(0, "unknown", -1)] * 2
    s = possession_summary(states)
    assert s["team_a"] == 0.75 and s["team_b"] == 0.25  # of attributed frames
    assert s["unknown_share"] == 0.2 and s["keeper_share"] == 0.0  # of all frames
