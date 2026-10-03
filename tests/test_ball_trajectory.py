"""The ball sub-system's offline half, over synthetic candidates."""

from __future__ import annotations

import numpy as np
import pytest

from football_tracker.config import BallConfig
from football_tracker.homography.record import FrameCalibration
from football_tracker.tracking.ball import BallCandidate
from football_tracker.tracking.ball_trajectory import (
    BallState,
    build_trajectory,
    close_flags,
    cumulative_warps,
    flag_airborne,
    interpolate_gaps,
    solve_path,
)


def _cand(frame_idx: int, x: float, y: float, conf: float = 0.9) -> BallCandidate:
    return BallCandidate(
        frame_idx=frame_idx,
        xy=np.array([x, y], dtype=np.float32),
        wh=np.array([12.0, 12.0], dtype=np.float32),
        confidence=conf,
        source="full",
    )


def _translation(tx: float, ty: float = 0.0) -> np.ndarray:
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]])


class TestCumulativeWarps:
    def test_translations_accumulate(self):
        gmc = {t: _translation(2.0) for t in range(1, 4)}
        warps = cumulative_warps(gmc, 0, 3)
        np.testing.assert_allclose(warps[3][:2, :], _translation(6.0))

    def test_missing_frames_are_identity(self):
        warps = cumulative_warps({}, 0, 5)
        np.testing.assert_allclose(warps[5][:2, :], _translation(0.0))

    def test_span_start_is_identity(self):
        gmc = {1: _translation(2.0)}
        np.testing.assert_allclose(cumulative_warps(gmc, 1, 1)[1][:2, :], _translation(0.0))


class TestSolvePath:
    def test_picks_moving_ball_over_static_low_conf_distractor(self):
        ball = [_cand(t, 100.0 + 5.0 * t, 100.0, conf=0.85) for t in range(20)]
        sock = [_cand(t, 500.0, 300.0, conf=0.5) for t in range(20)]
        path = solve_path(ball + sock, {}, BallConfig())
        assert len(path) == 20
        assert all(c.xy[1] == pytest.approx(100.0) for c in path)

    def test_never_hops_between_distant_candidates(self):
        # Ball and distractor are 400+ px apart: hopping needs speed > max_speed_px.
        ball = [_cand(t, 100.0 + 5.0 * t, 100.0, conf=0.6) for t in range(10)]
        distractor = [_cand(t, 800.0, 600.0, conf=0.9) for t in range(10)]
        path = solve_path(ball + distractor, {}, BallConfig())
        ys = {float(c.xy[1]) for c in path}
        assert len(ys) == 1  # one consistent object, never a mixture

    def test_bridges_a_detection_gap(self):
        seg1 = [_cand(t, 100.0 + 5.0 * t, 100.0) for t in range(10)]
        seg2 = [_cand(t, 100.0 + 5.0 * t, 100.0) for t in range(15, 25)]
        path = solve_path(seg1 + seg2, {}, BallConfig())
        frames = [c.frame_idx for c in path]
        assert frames == list(range(10)) + list(range(15, 25))

    def test_weak_off_track_candidate_not_selected(self):
        ball = [_cand(t, 100.0 + 5.0 * t, 100.0, conf=0.9) for t in range(11)]
        junk = _cand(5, 400.0, 400.0, conf=0.05)
        path = solve_path([*ball, junk], {}, BallConfig())
        assert len(path) == 11
        assert all(c.xy[1] == pytest.approx(100.0) for c in path)

    def test_gmc_compensation_unblocks_camera_pan(self):
        # Ball static in the world; camera pans +10 px/frame, so image positions drift.
        cands = [_cand(t, 100.0 + 10.0 * t, 100.0) for t in range(10)]
        gmc = {t: _translation(10.0) for t in range(1, 10)}
        cfg = BallConfig(max_speed_px=5.0)  # tighter than the raw 10 px/frame drift
        assert len(solve_path(cands, gmc, cfg)) == 10  # compensated speed ~0 passes
        assert len(solve_path(cands, {}, cfg)) < 10  # uncompensated speed is gated

    def test_at_most_one_candidate_per_frame(self):
        doubles = [_cand(t, 100.0, 100.0, conf=0.9) for t in range(5)] + [
            _cand(t, 105.0, 100.0, conf=0.8) for t in range(5)
        ]
        path = solve_path(doubles, {}, BallConfig())
        frames = [c.frame_idx for c in path]
        assert len(frames) == len(set(frames))

    def test_empty_input(self):
        assert solve_path([], {}, BallConfig()) == []

    def test_low_conf_consistent_run_beats_missing(self):
        # The candidate layer is deliberately permissive (roi_conf 0.10); the path
        # economics must accept a motion-consistent low-confidence run rather than
        # declaring the ball missing.
        cands = [_cand(t, 100.0 + 5.0 * t, 100.0, conf=0.2) for t in range(10)]
        path = solve_path(cands, {}, BallConfig())
        assert len(path) == 10

    def test_moving_low_conf_ball_beats_static_equal_conf_distractor(self):
        # The speed cost must not overwhelm confidence: a real moving ball should
        # not lose to a static distractor of *lower* confidence just because the
        # distractor's edges are free.
        ball = [_cand(t, 100.0 + 5.0 * t, 100.0, conf=0.6) for t in range(20)]
        sock = [_cand(t, 500.0, 300.0, conf=0.4) for t in range(20)]
        path = solve_path(ball + sock, {}, BallConfig())
        assert len(path) == 20
        assert all(c.xy[1] == pytest.approx(100.0) for c in path)


def _by_frame(states: list[BallState]) -> dict[int, BallState]:
    return {s.frame_idx: s for s in states}


class TestInterpolateGaps:
    def test_detected_frames_pass_through(self):
        path = [_cand(0, 100.0, 100.0, conf=0.8), _cand(1, 105.0, 100.0, conf=0.7)]
        states = interpolate_gaps(path, (0, 1), {}, BallConfig())
        assert [s.source for s in states] == ["detected", "detected"]
        assert states[0].confidence == pytest.approx(0.8)
        assert states[1].xy == pytest.approx([105.0, 100.0])

    def test_short_gap_filled_linearly(self):
        path = [_cand(0, 100.0, 100.0), _cand(4, 140.0, 100.0)]
        states = _by_frame(interpolate_gaps(path, (0, 4), {}, BallConfig()))
        mid = states[2]
        assert mid.source == "interpolated"
        assert mid.confidence == 0.0
        assert mid.xy == pytest.approx([120.0, 100.0])

    def test_stabilized_fill_under_uneven_camera_pan(self):
        # World-static ball; camera pans +10 px/frame on frames 1-2 only. Its image
        # position is 120 from frame 2 onward — naive linear fill would say 110.
        path = [_cand(0, 100.0, 100.0), _cand(4, 120.0, 100.0)]
        gmc = {1: _translation(10.0), 2: _translation(10.0)}
        states = _by_frame(interpolate_gaps(path, (0, 4), gmc, BallConfig()))
        assert states[2].xy == pytest.approx([120.0, 100.0], abs=1e-6)

    def test_long_gap_left_missing(self):
        path = [_cand(0, 100.0, 100.0), _cand(10, 200.0, 100.0)]
        cfg = BallConfig(max_interp_gap=3)
        states = _by_frame(interpolate_gaps(path, (0, 10), {}, cfg))
        assert states[5].source == "missing"
        assert states[5].xy is None
        assert states[0].source == "detected"
        assert states[10].source == "detected"

    def test_frames_outside_detections_are_missing(self):
        path = [_cand(3, 100.0, 100.0), _cand(5, 110.0, 100.0)]
        states = _by_frame(interpolate_gaps(path, (0, 10), {}, BallConfig()))
        assert states[0].source == "missing"
        assert states[8].source == "missing"
        assert states[4].source == "interpolated"

    def test_covers_every_frame_in_range(self):
        path = [_cand(2, 100.0, 100.0)]
        states = interpolate_gaps(path, (0, 5), {}, BallConfig())
        assert [s.frame_idx for s in states] == [0, 1, 2, 3, 4, 5]


def _detected(frame_idx: int, x: float, y: float) -> BallState:
    return BallState(
        frame_idx=frame_idx,
        xy=np.array([x, y]),
        source="detected",
        airborne=False,
        confidence=0.9,
    )


def _missing(frame_idx: int) -> BallState:
    return BallState(frame_idx=frame_idx, xy=None, source="missing", airborne=False, confidence=0.0)


def _calib_h(frame_idx: int, H: np.ndarray | None) -> FrameCalibration:
    return FrameCalibration(
        frame_idx=frame_idx,
        H=H,
        source="solved" if H is not None else "none",
        low_confidence=False,
        gmc=_translation(0.0),
    )


class TestCloseFlags:
    def test_fills_small_holes(self):
        flags = np.array([True] * 10 + [False] * 3 + [True] * 10)
        closed = close_flags(flags, max_hole=5, min_island=5)
        assert closed.all()

    def test_keeps_large_holes(self):
        flags = np.array([True] * 10 + [False] * 8 + [True] * 10)
        closed = close_flags(flags, max_hole=5, min_island=5)
        assert not closed[14]

    def test_drops_short_islands(self):
        flags = np.array([False] * 10 + [True] * 3 + [False] * 10)
        closed = close_flags(flags, max_hole=2, min_island=5)
        assert not closed.any()


class TestFlagAirborne:
    def _arc_states(self) -> list[BallState]:
        # Gravity-shaped arc in image space: apex at t = 40, curvature 0.2 px/frame².
        return [_detected(t, 100.0 + 5.0 * t, 500.0 - 16.0 * t + 0.2 * t * t) for t in range(81)]

    def test_parabolic_arc_is_flagged(self):
        out = flag_airborne(self._arc_states(), {}, {}, BallConfig(), fps=60.0)
        by = _by_frame(out)
        assert by[40].airborne  # apex
        assert any(s.airborne for s in out)

    def test_linear_ground_roll_not_flagged(self):
        states = [_detected(t, 100.0 + 5.0 * t, 500.0) for t in range(81)]
        # No H → the projected-speed signal is unavailable; the parabola must not fire.
        out = flag_airborne(states, {}, {}, BallConfig(), fps=60.0)
        assert not any(s.airborne for s in out)

    def test_fast_projected_speed_flags_when_h_available(self):
        # H = identity → px ≈ m; 5 px/frame @ 60 fps = 300 m/s: far past any ground ball.
        states = [_detected(t, 100.0 + 5.0 * t, 500.0) for t in range(60)]
        calib = {t: _calib_h(t, np.eye(3)) for t in range(60)}
        out = flag_airborne(states, {}, calib, BallConfig(), fps=60.0)
        assert any(s.airborne for s in out)

    def test_slow_projected_speed_not_flagged(self):
        # 0.3 px/frame @ 60 fps = 18 m/s < the 25 m/s launch threshold.
        states = [_detected(t, 100.0 + 0.3 * t, 500.0) for t in range(60)]
        calib = {t: _calib_h(t, np.eye(3)) for t in range(60)}
        out = flag_airborne(states, {}, calib, BallConfig(), fps=60.0)
        assert not any(s.airborne for s in out)

    def test_missing_frames_never_flagged(self):
        states = [
            BallState(frame_idx=t, xy=None, source="missing", airborne=False, confidence=0.0)
            for t in range(40)
        ]
        out = flag_airborne(states, {}, {}, BallConfig(), fps=60.0)
        assert not any(s.airborne for s in out)

    def test_fast_interpolated_fill_not_flagged(self):
        # A long detection gap filled by a straight fast line (36 px/frame) is an
        # interpolation artifact, not a launched ball — the airborne heuristic must
        # ignore it.
        states = []
        for t in range(60):
            if t in (0, 59):
                src, xy = "detected", np.array([100.0 + 36.0 * t, 500.0])
            else:
                src, xy = "interpolated", np.array([100.0 + 36.0 * t, 500.0])
            states.append(BallState(frame_idx=t, xy=xy, source=src, airborne=False, confidence=0.9))
        calib = {t: _calib_h(t, np.eye(3)) for t in range(60)}
        out = flag_airborne(states, {}, calib, BallConfig(), fps=60.0)
        assert not any(s.airborne for s in out)

    def test_fast_detected_jump_across_large_gap_not_flagged(self):
        # Two real detections 23 frames apart, 840 px apart (ball undetected between).
        # The apparent speed is a gap-average — could be a fast ground ball far→near,
        # not necessarily airborne — so it must NOT flag.
        states = []
        for t in range(60):
            if t in (0, 23):
                states.append(
                    BallState(
                        frame_idx=t,
                        xy=np.array([100.0 + 36.5 * t, 700.0 + 32.0 * t]),
                        source="detected",
                        airborne=False,
                        confidence=0.8,
                    )
                )
            else:
                states.append(_missing(t))
        calib = {t: _calib_h(t, np.eye(3)) for t in range(60)}
        out = flag_airborne(states, {}, calib, BallConfig(), fps=60.0)
        assert not any(s.airborne for s in out)


class TestBuildTrajectory:
    def test_end_to_end_detect_interpolate_miss(self):
        # Moving ball seen on frames 0-19 (with a short 3-frame dropout) and again
        # 55-69. The 35-frame outage is beyond max_interp_gap, so it stays missing —
        # attributing it to a nearby player is possession's job, not the tracker's.
        seen = [t for t in list(range(20)) + list(range(55, 70)) if t not in (7, 8, 9)]
        cands = [_cand(t, 100.0 + 2.0 * t, 100.0, conf=0.85) for t in seen]
        stream = [_calib_h(t, None) for t in range(70)]
        states = build_trajectory(cands, (0, 69), stream, BallConfig(), fps=60.0)
        by = _by_frame(states)
        assert len(states) == 70
        assert by[5].source == "detected"
        assert by[8].source == "interpolated"  # short dropout filled
        assert by[8].xy == pytest.approx([116.0, 100.0], abs=1.0)
        assert by[30].source == "missing"  # long outage is honestly unknown
        assert by[60].source == "detected"
        assert not any(s.airborne for s in states)  # flat trajectory, no H
