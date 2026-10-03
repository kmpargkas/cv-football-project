"""Calibration quality: per-anchor soundness and stream continuity, over arrays only."""

from __future__ import annotations

import numpy as np

from football_tracker.homography.pitch import PitchSpec
from football_tracker.homography.quality import END_ZONE_M, anchor_quality

SPEC = PitchSpec()


def _square_correspondence(n_good: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """A clean pitch->image mapping under a known homography."""
    rng = np.random.default_rng(0)
    pitch = np.stack([rng.uniform(0, 105, n_good), rng.uniform(0, 68, n_good)], axis=1)
    H = np.array([[20.0, 0.0, 100.0], [0.0, 20.0, 50.0], [0.0, 0.0, 1.0]])
    hom = np.concatenate([pitch, np.ones((len(pitch), 1))], axis=1) @ H.T
    img = hom[:, :2] / hom[:, 2:]
    return img, pitch


def _correspondence(pitch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact image points for the given pitch points, under a known homography."""
    H = np.array([[20.0, 0.0, 100.0], [0.0, 20.0, 50.0], [0.0, 0.0, 1.0]])
    hom = np.concatenate([pitch, np.ones((len(pitch), 1))], axis=1) @ H.T
    return hom[:, :2] / hom[:, 2:], pitch


def _anchor_reaching(length: float, furthest_x: float) -> tuple[np.ndarray, np.ndarray]:
    """A solvable anchor whose furthest click sits at ``furthest_x``."""
    return _correspondence(
        np.array(
            [
                [0.0, 0.0],
                [furthest_x, 34.0],
                [length / 2, 68.0],
                [length / 2, 0.0],
                [10.0, 60.0],
                [5.0, 10.0],
            ]
        )
    )


class TestAnchorQuality:
    def test_a_clean_anchor_has_full_inlier_ratio_and_tiny_residual(self):
        img, pitch = _square_correspondence()
        q = anchor_quality(img, pitch, SPEC, ransac_thresh_px=10.0)
        assert q.n_points == 8
        assert q.n_inliers == 8
        assert q.inlier_ratio == 1.0
        assert q.median_residual_m < 1e-3  # sub-millimetre on the pitch = exact
        assert q.outlier_indices == []

    def test_a_displaced_point_is_reported_as_an_outlier(self):
        img, pitch = _square_correspondence()
        img[3] += np.array([200.0, 200.0])  # one badly-clicked feature
        q = anchor_quality(img, pitch, SPEC, ransac_thresh_px=10.0)
        assert 3 in q.outlier_indices
        assert q.inlier_ratio < 1.0

    def test_outliers_are_reported_worst_first(self):
        img, pitch = _square_correspondence()
        img[1] += np.array([60.0, 0.0])
        img[5] += np.array([300.0, 0.0])
        q = anchor_quality(img, pitch, SPEC, ransac_thresh_px=10.0)
        assert q.outlier_indices[0] == 5  # the worse one leads

    def test_x_range_and_end_reach_are_reported(self):
        img, pitch = _square_correspondence()
        pitch[0] = [0.0, 34.0]
        pitch[1] = [105.0, 34.0]
        q = anchor_quality(img, pitch, SPEC, ransac_thresh_px=1e6)
        assert q.x_min == 0.0
        assert q.x_max == 105.0
        assert q.reaches_left is True
        assert q.reaches_right is True

    def test_an_anchor_short_of_both_ends_reports_neither(self):
        img, pitch = _square_correspondence()
        pitch[:, 0] = np.linspace(16.5, 88.5, len(pitch))
        q = anchor_quality(img, pitch, SPEC, ransac_thresh_px=1e6)
        assert q.reaches_left is False
        assert q.reaches_right is False

    def test_too_few_points_returns_an_unsolvable_marker_rather_than_raising(self):
        q = anchor_quality(np.zeros((3, 2)), np.zeros((3, 2)), SPEC, ransac_thresh_px=10.0)
        assert q.n_inliers == 0
        assert np.isnan(q.median_residual_m)


class TestEndZoneScalesWithThePitch:
    """``reaches_right`` is measured from the goal line, not from a fixed x."""

    def test_a_click_inside_the_goal_area_reaches_it_at_any_length(self):
        for length in (100.0, 105.0, 110.0):
            img, pitch = _anchor_reaching(length, length - END_ZONE_M)
            q = anchor_quality(img, pitch, PitchSpec(length=length), ransac_thresh_px=10.0)
            assert q.reaches_right, f"{length} m pitch"

    def test_a_click_only_as_far_as_the_penalty_box_line_does_not(self):
        for length in (100.0, 105.0, 110.0):
            img, pitch = _anchor_reaching(length, length - 16.5)
            q = anchor_quality(img, pitch, PitchSpec(length=length), ransac_thresh_px=10.0)
            assert not q.reaches_right, f"{length} m pitch"

    def test_the_same_click_reads_differently_on_a_longer_pitch(self):
        """x=94 is inside the goal area on a 105 m pitch and 16 m short of it on a 110 m one."""
        img, pitch = _anchor_reaching(110.0, 94.0)
        assert anchor_quality(
            img, pitch, PitchSpec(length=105.0), ransac_thresh_px=10.0
        ).reaches_right
        assert not anchor_quality(
            img, pitch, PitchSpec(length=110.0), ransac_thresh_px=10.0
        ).reaches_right

    def test_the_left_end_zone_is_measured_from_zero(self):
        """The left goal line is always x=0, so that half needs no scaling."""
        img, pitch = _anchor_reaching(105.0, 50.0)
        assert anchor_quality(img, pitch, SPEC, ransac_thresh_px=10.0).reaches_left


class TestStreamJitter:
    """Continuity, not accuracy: does the projected pitch hold still between frames?

    Propagation reproduces the camera motion exactly, so it must score zero; only
    forced frames -- anchors and the blend around them -- can register.
    """

    PAN = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 0.0]])  # 3 px/frame of camera pan

    def _h(self, dx: float = 0.0) -> np.ndarray:
        import cv2

        image_corners = np.array(
            [(100 + dx, 80), (1180 + dx, 100), (1260 + dx, 700), (30 + dx, 690)],
            dtype=np.float32,
        )
        pitch_corners = SPEC.boundary.astype(np.float32)
        return np.linalg.inv(cv2.getPerspectiveTransform(pitch_corners, image_corners))

    def _propagated(self, n: int = 10) -> list:
        from football_tracker.homography.propagate import FrameMotion, propagate_h
        from football_tracker.homography.record import FrameCalibration

        stream, H = [], self._h()
        for f in range(n):
            if f:
                H = propagate_h(H, FrameMotion(affine=self.PAN))
            stream.append(
                FrameCalibration(
                    frame_idx=f,
                    H=H,
                    source="propagated",
                    low_confidence=False,
                    gmc=self.PAN,
                )
            )
        return stream

    def test_pure_propagation_scores_exactly_zero(self):
        """Not approximately -- propagate_h applies the very motion this compensates for."""
        from football_tracker.homography.quality import stream_jitter

        jitter = stream_jitter(self._propagated(), SPEC)
        assert jitter.n_pairs == 9
        np.testing.assert_allclose(jitter.residual_px, 0.0, atol=1e-9)
        assert jitter.worst_frames == []

    def test_a_disagreeing_anchor_registers_and_is_named(self):
        """The point of carrying frame indices: 'max 720 px' cannot be acted on, but
        'frame 5 jumps' sends the annotator to the anchor that caused it."""
        from football_tracker.homography.quality import stream_jitter
        from football_tracker.homography.record import FrameCalibration

        stream = self._propagated()
        stream[5] = FrameCalibration(
            frame_idx=5,
            H=self._h(12.0),
            source="solved",
            low_confidence=False,
            gmc=self.PAN,
        )
        jitter = stream_jitter(stream, SPEC)
        assert jitter.max_px > 1.0
        # Entering the snap and leaving it both register.
        assert set(jitter.worst_frames) == {5, 6}
        assert jitter.worst_frames[0] in (5, 6)

    def test_the_median_alone_would_miss_a_seam(self):
        """Jitter is zero almost everywhere by construction, so a median reads as clean
        however bad the spike is. worst_frames is what makes the metric usable."""
        from football_tracker.homography.quality import stream_jitter
        from football_tracker.homography.record import FrameCalibration

        stream = self._propagated()
        stream[5] = FrameCalibration(
            frame_idx=5, H=self._h(12.0), source="solved", low_confidence=False, gmc=self.PAN
        )
        jitter = stream_jitter(stream, SPEC)
        assert jitter.median_px < 1e-6  # clean
        assert jitter.worst_frames  # ...but not clean

    def test_an_uncalibrated_stretch_breaks_the_chain_rather_than_bridging_it(self):
        """A pair spanning a gap would blame one frame for a whole gap's movement."""
        from football_tracker.homography.quality import stream_jitter
        from football_tracker.homography.record import FrameCalibration

        stream = self._propagated()
        stream[4] = FrameCalibration(
            frame_idx=4, H=None, source="none", low_confidence=True, gmc=self.PAN
        )
        jitter = stream_jitter(stream, SPEC)
        # 10 frames, one uncalibrated: pairs (0,1)..(2,3) and (5,6)..(8,9) survive; the
        # pairs touching frame 4 do not, and 3->5 is never bridged.
        assert 4 not in jitter.frame_idx
        assert 5 not in jitter.frame_idx
        np.testing.assert_allclose(jitter.residual_px, 0.0, atol=1e-9)

    def test_a_stream_with_no_measurable_pair_reports_nothing_rather_than_raising(self):
        from football_tracker.homography.quality import stream_jitter

        assert stream_jitter([], SPEC).n_pairs == 0
        assert np.isnan(stream_jitter([], SPEC).median_px)
        assert stream_jitter(self._propagated(1), SPEC).worst_frames == []
