"""The calibration gate: the measurement, its enforcement, and how it reports failure."""

from __future__ import annotations

import numpy as np
import pytest

from football_tracker.homography.gate import check_gate, region_breakdown


class TestCheckGate:
    def test_passes_under_the_threshold(self):
        assert check_gate(0.44, gate_median_m=0.5) == []

    def test_fails_over_the_threshold_and_names_both_numbers(self):
        problems = check_gate(0.937, gate_median_m=0.5)
        assert len(problems) == 1
        assert "0.937" in problems[0]
        assert "0.5" in problems[0]

    def test_passes_exactly_at_the_threshold(self):
        assert check_gate(0.5, gate_median_m=0.5) == []

    def test_is_skipped_when_no_threshold_is_given(self):
        assert check_gate(9.99, gate_median_m=None) == []

    def test_a_non_finite_median_fails_rather_than_silently_passing(self):
        """nan > 0.5 is False in IEEE terms, which would read as a pass."""
        problems = check_gate(float("nan"), gate_median_m=0.5)
        assert len(problems) == 1
        assert "nan" in problems[0].lower()


class TestRegionBreakdown:
    def test_splits_the_pitch_into_thirds(self):
        errors = np.array([0.1, 0.2, 0.5, 0.6, 1.0, 1.2])
        xs = np.array([5.0, 20.0, 40.0, 60.0, 90.0, 100.0])
        out = region_breakdown(errors, xs, pitch_length_m=105.0)
        assert out["near_third"]["n"] == 2
        assert out["middle_third"]["n"] == 2
        assert out["far_third"]["n"] == 2
        assert out["near_third"]["median_m"] == pytest.approx(0.15)
        assert out["middle_third"]["median_m"] == pytest.approx(0.55)
        assert out["far_third"]["median_m"] == pytest.approx(1.1)

    def test_every_point_lands_in_exactly_one_region(self):
        rng = np.random.default_rng(0)
        xs = rng.uniform(0.0, 105.0, size=200)
        errors = rng.uniform(0.0, 3.0, size=200)
        out = region_breakdown(errors, xs, pitch_length_m=105.0)
        assert sum(out[r]["n"] for r in out) == 200

    def test_boundaries_are_assigned_consistently(self):
        """Exact third boundaries go to the higher region."""
        xs = np.array([35.0, 70.0])
        out = region_breakdown(np.array([1.0, 2.0]), xs, pitch_length_m=105.0)
        assert out["near_third"]["n"] == 0
        assert out["middle_third"]["n"] == 1
        assert out["far_third"]["n"] == 1

    def test_reports_an_empty_region_without_dividing_by_zero(self):
        out = region_breakdown(np.array([0.1, 0.2]), np.array([1.0, 2.0]), pitch_length_m=105.0)
        assert out["far_third"]["n"] == 0
        assert np.isnan(out["far_third"]["median_m"])
        assert np.isnan(out["far_third"]["p90_m"])

    def test_scales_with_a_non_standard_pitch_length(self):
        """x=65 is the middle third of a 105 m pitch but the far third of a 90 m one."""
        errors = np.array([1.0])
        xs = np.array([65.0])
        assert region_breakdown(errors, xs, pitch_length_m=105.0)["middle_third"]["n"] == 1
        assert region_breakdown(errors, xs, pitch_length_m=90.0)["far_third"]["n"] == 1

    def test_handles_an_empty_measurement_set(self):
        out = region_breakdown(np.array([]), np.array([]), pitch_length_m=105.0)
        assert all(out[r]["n"] == 0 for r in out)


SPEC_LENGTH, SPEC_WIDTH = 105.0, 68.0


def _h_for(dx: float = 0.0) -> np.ndarray:
    """An image→pitch H for a 1280x720 view, optionally shifted ``dx`` px."""
    import cv2

    image_corners = np.array(
        [(100 + dx, 80), (1180 + dx, 100), (1260 + dx, 700), (30 + dx, 690)], dtype=np.float32
    )
    pitch_corners = np.array(
        [(0.0, 0.0), (SPEC_LENGTH, 0.0), (SPEC_LENGTH, SPEC_WIDTH), (0.0, SPEC_WIDTH)],
        dtype=np.float32,
    )
    return np.linalg.inv(cv2.getPerspectiveTransform(pitch_corners, image_corners))


def _anchor(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from football_tracker.homography.pitch import PitchSpec
    from football_tracker.homography.solve import project

    pitch_pts = PitchSpec().keypoints[:12]
    return project(np.linalg.inv(H), pitch_pts), pitch_pts


def _anchor_subset(H: np.ndarray, indices: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """An anchor built from just ``indices`` of the pitch keypoints, spread but sparse."""
    from football_tracker.homography.pitch import PitchSpec
    from football_tracker.homography.solve import project

    pitch_pts = PitchSpec().keypoints[list(indices)]
    return project(np.linalg.inv(H), pitch_pts), pitch_pts


def _still_gmc(frames: range) -> dict[int, np.ndarray]:
    """A GMC stream for a locked-off camera: identity motion everywhere."""
    return {f: np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) for f in frames}


class TestHoldout:
    """Held-out accuracy for the bidirectional stream.

    The camera is still and anchor 0 is deliberately wrong by 20 px, so propagating
    from it alone would hand frame 10 the whole error. The scorer must reach past the
    held-out frame to anchor 20 as well.
    """

    def setup_method(self):
        self.gmc = _still_gmc(range(0, 21))
        self.anchors = {0: _anchor(_h_for(20.0)), 10: _anchor(_h_for()), 20: _anchor(_h_for())}

    def test_uses_the_following_anchor_not_only_the_preceding_one(self):
        from football_tracker.homography.gate import holdout_errors

        _, near, _, _ = holdout_errors(self.anchors, self.gmc)
        far = {**self.anchors, 20: _anchor(_h_for(40.0))}
        _, moved, _, _ = holdout_errors(far, self.gmc)
        # Only the anchor *after* the held-out frame changed; if it were ignored,
        # the two error sets would be identical.
        assert not np.allclose(np.median(near), np.median(moved))

    def test_the_held_out_anchor_is_never_used_to_fit_its_own_score(self):
        """Corrupting only the held-out anchor's *image* clicks must not improve it."""
        from football_tracker.homography.gate import holdout_errors

        _, base, _, _ = holdout_errors(self.anchors, self.gmc)
        nudged = dict(self.anchors)
        img, pitch = nudged[10]
        nudged[10] = (img + 5.0, pitch)
        _, after, _, _ = holdout_errors(nudged, self.gmc)
        # Its labels moved 5 px away from a calibration that did not change, so
        # its own error must grow -- proof the fit never saw them.
        assert np.median(after) > np.median(base)


class TestAnchorAcceptance:
    """The harness must accept exactly the anchors the calibrator accepts.

    ``fuse`` applies its plausibility gate only when told the frame shape; without it
    the holdout can score from an anchor the shipped stream threw away.
    """

    FRAME_SHAPE = (720, 1280)

    def setup_method(self):
        self.gmc = _still_gmc(range(0, 21))
        # Anchor 0 is wrong by 20 px; anchor 20 is correct but carries only five points,
        # one short of the six inliers `fuse.MIN_INLIERS` demands.
        self.anchors = {
            0: _anchor(_h_for(20.0)),
            10: _anchor(_h_for()),
            20: _anchor_subset(_h_for(), (0, 5, 13, 16, 24)),
        }

    def _frame_10(self, **kwargs):
        from football_tracker.homography.gate import holdout_errors

        _, _, _, records = holdout_errors(self.anchors, self.gmc, **kwargs)
        return next(r for r in records if r["held_frame"] == 10)

    def test_a_thin_anchor_is_ignored_when_the_frame_shape_is_known(self):
        """With the shape, anchor 20 is rejected and frame 10 falls back to the bad
        anchor 0 alone -- so its error must be worse than when 20 was wrongly trusted."""
        without = self._frame_10()
        with_shape = self._frame_10(frame_shape=self.FRAME_SHAPE)
        assert with_shape["median_m"] > without["median_m"]

    def test_a_sound_anchor_still_passes_the_plausibility_gate(self):
        """The gate must reject the thin anchor and nothing else: a shape that threw out
        every anchor would leave the harness scoring nothing and raising instead."""
        from football_tracker.homography.gate import holdout_errors

        good = {0: _anchor(_h_for(20.0)), 10: _anchor(_h_for()), 20: _anchor(_h_for())}
        _, m, _, records = holdout_errors(good, self.gmc, frame_shape=self.FRAME_SHAPE)
        assert len(records) == 2
        assert np.isfinite(m).all()


class TestUnmeasurableReason:
    """Three settings can leave a frame uncovered; the message must name the one that did."""

    def _gmc(self, frames):
        return dict.fromkeys(frames, np.eye(2, 3))

    def test_a_stream_ending_before_the_anchors_names_the_range_settings(self):
        from football_tracker.homography.gate import _unmeasurable_reason

        msg = _unmeasurable_reason(16, 0, 20, self._gmc(range(16)))
        assert "io.frame_stop" in msg
        assert "ends at 15" in msg

    def test_a_stream_starting_after_the_anchors_names_frame_start(self):
        from football_tracker.homography.gate import _unmeasurable_reason

        msg = _unmeasurable_reason(0, 0, 20, self._gmc(range(8, 21)))
        assert "io.frame_start" in msg
        assert "starts at 8" in msg

    def test_a_gap_inside_the_streams_own_range_names_the_stride(self):
        from football_tracker.homography.gate import _unmeasurable_reason

        msg = _unmeasurable_reason(1, 0, 20, self._gmc(range(0, 21, 2)))
        assert "io.frame_stride" in msg

    def test_an_empty_stream_says_so_rather_than_indexing_nothing(self):
        from football_tracker.homography.gate import _unmeasurable_reason

        assert "no frames" in _unmeasurable_reason(0, 0, 20, {})


class TestGateFailureReporting:
    """A failure that stops the measurement must arrive as GATE FAIL, like a failed
    threshold does -- a traceback names the line that noticed, not the setting to fix."""

    def _write_anchors(self, path, frames):
        import json

        image_pts, pitch_pts = _anchor(_h_for())
        points = [
            {"image": [float(i[0]), float(i[1])], "pitch": [float(p[0]), float(p[1])]}
            for i, p in zip(image_pts, pitch_pts, strict=True)
        ]
        path.write_text(json.dumps({"frames": {str(f): points for f in frames}}))

    def _write_calibration(self, path, frames):
        from football_tracker.homography.record import FrameCalibration
        from football_tracker.homography.store import save_calibration

        gmc = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        save_calibration(
            path,
            [
                FrameCalibration(
                    frame_idx=f, H=None, source="propagated", low_confidence=False, gmc=gmc
                )
                for f in frames
            ],
            frame_shape=(720, 1280),
        )

    def test_a_stream_too_short_for_its_anchors_exits_as_a_gate_failure(
        self, tmp_path, monkeypatch, capsys
    ):
        import sys

        from football_tracker.homography.gate import main

        anchors, calibration = tmp_path / "anchors.json", tmp_path / "calibration.npz"
        self._write_anchors(anchors, (0, 10, 20))
        self._write_calibration(calibration, range(16))  # labelled to 20, calibrated to 15

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "gate_calibration.py",
                "--calibration",
                str(calibration),
                "--anchors",
                str(anchors),
                "--gate-median-m",
                "0.5",
            ],
        )
        with pytest.raises(SystemExit) as exit_info:
            main()

        assert exit_info.value.code == 1
        err = capsys.readouterr().err
        assert "GATE FAIL" in err
        assert "io.frame_stop" in err  # the cause, not a guess at the cause

    def _write_metrics(self, path, fingerprint):
        import json

        path.write_text(json.dumps({"anchors_sha256_12": fingerprint}))

    def test_anchors_the_calibration_was_not_built_from_are_refused(
        self, tmp_path, monkeypatch, capsys
    ):
        """Provable, so never waived: scoring these would certify a stream against a
        different clip's labels, which no downstream stage can detect."""
        import sys

        from football_tracker.homography.gate import main

        anchors, calibration = tmp_path / "anchors.json", tmp_path / "calibration.npz"
        self._write_anchors(anchors, (0, 10, 20))
        self._write_calibration(calibration, range(21))
        self._write_metrics(tmp_path / "calibration_metrics.json", "000000000000")

        monkeypatch.setattr(
            sys,
            "argv",
            ["gate_calibration.py", "--calibration", str(calibration), "--anchors", str(anchors)],
        )
        with pytest.raises(SystemExit) as exit_info:
            main()

        assert exit_info.value.code == 1
        err = capsys.readouterr().err
        assert "GATE FAIL" in err
        assert "not the ones the calibration was built from" in err

    def test_matching_anchors_pass_the_provenance_check(self, tmp_path, monkeypatch, capsys):
        import sys

        from football_tracker.homography.gate import main
        from football_tracker.homography.provenance import anchor_fingerprint

        anchors, calibration = tmp_path / "anchors.json", tmp_path / "calibration.npz"
        self._write_anchors(anchors, (0, 10, 20))
        self._write_calibration(calibration, range(21))
        self._write_metrics(tmp_path / "calibration_metrics.json", anchor_fingerprint(anchors))

        monkeypatch.setattr(
            sys,
            "argv",
            ["gate_calibration.py", "--calibration", str(calibration), "--anchors", str(anchors)],
        )
        main()  # no SystemExit
        assert "GATE FAIL" not in capsys.readouterr().err

    def test_a_missing_fingerprint_is_noted_rather_than_failing(
        self, tmp_path, monkeypatch, capsys
    ):
        """The check cannot be made on a hand-made or older stream; say so and continue."""
        import sys

        from football_tracker.homography.gate import main

        anchors, calibration = tmp_path / "anchors.json", tmp_path / "calibration.npz"
        self._write_anchors(anchors, (0, 10, 20))
        self._write_calibration(calibration, range(21))
        # no calibration_metrics.json at all

        monkeypatch.setattr(
            sys,
            "argv",
            ["gate_calibration.py", "--calibration", str(calibration), "--anchors", str(anchors)],
        )
        main()
        err = capsys.readouterr().err
        assert "no anchors fingerprint" in err
        assert "GATE FAIL" not in err
