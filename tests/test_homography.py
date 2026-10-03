"""The homography stack (pitch, solve, propagate, fuse, mask, store, render), all synthetic."""

import json
import warnings

import cv2
import numpy as np
import pytest

from football_tracker.homography.mask import foot_points, on_pitch, pitch_mask
from football_tracker.homography.pitch import PitchSpec, default_control_points
from football_tracker.homography.propagate import (
    FrameMotion,
    MotionEstimator,
    propagate_h,
)
from football_tracker.homography.solve import (
    HomographySolution,
    plausible,
    project,
    solve_homography,
    validate,
)

SPEC = PitchSpec()
FRAME_SHAPE = (720, 1280)  # (h, w)


def make_ground_truth_h() -> np.ndarray:
    """A camera-plausible image→pitch homography for a 1280×720 frame."""
    image_corners = np.array([(100, 80), (1180, 100), (1260, 700), (30, 690)], dtype=np.float32)
    pitch_corners = SPEC.boundary.astype(np.float32)
    H_pitch2img = cv2.getPerspectiveTransform(pitch_corners, image_corners)
    return np.linalg.inv(H_pitch2img)


class TestPitchSpec:
    def test_keypoint_layout(self):
        kp = SPEC.keypoints
        assert kp.shape == (32, 2)
        assert kp[:, 0].min() == 0.0 and kp[:, 0].max() == SPEC.length
        assert kp[:, 1].min() == 0.0 and kp[:, 1].max() == SPEC.width
        # Left/right mirror symmetry: vertex i ↔ mirrored partner exists.
        mirrored = np.column_stack([SPEC.length - kp[:, 0], kp[:, 1]])
        for m in mirrored:
            assert np.min(np.linalg.norm(kp - m, axis=1)) < 1e-9

    def test_named_vertices(self):
        kp = SPEC.keypoints
        np.testing.assert_allclose(kp[0], (0, 0))  # vertex 1: corner
        np.testing.assert_allclose(kp[8], (11.0, 34.0))  # vertex 9: penalty spot
        np.testing.assert_allclose(kp[21], (94.0, 34.0))  # vertex 22: other spot
        np.testing.assert_allclose(kp[30], (52.5 - 9.15, 34.0))  # vertex 31
        np.testing.assert_allclose(kp[29], (105.0, 68.0))  # vertex 30: far corner

    def test_edges_are_valid_indices(self):
        for a, b in SPEC.edges:
            assert 0 <= a < 32 and 0 <= b < 32 and a != b

    def test_line_segments_cover_circle_and_arcs(self):
        segments = SPEC.line_segments()
        # 33 straight edges + centre circle + 2 penalty arcs
        assert len(segments) == 36
        for seg in segments:
            assert np.isfinite(seg).all()


class TestSolve:
    def test_project_keeps_two_columns_on_empty_input(self):
        """``cv2.perspectiveTransform`` returns None rather than an empty array, and
        callers index the result by column. A frame with no detections is ordinary."""
        out = project(np.eye(3), np.zeros((0, 2)))
        assert out.shape == (0, 2)
        assert out[:, 0].shape == (0,)

    def test_plausible_rejects_a_frame_straddling_the_horizon(self):
        """Corners either side of the horizon fold through infinity. Dehomogenising them
        gives finite but meaningless coordinates whose area can land inside the accepted
        band, so checking finiteness alone would let the mapping through."""
        from football_tracker.homography.solve import _project_homogeneous

        height, width = FRAME_SHAPE
        H = make_ground_truth_h().copy()
        H[2, 1] = -2.0 * H[2, 2] / height  # put the horizon across mid-frame
        solution = HomographySolution(H=H, inliers=9, total=9, reproj_err_px=1.0)
        frame_corners = np.array(
            [(0, 0), (width, 0), (width, height), (0, height)], dtype=np.float64
        )
        # The pitch leg still passes, so the rejection has to come from the frame leg.
        assert _project_homogeneous(solution.H_inv, SPEC.boundary) is not None
        assert _project_homogeneous(solution.H, frame_corners) is None
        assert not plausible(solution, SPEC.boundary, FRAME_SHAPE)

    def test_recovers_known_homography(self):
        H_true = make_ground_truth_h()
        image_pts = project(np.linalg.inv(H_true), SPEC.keypoints)
        solution = solve_homography(image_pts, SPEC.keypoints)
        assert solution is not None
        np.testing.assert_allclose(project(solution.H, image_pts), SPEC.keypoints, atol=1e-6)
        assert solution.inliers == 32
        assert solution.reproj_err_px < 0.1

    def test_robust_to_outliers_and_noise(self):
        rng = np.random.default_rng(42)
        H_true = make_ground_truth_h()
        image_pts = project(np.linalg.inv(H_true), SPEC.keypoints)
        image_pts += rng.normal(0, 0.5, image_pts.shape)  # px noise
        image_pts[3] += 250.0  # gross outliers
        image_pts[17] -= 180.0
        solution = solve_homography(image_pts, SPEC.keypoints)
        assert solution is not None
        assert solution.inliers >= 25
        # Outliers rejected: mapping stays accurate to a few cm on the pitch.
        clean = project(np.linalg.inv(H_true), SPEC.keypoints[8:9])
        np.testing.assert_allclose(project(solution.H, clean), SPEC.keypoints[8:9], atol=0.2)

    def test_too_few_points(self):
        assert solve_homography(np.zeros((3, 2)), SPEC.keypoints[:3]) is None

    def test_validate_accepts_good_and_rejects_degenerate(self):
        H_true = make_ground_truth_h()
        image_pts = project(np.linalg.inv(H_true), SPEC.keypoints)
        good = solve_homography(image_pts, SPEC.keypoints)
        assert validate(good, SPEC.boundary, FRAME_SHAPE)
        assert not validate(None, SPEC.boundary, FRAME_SHAPE)
        # Collapsed H: everything maps to (nearly) one point.
        collapsed = HomographySolution(
            H=np.array([[1e-9, 0, 0], [0, 1e-9, 0], [0, 0, 1.0]]),
            inliers=32,
            total=32,
            reproj_err_px=0.0,
        )
        assert not plausible(collapsed, SPEC.boundary, FRAME_SHAPE)

    def test_validate_rejects_high_error(self):
        H_true = make_ground_truth_h()
        image_pts = project(np.linalg.inv(H_true), SPEC.keypoints)
        solution = solve_homography(image_pts, SPEC.keypoints)
        assert not validate(solution, SPEC.boundary, FRAME_SHAPE, max_reproj_err_px=-1.0)


def textured_frame(rng: np.random.Generator) -> np.ndarray:
    """A feature-rich grayscale frame for optical-flow tests."""
    frame = rng.integers(0, 255, size=(360, 640), dtype=np.uint8)
    return cv2.GaussianBlur(frame, (5, 5), 0)


class TestPropagate:
    @pytest.mark.parametrize(
        "dx,dy,scale,angle_deg",
        [(6.0, -3.0, 1.0, 0.0), (2.0, 1.0, 1.02, 0.4)],
    )
    def test_recovers_similarity_motion(self, dx, dy, scale, angle_deg):
        rng = np.random.default_rng(7)
        prev = textured_frame(rng)
        M = cv2.getRotationMatrix2D((320, 180), angle_deg, scale)
        M[:, 2] += (dx, dy)
        curr = cv2.warpAffine(prev, M, (640, 360))
        est = MotionEstimator(downscale=1.0)
        assert est.step(prev) is None  # first frame
        motion = est.step(curr)
        assert motion is not None
        recovered_scale = np.hypot(motion.affine[0, 0], motion.affine[1, 0])
        assert abs(recovered_scale - scale) < 0.01
        # Check a central point maps consistently (borders lose flow support).
        pt = np.array([320.0, 180.0, 1.0])
        np.testing.assert_allclose(motion.affine @ pt, M @ pt, atol=1.5)

    def test_propagate_h_composition(self):
        H_prev = make_ground_truth_h()
        affine = np.array([[1.01, -0.002, 5.0], [0.002, 1.01, -3.0]])
        motion = FrameMotion(affine=affine)
        H_curr = propagate_h(H_prev, motion)
        # A pitch point seen at x_prev appears at A·x_prev now; the propagated
        # H must send it back to the same pitch coordinates.
        image_prev = project(np.linalg.inv(H_prev), SPEC.keypoints[:5])
        image_curr = project(motion.as_homography, image_prev)
        np.testing.assert_allclose(project(H_curr, image_curr), SPEC.keypoints[:5], atol=1e-8)


class TestMask:
    def test_foot_points(self):
        boxes = np.array([[10, 20, 30, 60], [100, 0, 140, 80]])
        np.testing.assert_allclose(foot_points(boxes), [[20, 60], [120, 80]])

    def test_on_pitch_partitions_points(self):
        H = make_ground_truth_h()
        H_inv = np.linalg.inv(H)
        inside = project(H_inv, np.array([[52.5, 34.0], [1.0, 1.0]]))  # centre, corner
        outside = project(H_inv, np.array([[52.5, 90.0], [-30.0, 34.0]]))
        pts = np.vstack([inside, outside])
        np.testing.assert_array_equal(
            on_pitch(pts, H, SPEC, margin_m=1.0), [True, True, False, False]
        )

    def test_pitch_mask_covers_pitch_only(self):
        H = make_ground_truth_h()
        mask = pitch_mask(H, FRAME_SHAPE, SPEC, margin_m=0.0)
        assert mask.shape == FRAME_SHAPE
        H_inv = np.linalg.inv(H)
        centre = project(H_inv, np.array([[52.5, 34.0]]))[0].astype(int)
        assert mask[centre[1], centre[0]] == 255
        assert mask[5, 5] == 0  # image corner, off pitch
        assert 0 < mask.mean() < 255

    def test_on_pitch_handles_a_frame_with_no_detections(self):
        """``on_pitch`` carries no empty guard of its own -- it relies on ``project``
        returning ``(0, 2)``."""
        out = on_pitch(np.zeros((0, 2)), make_ground_truth_h(), SPEC)
        assert out.shape == (0,)
        assert out.dtype == bool


class TestStoreAndRender:
    def test_load_reads_each_array_once(self, tmp_path):
        """Indexing the NpzFile per frame decompresses afresh each time, so every frame
        would hold its own full-length copy; frames must share one backing array per field."""
        from football_tracker.homography.record import FrameCalibration
        from football_tracker.homography.store import load_calibration, save_calibration

        stream = [
            FrameCalibration(
                frame_idx=i,
                H=make_ground_truth_h(),
                source="propagated",
                low_confidence=False,
                gmc=np.eye(2, 3),
            )
            for i in range(6)
        ]
        path = tmp_path / "calib.npz"
        save_calibration(path, stream)
        loaded = load_calibration(path)
        assert loaded[0].gmc.base is loaded[-1].gmc.base is not None
        assert loaded[0].H.base is loaded[-1].H.base is not None

    def test_the_source_encoding_order_is_fixed(self):
        """``_SOURCES`` index is the integer stored in the .npz, so reordering it would
        silently reinterpret every calibration file ever written."""
        from typing import get_args

        from football_tracker.homography.record import CalibrationSource
        from football_tracker.homography.store import _SOURCES

        assert _SOURCES == ("none", "solved", "propagated")
        assert set(_SOURCES) == set(get_args(CalibrationSource))

    def test_store_roundtrip(self, tmp_path):
        from football_tracker.homography.record import FrameCalibration
        from football_tracker.homography.store import load_calibration, save_calibration

        stream = [
            FrameCalibration(
                frame_idx=0,
                H=make_ground_truth_h(),
                source="solved",
                low_confidence=False,
                gmc=np.array([[1.0, 0.0, 2.5], [0.0, 1.0, -1.0]]),
                inliers=17,
                reproj_err_px=1.25,
            ),
            FrameCalibration(
                frame_idx=1,
                H=None,
                source="none",
                low_confidence=True,
                gmc=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            ),
        ]
        path = tmp_path / "calib.npz"
        save_calibration(path, stream)
        loaded = load_calibration(path)
        assert len(loaded) == 2
        np.testing.assert_allclose(loaded[0].H, stream[0].H)
        np.testing.assert_allclose(loaded[0].gmc, stream[0].gmc)
        assert loaded[0].source == "solved" and loaded[0].inliers == 17
        assert loaded[0].reproj_err_px == 1.25
        assert loaded[1].H is None and loaded[1].low_confidence
        assert loaded[1].source == "none"

    def test_the_frame_shape_round_trips(self, tmp_path):
        """H maps *cropped-frame* pixels to metres, so a reader holding only the .npz
        cannot check anything against the frame bounds unless the shape travels with it."""
        from football_tracker.homography.record import FrameCalibration
        from football_tracker.homography.store import load_frame_shape, save_calibration

        stream = [
            FrameCalibration(
                frame_idx=0,
                H=make_ground_truth_h(),
                source="solved",
                low_confidence=False,
                gmc=np.eye(2, 3),
            )
        ]
        path = tmp_path / "calib.npz"
        save_calibration(path, stream, frame_shape=FRAME_SHAPE)
        assert load_frame_shape(path) == FRAME_SHAPE

    def test_a_stream_saved_without_a_frame_shape_reads_back_none(self, tmp_path):
        """None, not a default resolution: a guessed shape would silently change which
        anchor fits pass the plausibility gate, which is worse than admitting ignorance."""
        from football_tracker.homography.record import FrameCalibration
        from football_tracker.homography.store import (
            load_calibration,
            load_frame_shape,
            save_calibration,
        )

        stream = [
            FrameCalibration(
                frame_idx=0,
                H=None,
                source="none",
                low_confidence=True,
                gmc=np.eye(2, 3),
            )
        ]
        path = tmp_path / "calib.npz"
        save_calibration(path, stream)
        assert load_frame_shape(path) is None
        assert len(load_calibration(path)) == 1  # the rest of the schema is unaffected

    def test_render_smoke(self):
        from football_tracker.homography.render import draw_pitch_overlay, render_radar

        H = make_ground_truth_h()
        frame = np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8)
        overlay = draw_pitch_overlay(frame, H, SPEC)
        assert overlay.shape == frame.shape
        assert overlay.any()  # something was drawn
        radar = render_radar(SPEC, H, FRAME_SHAPE)
        assert radar.any()
        assert render_radar(SPEC, None).shape == radar.shape


def anchor_from_h(H_img2pitch: np.ndarray, n: int = 12) -> "tuple[np.ndarray, np.ndarray]":
    """A perfect (image_pts, pitch_pts) anchor from a ground-truth H."""
    pitch_pts = SPEC.keypoints[:n]
    image_pts = project(np.linalg.inv(H_img2pitch), pitch_pts)
    return image_pts, pitch_pts


class TestLoadAnchors:
    def test_load_anchors_parses_frames(self, tmp_path):
        """Every frame carries >=4 points, so the below-4 fallback (tested below) stays out."""
        from football_tracker.homography.anchors import load_anchors

        doc = {
            "video": "x.mp4",
            "crop": {"top": 0},
            "frames": {
                "0": [
                    {"image": [10.0, 20.0], "pitch": [0.0, 0.0]},
                    {"image": [11.0, 21.0], "pitch": [0.0, 68.0]},
                    {"image": [12.0, 22.0], "pitch": [105.0, 0.0]},
                    {"image": [13.0, 23.0], "pitch": [105.0, 68.0]},
                ],
                "150": [
                    {"image": [30.0, 40.0], "pitch": [52.5, 0.0]},
                    {"image": [50.0, 60.0], "pitch": [52.5, 68.0]},
                    {"image": [31.0, 41.0], "pitch": [16.5, 24.84]},
                    {"image": [51.0, 61.0], "pitch": [88.5, 43.16]},
                ],
            },
        }
        path = tmp_path / "pitch_points.json"
        path.write_text(json.dumps(doc))
        anchors = load_anchors(path)
        assert set(anchors) == {0, 150}
        assert anchors[150][0].shape == (4, 2) and anchors[150][1].shape == (4, 2)
        np.testing.assert_allclose(anchors[0][0][0], [10.0, 20.0])

    @staticmethod
    def _phantom_doc() -> dict:
        """One frame: 4 real crossings plus the 4 phantom vertices (1-based 11/12/19/20).

        The phantom pitch coords subdivide the penalty-box line at a y where
        nothing crosses it — paint, but no cue for where along it to click.
        """
        return {
            "video": "x.mp4",
            "crop": {"top": 0},
            "frames": {
                "0": [
                    {"image": [1.0, 1.0], "pitch": [0.0, 0.0], "vertex": 1},
                    {"image": [2.0, 2.0], "pitch": [52.5, 0.0], "vertex": 14},
                    {"image": [3.0, 3.0], "pitch": [52.5, 68.0], "vertex": 17},
                    {"image": [4.0, 4.0], "pitch": [105.0, 0.0], "vertex": 25},
                    {"image": [9.0, 9.0], "pitch": [16.5, 24.84], "vertex": 11},
                    {"image": [9.0, 9.0], "pitch": [16.5, 43.16], "vertex": 12},
                    {"image": [9.0, 9.0], "pitch": [88.5, 24.84], "vertex": 19},
                    {"image": [9.0, 9.0], "pitch": [88.5, 43.16], "vertex": 20},
                ]
            },
        }

    def test_phantom_vertices_are_dropped_by_default(self, tmp_path):
        """1-based 'vertex' in the file maps to 0-based PHANTOM_VERTICES -- the
        off-by-one here would silently filter the wrong four features."""
        from football_tracker.homography.anchors import load_anchors

        path = tmp_path / "p.json"
        path.write_text(json.dumps(self._phantom_doc()))
        _, pitch = load_anchors(path)[0]
        assert len(pitch) == 4
        for bad in ([16.5, 24.84], [16.5, 43.16], [88.5, 24.84], [88.5, 43.16]):
            assert not any(np.allclose(p, bad) for p in pitch)

    def test_drop_phantom_false_keeps_every_vertex(self, tmp_path):
        from football_tracker.homography.anchors import load_anchors

        path = tmp_path / "p.json"
        path.write_text(json.dumps(self._phantom_doc()))
        assert len(load_anchors(path, drop_phantom=False)[0][1]) == 8

    def test_points_without_a_vertex_field_are_kept(self, tmp_path):
        """The field is optional in the format; absence must not discard a label."""
        from football_tracker.homography.anchors import load_anchors

        doc = self._phantom_doc()
        doc["frames"]["0"].append({"image": [7.0, 7.0], "pitch": [11.0, 34.0]})
        path = tmp_path / "p.json"
        path.write_text(json.dumps(doc))
        assert len(load_anchors(path)[0][1]) == 5

    def test_a_frame_that_would_drop_below_four_points_keeps_them_all(self, tmp_path):
        """A homography needs 4 correspondences; crippling the frame is worse."""
        from football_tracker.homography.anchors import load_anchors

        doc = {
            "video": "x.mp4",
            "crop": {"top": 0},
            "frames": {
                "0": [
                    {"image": [1.0, 1.0], "pitch": [0.0, 0.0], "vertex": 1},
                    {"image": [2.0, 2.0], "pitch": [52.5, 0.0], "vertex": 14},
                    {"image": [9.0, 9.0], "pitch": [16.5, 24.84], "vertex": 11},
                    {"image": [9.0, 9.0], "pitch": [88.5, 24.84], "vertex": 19},
                ]
            },
        }
        path = tmp_path / "p.json"
        path.write_text(json.dumps(doc))
        assert len(load_anchors(path)[0][1]) == 4

    def test_a_frame_that_would_drop_below_four_points_warns(self, tmp_path):
        """The fallback reinstates phantom/excluded points silently otherwise --
        a bad click surviving as good data with nothing printed anywhere."""
        from football_tracker.homography.anchors import load_anchors

        doc = {
            "video": "x.mp4",
            "crop": {"top": 0},
            "frames": {
                "0": [
                    {"image": [1.0, 1.0], "pitch": [0.0, 0.0], "vertex": 1},
                    {"image": [2.0, 2.0], "pitch": [52.5, 0.0], "vertex": 14},
                    {"image": [9.0, 9.0], "pitch": [16.5, 24.84], "vertex": 11},
                    {"image": [9.0, 9.0], "pitch": [88.5, 24.84], "vertex": 19},
                ]
            },
        }
        path = tmp_path / "p.json"
        path.write_text(json.dumps(doc))
        with pytest.warns(UserWarning, match=r"frame 0.*2 of 4"):
            load_anchors(path)

    def test_a_normal_frame_produces_no_warning(self, tmp_path, recwarn):
        from football_tracker.homography.anchors import load_anchors

        doc = {
            "video": "x.mp4",
            "crop": {"top": 0},
            "frames": {
                "0": [
                    {"image": [10.0, 20.0], "pitch": [0.0, 0.0]},
                    {"image": [30.0, 40.0], "pitch": [52.5, 0.0]},
                    {"image": [50.0, 60.0], "pitch": [52.5, 68.0]},
                    {"image": [70.0, 80.0], "pitch": [0.0, 68.0]},
                ]
            },
        }
        path = tmp_path / "p.json"
        path.write_text(json.dumps(doc))
        load_anchors(path)
        assert len(recwarn) == 0


def shifted_h(H_img2pitch: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """``H`` as seen by a camera whose image is translated by (dx, dy) px.

    Anchors built from it disagree with ``H_img2pitch`` anchors by exactly (dx, dy).
    """
    shift = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])
    return H_img2pitch @ np.linalg.inv(shift)


# Re-solving an anchor from a known H lands ~1e-4 px from the analytic matrix; this is
# the noise floor of "same homography", far below the ~20 px disagreements fused here.
REFIT_TOL_PX = 1e-3


def control_point_shift(H_a: np.ndarray, H_b: np.ndarray) -> float:
    """Largest image-space move of a control point between two homographies (px)."""
    cps = default_control_points(SPEC)
    a = project(np.linalg.inv(H_a), cps)
    b = project(np.linalg.inv(H_b), cps)
    return float(np.linalg.norm(a - b, axis=1).max())


class TestBlendHomographies:
    def test_weight_zero_returns_first(self):
        from football_tracker.homography.fuse import blend_homographies

        H_a = make_ground_truth_h()
        H_b = shifted_h(H_a, 20.0, 0.0)
        blended = blend_homographies(H_a, H_b, 0.0, default_control_points(SPEC))
        assert control_point_shift(blended, H_a) < REFIT_TOL_PX

    def test_weight_one_returns_second(self):
        from football_tracker.homography.fuse import blend_homographies

        H_a = make_ground_truth_h()
        H_b = shifted_h(H_a, 20.0, 0.0)
        blended = blend_homographies(H_a, H_b, 1.0, default_control_points(SPEC))
        assert control_point_shift(blended, H_b) < REFIT_TOL_PX

    def test_half_weight_lands_midway_in_image_space(self):
        from football_tracker.homography.fuse import blend_homographies

        H_a = make_ground_truth_h()
        H_b = shifted_h(H_a, 20.0, 0.0)
        blended = blend_homographies(H_a, H_b, 0.5, default_control_points(SPEC))
        pitch_pt = np.array([[52.5, 34.0]])
        a = project(np.linalg.inv(H_a), pitch_pt)
        b = project(np.linalg.inv(H_b), pitch_pt)
        got = project(np.linalg.inv(blended), pitch_pt)
        np.testing.assert_allclose(got, (a + b) / 2, atol=1e-6)


class TestFuseAnchors:
    """Two anchors that disagree by 20 px, with identity motion between them.

    Forward-only propagation must hold the first anchor then snap 20 px at the
    second; the whole point of fusing is to spread that disagreement across the
    gap instead.
    """

    def setup_method(self):
        self.H_a = make_ground_truth_h()
        self.H_b = shifted_h(self.H_a, 20.0, 0.0)
        self.anchors = {0: anchor_from_h(self.H_a), 10: anchor_from_h(self.H_b)}
        self.frames = list(range(13))

    def _fused(self):
        from football_tracker.homography.fuse import fuse_anchors

        return fuse_anchors(
            self.anchors, motions={}, frames=self.frames, spec=SPEC, frame_shape=FRAME_SHAPE
        )

    def test_anchor_frames_keep_their_own_solution(self):
        fused = self._fused()
        assert control_point_shift(fused[0], self.H_a) < REFIT_TOL_PX
        assert control_point_shift(fused[10], self.H_b) < REFIT_TOL_PX

    def test_no_frame_to_frame_jump_across_the_gap(self):
        fused = self._fused()
        jumps = [control_point_shift(fused[f], fused[f + 1]) for f in range(10)]
        # 20 px spread over 10 frames is 2 px/frame; forward-only would be 20.
        assert max(jumps) < 3.0

    def test_disagreement_is_spread_evenly_not_dumped_on_one_frame(self):
        fused = self._fused()
        jumps = [control_point_shift(fused[f], fused[f + 1]) for f in range(10)]
        assert max(jumps) - min(jumps) < 0.5

    def test_frames_after_last_anchor_hold_it(self):
        fused = self._fused()
        assert control_point_shift(fused[12], self.H_b) < REFIT_TOL_PX

    def test_frames_before_first_anchor_are_calibrated(self):
        from football_tracker.homography.fuse import fuse_anchors

        fused = fuse_anchors(
            {5: anchor_from_h(self.H_a)},
            motions={},
            frames=list(range(8)),
            spec=SPEC,
            frame_shape=FRAME_SHAPE,
        )
        assert control_point_shift(fused[0], self.H_a) < REFIT_TOL_PX

    def test_motion_is_applied_between_anchors(self):
        """With a real dx=2/frame pan and matching anchors, fusing is exact."""
        from football_tracker.homography.fuse import fuse_anchors

        H_true = make_ground_truth_h()
        motions = {
            f: FrameMotion(np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]])) for f in range(1, 11)
        }
        anchors = {0: anchor_from_h(H_true), 10: anchor_from_h(shifted_h(H_true, 20.0, 0.0))}
        fused = fuse_anchors(
            anchors, motions=motions, frames=list(range(11)), spec=SPEC, frame_shape=FRAME_SHAPE
        )
        # Frame 4 has panned 8 px; a pitch point must land 8 px right of frame 0.
        pitch_pt = np.array([[52.5, 34.0]])
        at0 = project(np.linalg.inv(fused[0]), pitch_pt)
        at4 = project(np.linalg.inv(fused[4]), pitch_pt)
        np.testing.assert_allclose(at4, at0 + [8.0, 0.0], atol=1e-6)

    def test_unsolvable_anchor_is_skipped_not_fatal(self):
        from football_tracker.homography.fuse import fuse_anchors

        degenerate = (np.zeros((5, 2)), np.zeros((5, 2)))
        fused = fuse_anchors(
            {0: anchor_from_h(self.H_a), 5: degenerate, 10: anchor_from_h(self.H_b)},
            motions={},
            frames=self.frames,
            spec=SPEC,
            frame_shape=FRAME_SHAPE,
        )
        assert control_point_shift(fused[0], self.H_a) < REFIT_TOL_PX
        jumps = [control_point_shift(fused[f], fused[f + 1]) for f in range(10)]
        assert max(jumps) < 3.0

    def test_no_anchors_yields_no_calibration(self):
        from football_tracker.homography.fuse import fuse_anchors

        fused = fuse_anchors({}, motions={}, frames=[0, 1, 2], spec=SPEC, frame_shape=FRAME_SHAPE)
        assert all(fused[f] is None for f in (0, 1, 2))


class TestExcludedClicks:
    """Per-click opt-out for labels that are demonstrably not on the feature.

    Distinct from PHANTOM_VERTICES, which condemns a *vertex* everywhere: this
    marks one click in one frame, for a reason recorded next to it.
    """

    def _doc(self, extra: dict | None = None) -> dict:
        points = [
            {"image": [10.0, 10.0], "pitch": [0.0, 0.0], "vertex": 1},
            {"image": [20.0, 10.0], "pitch": [105.0, 0.0], "vertex": 25},
            {"image": [20.0, 20.0], "pitch": [105.0, 68.0], "vertex": 29},
            {"image": [10.0, 20.0], "pitch": [0.0, 68.0], "vertex": 5},
            {"image": [15.0, 15.0], "pitch": [52.5, 34.0], "vertex": 30, **(extra or {})},
        ]
        return {"frames": {"0": points}}

    def test_a_marked_click_is_dropped(self, tmp_path):
        from football_tracker.homography.anchors import load_anchors

        path = tmp_path / "p.json"
        path.write_text(json.dumps(self._doc({"exclude": "clicked on a player"})))
        assert len(load_anchors(path)[0][0]) == 4

    def test_an_unmarked_click_is_kept(self, tmp_path):
        from football_tracker.homography.anchors import load_anchors

        path = tmp_path / "p.json"
        path.write_text(json.dumps(self._doc()))
        assert len(load_anchors(path)[0][0]) == 5

    def test_exclusions_can_be_disabled(self, tmp_path):
        from football_tracker.homography.anchors import load_anchors

        path = tmp_path / "p.json"
        path.write_text(json.dumps(self._doc({"exclude": "clicked on a player"})))
        assert len(load_anchors(path, drop_excluded=False)[0][0]) == 5

    def test_a_frame_is_never_stripped_below_a_solvable_fit(self, tmp_path):
        """Same guard as the phantom path: keep the frame intact rather than cripple it."""
        from football_tracker.homography.anchors import load_anchors

        doc = self._doc()
        for point in doc["frames"]["0"][:3]:
            point["exclude"] = "test"
        path = tmp_path / "p.json"
        path.write_text(json.dumps(doc))
        assert len(load_anchors(path)[0][0]) == 5


class TestMalformedAnchorFiles:
    """pitch_points.json is hand-editable (``exclude``), so a typo must name the file and
    frame rather than surface as a bare KeyError."""

    @staticmethod
    def _four() -> list[dict]:
        return [
            {"image": [10.0, 10.0], "pitch": [0.0, 0.0]},
            {"image": [20.0, 10.0], "pitch": [105.0, 0.0]},
            {"image": [20.0, 20.0], "pitch": [105.0, 68.0]},
            {"image": [10.0, 20.0], "pitch": [0.0, 68.0]},
        ]

    def test_a_frame_with_no_points_keeps_the_two_column_shape(self, tmp_path):
        """Every other return is (N, 2). An empty frame degrading to (0,) reads as a 1-D
        array to anything that projects it, which fails far from the cause."""
        from football_tracker.homography.anchors import load_anchors

        path = tmp_path / "p.json"
        path.write_text(json.dumps({"frames": {"0": []}}))
        image, pitch = load_anchors(path)[0]
        assert image.shape == (0, 2)
        assert pitch.shape == (0, 2)

    def test_a_point_missing_a_coordinate_field_names_the_file_and_frame(self, tmp_path):
        from football_tracker.homography.anchors import load_anchors

        points = self._four()
        del points[2]["image"]
        path = tmp_path / "p.json"
        path.write_text(json.dumps({"frames": {"7": points}}))
        with pytest.raises(ValueError, match=r"p\.json.*frame 7.*'image'"):
            load_anchors(path)

    def test_a_coordinate_that_is_not_an_xy_pair_is_rejected(self, tmp_path):
        from football_tracker.homography.anchors import load_anchors

        points = self._four()
        points[1]["pitch"] = [1.0, 2.0, 3.0]
        path = tmp_path / "p.json"
        path.write_text(json.dumps({"frames": {"0": points}}))
        with pytest.raises(ValueError, match=r"frame 0.*'pitch'"):
            load_anchors(path)

    def test_a_non_integer_frame_key_is_rejected_before_any_warning(self, tmp_path):
        """This frame also trips the under-4 fallback; a point-count warning on a file whose
        real fault is its key would send the user after the wrong thing."""
        from football_tracker.homography.anchors import load_anchors

        path = tmp_path / "p.json"
        path.write_text(json.dumps({"frames": {"12a": self._four()[:1]}}))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match=r"frame key '12a' is not an integer"):
                load_anchors(path)

    def test_the_filter_flags_are_keyword_only(self, tmp_path):
        """Positionally, ``load_anchors(path, False)`` would silently mean drop_phantom."""
        from football_tracker.homography.anchors import load_anchors

        path = tmp_path / "p.json"
        path.write_text(json.dumps({"frames": {"0": self._four()}}))
        with pytest.raises(TypeError):
            load_anchors(path, False)


class TestFusedCalibrationStream:
    """The shipping-shape output: one FrameCalibration per frame, fused H inside."""

    def setup_method(self):
        self.H_true = make_ground_truth_h()
        self.anchors = {0: anchor_from_h(self.H_true), 10: anchor_from_h(self.H_true)}
        self.frames = list(range(13))

    def _stream(self, **kwargs):
        from football_tracker.homography.fuse import fused_calibration_stream

        return fused_calibration_stream(
            self.anchors,
            motions={},
            frames=self.frames,
            spec=SPEC,
            frame_shape=FRAME_SHAPE,
            **kwargs,
        )

    def test_one_entry_per_frame_in_order(self):
        stream = self._stream()
        assert [c.frame_idx for c in stream] == self.frames

    def test_anchor_frames_are_marked_solved_and_the_rest_propagated(self):
        stream = {c.frame_idx: c for c in self._stream()}
        assert stream[0].source == "solved"
        assert stream[10].source == "solved"
        assert all(stream[f].source == "propagated" for f in (1, 5, 9, 11, 12))

    def test_gmc_is_carried_through_for_downstream_consumers(self):
        from football_tracker.homography.fuse import fused_calibration_stream

        affine = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]])
        motions = {f: FrameMotion(affine) for f in range(1, 13)}
        stream = fused_calibration_stream(
            self.anchors, motions, self.frames, spec=SPEC, frame_shape=FRAME_SHAPE
        )
        np.testing.assert_allclose(stream[5].gmc, affine)
        # Frame 0 has no predecessor, so it reports identity rather than nothing.
        np.testing.assert_allclose(stream[0].gmc, np.array([[1.0, 0, 0], [0, 1.0, 0]]))

    def test_confidence_is_measured_to_the_nearest_anchor_in_either_direction(self):
        """A fused frame is bracketed, so the distance to the closer anchor is what matters."""
        stream = {c.frame_idx: c for c in self._stream(max_propagated_frames=3)}
        assert not stream[9].low_confidence  # 1 frame before anchor 10
        assert not stream[12].low_confidence  # 2 frames after anchor 10
        assert stream[5].low_confidence  # 5 from either — beyond the budget

    def test_no_usable_anchor_yields_uncalibrated_frames(self):
        from football_tracker.homography.fuse import fused_calibration_stream

        stream = fused_calibration_stream(
            {}, motions={}, frames=[0, 1], spec=SPEC, frame_shape=FRAME_SHAPE
        )
        assert all(c.H is None and c.source == "none" and c.low_confidence for c in stream)
