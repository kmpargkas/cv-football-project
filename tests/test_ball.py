"""The ball sub-system's per-frame half, over fake detectors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from football_tracker.config import BallConfig, DetectionConfig
from football_tracker.detection.interface import Detections, ObjectClass
from football_tracker.homography.pitch import PitchSpec
from football_tracker.homography.record import FrameCalibration
from football_tracker.tracking.ball import (
    BallCandidate,
    BallKalman,
    BallObserver,
    gate_candidates,
    merge_candidates,
)

IDENTITY_2X3 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _calib(frame_idx: int, H, low_confidence: bool = False) -> FrameCalibration:
    return FrameCalibration(
        frame_idx=frame_idx,
        H=H,
        source="solved" if H is not None else "none",
        low_confidence=low_confidence,
        gmc=IDENTITY_2X3,
    )


def _cand(
    x: float, y: float, conf: float = 0.5, w: float = 12.0, h: float = 12.0, source: str = "full"
) -> BallCandidate:
    return BallCandidate(
        frame_idx=0,
        xy=np.array([x, y], dtype=np.float32),
        wh=np.array([w, h], dtype=np.float32),
        confidence=conf,
        source=source,
    )


def _run_track(kf: BallKalman, points: list[tuple[float, float]]) -> None:
    """Feed a point sequence through predict/update cycles (initiates on the first)."""
    kf.initiate(np.array(points[0], dtype=np.float64))
    for xy in points[1:]:
        kf.predict()
        kf.update(np.array(xy, dtype=np.float64))


class TestBallKalman:
    def test_uninitialized_until_initiate(self):
        kf = BallKalman(q_pos=1.0, q_vel=0.5, r_px=5.0)
        assert kf.initialized is False
        kf.initiate(np.array([10.0, 20.0]))
        assert kf.initialized is True

    def test_constant_velocity_prediction(self):
        kf = BallKalman(q_pos=1.0, q_vel=0.5, r_px=5.0)
        # Ball moving +5 px/frame in x, constant y — noiseless.
        _run_track(kf, [(100.0 + 5.0 * t, 100.0) for t in range(7)])
        pred = kf.predict()  # next point should be (135, 100)
        assert pred == pytest.approx([135.0, 100.0], abs=2.0)

    def test_apply_affine_translation_moves_position_not_velocity(self):
        kf = BallKalman(q_pos=1.0, q_vel=0.5, r_px=5.0)
        # Stationary ball at (100, 100): velocity ~0.
        _run_track(kf, [(100.0, 100.0)] * 6)
        A = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, -5.0]])
        kf.apply_affine(A)
        pred = kf.predict()
        assert pred == pytest.approx([110.0, 95.0], abs=1.0)

    def test_apply_affine_rotation_warps_velocity(self):
        kf = BallKalman(q_pos=1.0, q_vel=0.5, r_px=5.0)
        # Moving +5 px/frame in x; state ends near pos (130, 100), vel (5, 0).
        _run_track(kf, [(100.0 + 5.0 * t, 100.0) for t in range(7)])
        # 90° CCW about the origin: pos -> (-100, 130), vel -> (0, 5).
        A = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
        kf.apply_affine(A)
        pred = kf.predict()
        assert pred == pytest.approx([-100.0, 135.0], abs=2.0)

    def test_gating_distance_orders_near_before_far(self):
        kf = BallKalman(q_pos=1.0, q_vel=0.5, r_px=5.0)
        _run_track(kf, [(100.0, 100.0)] * 5)
        pred = kf.predict()
        near = kf.gating_distance(pred)
        far = kf.gating_distance(pred + np.array([200.0, 0.0]))
        assert 0.0 <= near < 1.0
        assert far > 9.21  # well outside the chi2(0.99, 2 dof) gate
        assert np.isfinite(far)

    def test_time_since_update_counts_coasting(self):
        kf = BallKalman(q_pos=1.0, q_vel=0.5, r_px=5.0)
        kf.initiate(np.array([50.0, 50.0]))
        assert kf.time_since_update == 0
        kf.predict()
        kf.predict()
        assert kf.time_since_update == 2
        kf.update(np.array([50.0, 50.0]))
        assert kf.time_since_update == 0

    def test_gating_stays_finite_after_affine_chain(self):
        kf = BallKalman(q_pos=1.0, q_vel=0.5, r_px=5.0)
        _run_track(kf, [(100.0 + 5.0 * t, 100.0) for t in range(4)])
        rng = np.random.default_rng(0)
        for _ in range(50):
            angle = rng.uniform(-0.05, 0.05)
            c, s = np.cos(angle), np.sin(angle)
            A = np.array([[c, -s, rng.uniform(-3, 3)], [s, c, rng.uniform(-3, 3)]])
            kf.apply_affine(A)
            kf.predict()
        d = kf.gating_distance(kf.predict())
        assert np.isfinite(d) and d >= 0.0


class TestMergeCandidates:
    def test_collapses_near_duplicates_to_higher_confidence(self):
        roi = _cand(100.0, 100.0, conf=0.8, source="roi")
        full = _cand(105.0, 100.0, conf=0.4, source="full")
        merged = merge_candidates([full, roi], merge_dist_px=20.0)
        assert len(merged) == 1
        assert merged[0].source == "roi"
        assert merged[0].confidence == pytest.approx(0.8)

    def test_keeps_distant_candidates(self):
        a = _cand(100.0, 100.0, conf=0.8)
        b = _cand(400.0, 100.0, conf=0.4)
        merged = merge_candidates([a, b], merge_dist_px=20.0)
        assert len(merged) == 2

    def test_empty_input(self):
        assert merge_candidates([], merge_dist_px=20.0) == []


def _ball_detections(*centers: tuple[float, float], conf: float = 0.9, size: float = 12.0):
    """Detections with one ball box (size×size) per center."""
    if not centers:
        return Detections.empty()
    half = size / 2.0
    return Detections(
        xyxy=[[x - half, y - half, x + half, y + half] for x, y in centers],
        confidence=[conf] * len(centers),
        class_id=[int(ObjectClass.BALL)] * len(centers),
    )


class _FakeRoiDetector:
    """Scripted Detector: returns canned detections (in crop coords), records calls."""

    def __init__(self, detections: Detections | None = None):
        self.detections = detections if detections is not None else Detections.empty()
        self.frames_seen: list[np.ndarray] = []

    def detect(self, frame: np.ndarray) -> Detections:
        self.frames_seen.append(frame)
        return self.detections


def _observer(roi_detector=None, **cfg_overrides) -> BallObserver:
    cfg = BallConfig(**cfg_overrides)
    return BallObserver(
        roi_detector=roi_detector,
        kalman=BallKalman(q_pos=cfg.q_pos, q_vel=cfg.q_vel, r_px=cfg.r_px),
        calib_by_idx={},
        spec=PitchSpec(),
        cfg=cfg,
    )


class _FakeYOLODetector:
    """Stands in for YOLODetector in from_config: records how it was reconfigured."""

    def __init__(self, tag: str = "main"):
        self.tag = tag
        self.reconfigured_with: dict | None = None

    def reconfigured(self, **kwargs):
        self.reconfigured_with = kwargs
        clone = _FakeYOLODetector(f"{self.tag}:roi")
        return clone

    def detect(self, frame: np.ndarray) -> Detections:
        return Detections.empty()


class TestBallObserverDetectorSeam:
    """`ball.weights` selects the ROI detector.

    A ball-only fine-tune cannot drive the player pass, so when it is set the ROI pass
    loads its own model instead of sharing the main detector's.
    """

    def _from_config(self, ball_cfg, detection_cfg, detector):
        return BallObserver.from_config(ball_cfg, detection_cfg, [], PitchSpec(), detector)

    def test_shares_main_detector_when_no_ball_weights(self):
        main = _FakeYOLODetector()
        obs = self._from_config(BallConfig(), DetectionConfig(weights=Path("run1.pt")), main)
        assert main.reconfigured_with == {"imgsz": 640, "conf": 0.10, "ball_conf": None}
        assert obs._roi_detector is not None

    def test_ball_weights_loads_a_separate_detector(self, monkeypatch):
        """With ball.weights set, the ROI pass gets its own model, not the main one."""
        main = _FakeYOLODetector()
        built: list = []

        def fake_from_config(cfg, device="auto"):
            built.append((cfg.weights, cfg.imgsz, cfg.conf, cfg.ball_conf))
            return _FakeYOLODetector("ball")

        monkeypatch.setattr(
            "football_tracker.detection.yolo.YOLODetector.from_config",
            staticmethod(fake_from_config),
        )
        obs = self._from_config(
            BallConfig(weights=Path("ball_ft.pt")),
            DetectionConfig(weights=Path("run1.pt")),
            main,
        )
        # The main detector must NOT have been reconfigured — it stays the player model.
        assert main.reconfigured_with is None
        # A second detector was built from the ball weights at ROI settings.
        assert built == [(Path("ball_ft.pt"), 640, 0.10, None)]
        assert obs._roi_detector is not None

    def test_ball_weights_works_without_a_main_detector(self, monkeypatch):
        """Ball-only runs (no player detector) still get their ROI model."""
        built: list = []

        def fake_from_config(cfg, device="auto"):
            built.append(cfg.weights)
            return _FakeYOLODetector("ball")

        monkeypatch.setattr(
            "football_tracker.detection.yolo.YOLODetector.from_config",
            staticmethod(fake_from_config),
        )
        obs = self._from_config(BallConfig(weights=Path("ball_ft.pt")), DetectionConfig(), None)
        assert built == [Path("ball_ft.pt")]
        assert obs._roi_detector is not None

    def test_no_detector_and_no_weights_disables_the_roi_pass(self):
        obs = self._from_config(BallConfig(), DetectionConfig(), None)
        assert obs._roi_detector is None


class TestCandidateGateOrdering:
    """Size-gating must precede duplicate-merging: merging is greedy by confidence, so
    an over-large false box could swallow the real ball and then be deleted itself."""

    FRAME = np.zeros((1080, 1920, 3), dtype=np.uint8)

    def test_oversized_high_conf_box_does_not_suppress_the_real_ball(self):
        half_big, half_small = 25.0, 7.0  # 50 px (over max_box_px=40) and 14 px
        cx, cy = 500.0, 400.0
        detections = Detections(
            xyxy=[
                [cx - half_big, cy - half_big, cx + half_big, cy + half_big],
                [cx - half_small, cy - half_small, cx + half_small, cy + half_small],
            ],
            confidence=[0.8, 0.4],  # the junk box outranks the real one
            class_id=[int(ObjectClass.BALL), int(ObjectClass.BALL)],
        )
        obs = _observer(merge_dist_px=20.0, min_box_px=4.0, max_box_px=40.0)
        cands = obs.observe(0, self.FRAME, detections)

        assert len(cands) == 1, "the valid 14 px ball must survive the 50 px false box"
        assert float(np.max(cands[0].wh)) == pytest.approx(14.0)
        assert cands[0].confidence == pytest.approx(0.4)


class TestAssociationReachBound:
    """Association must stay physically bounded while the filter coasts: the chi2 gate
    alone grows with t^2 and passes the frame width within tens of coast frames."""

    FRAME = np.zeros((2000, 3000, 3), dtype=np.uint8)

    def _coasting_observer(self, coast: int):
        obs = _observer()
        obs.observe(0, self.FRAME, _ball_detections((1000.0, 800.0), conf=0.9))
        for t in range(1, coast + 1):  # no detections -> the filter coasts
            obs.observe(t, self.FRAME, Detections.empty())
        return obs

    def test_far_candidate_is_not_associated_deep_into_a_coast(self):
        """At 40 coast frames the chi2 gate is ~2450 px but the ball can only have
        travelled 40*41 = 1640 px. A candidate 2000 px away is inside the chi2 gate
        and outside physical reach — it must not be associated."""
        obs = self._coasting_observer(40)
        start = obs.last_prediction.copy()
        obs.observe(41, self.FRAME, _ball_detections((start[0] + 2000.0, start[1]), conf=0.9))
        assert obs._kalman.time_since_update > 0, "associated a physically impossible jump"

    def test_reacquires_on_best_candidate_once_the_prediction_is_stale(self):
        """Past roi_max_coast the prediction is declared stale, so re-acquire."""
        cfg = BallConfig()
        obs = self._coasting_observer(cfg.roi_max_coast + 2)
        obs.observe(cfg.roi_max_coast + 3, self.FRAME, _ball_detections((2500.0, 1500.0), conf=0.9))
        # Re-initiated at the new evidence rather than dragged there by association.
        assert obs._kalman.time_since_update == 0
        assert obs.last_prediction == pytest.approx([2500.0, 1500.0], abs=1.0)

    def test_nearby_candidate_still_associates_normally(self):
        """The bound must not break ordinary tracking: 6 px/frame is well inside."""
        obs = _observer()
        obs.observe(0, self.FRAME, _ball_detections((1000.0, 800.0), conf=0.9))
        for t in range(1, 6):
            obs.observe(t, self.FRAME, _ball_detections((1000.0 + 6.0 * t, 800.0), conf=0.9))
        assert obs._kalman.time_since_update == 0


class TestBallObserver:
    FRAME = np.zeros((1080, 1920, 3), dtype=np.uint8)

    def test_full_frame_ball_detections_become_candidates(self):
        obs = _observer()
        out = obs.observe(0, self.FRAME, _ball_detections((300.0, 400.0)))
        assert len(out) == 1
        assert out[0].source == "full"
        assert out[0].xy == pytest.approx([300.0, 400.0])

    def test_non_ball_detections_ignored(self):
        obs = _observer()
        players = Detections(
            xyxy=[[0, 0, 30, 80]], confidence=[0.9], class_id=[int(ObjectClass.PLAYER)]
        )
        assert obs.observe(0, self.FRAME, players) == []

    def test_no_roi_pass_until_kalman_initialized(self):
        roi = _FakeRoiDetector()
        obs = _observer(roi_detector=roi)
        obs.observe(0, self.FRAME, Detections.empty())
        assert roi.frames_seen == []

    def test_roi_crop_centered_on_prediction_and_mapped_back(self):
        # ROI detector reports the ball at (330, 330) in crop coordinates.
        roi = _FakeRoiDetector(_ball_detections((330.0, 330.0), conf=0.95))
        obs = _observer(roi_detector=roi, roi_size=640)
        # Frame 0 initializes the Kalman at (1000, 500); frame 1 predicts ~there.
        obs.observe(0, self.FRAME, _ball_detections((1000.0, 500.0)))
        out = obs.observe(1, self.FRAME, Detections.empty())
        assert len(roi.frames_seen) == 1
        assert roi.frames_seen[0].shape == (640, 640, 3)
        # Crop origin = prediction - 320 → (680, 180); 330 in crop = 1010, 510 in frame.
        (cand,) = [c for c in out if c.source == "roi"]
        assert cand.xy == pytest.approx([1010.0, 510.0], abs=1.0)

    def test_roi_clamped_by_shifting_at_frame_edge(self):
        roi = _FakeRoiDetector()
        obs = _observer(roi_detector=roi, roi_size=640)
        obs.observe(0, self.FRAME, _ball_detections((10.0, 10.0)))
        obs.observe(1, self.FRAME, Detections.empty())
        assert len(roi.frames_seen) == 1
        assert roi.frames_seen[0].shape == (640, 640, 3)  # shifted, never shrunk

    def test_full_and_roi_hits_of_same_ball_merge(self):
        # ROI sees the ball at crop coords mapping to ~(1005, 500); full sees (1000, 500).
        roi = _FakeRoiDetector(_ball_detections((325.0, 320.0), conf=0.95))
        obs = _observer(roi_detector=roi, roi_size=640, merge_dist_px=20.0)
        obs.observe(0, self.FRAME, _ball_detections((1000.0, 500.0)))
        out = obs.observe(1, self.FRAME, _ball_detections((1000.0, 500.0), conf=0.5))
        assert len(out) == 1
        assert out[0].source == "roi"  # higher confidence copy wins

    def test_roi_stops_after_coast_cap(self):
        roi = _FakeRoiDetector()
        obs = _observer(roi_detector=roi, roi_size=640, roi_max_coast=2)
        obs.observe(0, self.FRAME, _ball_detections((1000.0, 500.0)))
        for idx in range(1, 5):
            obs.observe(idx, self.FRAME, Detections.empty())
        # Coasting frames 1 and 2 get an ROI pass; 3 and 4 exceed the cap.
        assert len(roi.frames_seen) == 2

    def test_reacquires_after_long_coast(self):
        obs = _observer(roi_max_coast=1)
        obs.observe(0, self.FRAME, _ball_detections((1000.0, 500.0)))
        for idx in range(1, 4):
            obs.observe(idx, self.FRAME, Detections.empty())
        # A fresh detection far from the stale prediction re-initiates the filter.
        obs.observe(4, self.FRAME, _ball_detections((600.0, 900.0)))
        obs.observe(5, self.FRAME, Detections.empty())
        assert obs.last_prediction == pytest.approx([600.0, 900.0], abs=5.0)


class TestGateCandidates:
    def _gate(self, cands, calib, **kwargs):
        defaults = dict(min_box_px=4.0, max_box_px=40.0, mask_margin_m=6.0)
        defaults.update(kwargs)
        return gate_candidates(cands, calib, PitchSpec(), **defaults)

    def test_size_prior_rejects_too_small_and_too_large(self):
        tiny = _cand(50.0, 50.0, w=2.0, h=2.0)
        ok = _cand(50.0, 50.0, w=12.0, h=12.0)
        huge = _cand(50.0, 50.0, w=80.0, h=60.0)
        kept = self._gate([tiny, ok, huge], _calib(0, np.eye(3)))
        assert kept == [ok]

    def test_off_pitch_candidate_rejected_with_identity_h(self):
        # H = identity → image px ≈ pitch meters; PitchSpec is 105x68.
        on = _cand(50.0, 30.0)
        off = _cand(300.0, 300.0)
        kept = self._gate([on, off], _calib(0, np.eye(3)))
        assert kept == [on]

    def test_pass_through_when_h_is_none(self):
        off = _cand(300.0, 300.0)
        assert self._gate([off], _calib(0, None)) == [off]

    def test_pass_through_when_low_confidence(self):
        off = _cand(300.0, 300.0)
        assert self._gate([off], _calib(0, np.eye(3), low_confidence=True)) == [off]

    def test_pass_through_when_calib_missing(self):
        off = _cand(300.0, 300.0)
        assert self._gate([off], None) == [off]
