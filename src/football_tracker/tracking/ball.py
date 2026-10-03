"""Ball tracking, Pass A: per-frame candidate generation.

A constant-velocity Kalman filter, warped each frame by the cached camera motion,
places a native-resolution ROI re-detection around its prediction; full-frame and ROI
detections merge into :class:`BallCandidate` lists. The filter only places the ROI -
the offline path solve in ``ball_trajectory`` decides the trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from football_tracker.detection.interface import Detections, ObjectClass
from football_tracker.homography.mask import on_pitch

if TYPE_CHECKING:
    from football_tracker.config import BallConfig, DetectionConfig
    from football_tracker.detection.interface import Detector
    from football_tracker.detection.yolo import YOLODetector
    from football_tracker.homography.pitch import PitchSpec
    from football_tracker.homography.record import FrameCalibration

# Initial velocity std (px/frame) as a multiple of the measurement noise
_INIT_VEL_STD_FACTOR = 4.0


@dataclass(frozen=True)
class BallCandidate:
    """One ball hypothesis on one frame (crop-space pixels, box center)."""

    frame_idx: int
    xy: np.ndarray  # (2,) center, float32
    wh: np.ndarray  # (2,) box size, float32
    confidence: float
    source: str  # "full" | "roi"


def merge_candidates(candidates: list[BallCandidate], merge_dist_px: float) -> list[BallCandidate]:
    """Collapse near-duplicates (full-frame vs ROI hits of the same ball).

    Greedy by confidence: a candidate within ``merge_dist_px`` of an already-kept
    one is dropped, so the higher-confidence copy survives.
    """
    kept: list[BallCandidate] = []
    for cand in sorted(candidates, key=lambda c: -c.confidence):
        if all(float(np.linalg.norm(cand.xy - k.xy)) > merge_dist_px for k in kept):
            kept.append(cand)
    return kept


def gate_candidates(
    candidates: list[BallCandidate],
    calib: FrameCalibration | None,
    spec: PitchSpec,
    *,
    min_box_px: float,
    max_box_px: float,
    mask_margin_m: float,
) -> list[BallCandidate]:
    """Size prior + pitch-mask gate on a frame's candidates.

    The size prior kills player-sized misclassifications; the mask (generous
    margin — an airborne ball projects past the touchlines through the flat H)
    kills stray perimeter balls. Pass-through on missing/low-confidence
    calibration, mirroring :func:`~football_tracker.tracking.gmc.off_field_mask`.
    """
    sized = [c for c in candidates if min_box_px <= float(np.max(c.wh)) <= max_box_px]
    if not sized or calib is None or calib.H is None or calib.low_confidence:
        return sized
    centers = np.stack([c.xy for c in sized]).astype(np.float64)
    keep = on_pitch(centers, calib.H, spec, margin_m=mask_margin_m)
    return [c for c, k in zip(sized, keep, strict=True) if k]


class BallObserver:
    """Per-frame ball-candidate generator: full-frame detections + ROI re-detection."""

    def __init__(
        self,
        roi_detector: Detector | None,
        kalman: BallKalman,
        calib_by_idx: dict[int, FrameCalibration],
        spec: PitchSpec,
        cfg: BallConfig,
    ) -> None:
        self._roi_detector = roi_detector
        self._kalman = kalman
        self._calib_by_idx = calib_by_idx
        self._spec = spec
        self._cfg = cfg
        self._last_prediction: np.ndarray | None = None

    @classmethod
    def from_config(
        cls,
        cfg: BallConfig,
        detection_cfg: DetectionConfig,
        stream: list[FrameCalibration],
        spec: PitchSpec,
        detector: YOLODetector | None,
        device: str = "auto",
    ) -> BallObserver:
        """Build from the run config, resolving which model drives the ROI pass.

        ``cfg.weights`` unset → reuse ``detector`` at ROI thresholds. Set → load a separate
        detector from those weights, leaving ``detector`` untouched as the player model, so
        a ball-only specialist and a player generalist can run in the same pass.
        """
        roi_detector = None
        if cfg.weights is not None:
            from football_tracker.detection.yolo import YOLODetector as _YOLODetector

            roi_cfg = detection_cfg.model_copy(
                update={
                    "weights": cfg.weights,
                    "imgsz": cfg.roi_imgsz,
                    "conf": cfg.roi_conf,
                    "ball_conf": None,
                }
            )
            roi_detector = _YOLODetector.from_config(roi_cfg, device=device)
        elif detector is not None:
            roi_detector = detector.reconfigured(
                imgsz=cfg.roi_imgsz, conf=cfg.roi_conf, ball_conf=None
            )
        return cls(
            roi_detector=roi_detector,
            kalman=BallKalman(q_pos=cfg.q_pos, q_vel=cfg.q_vel, r_px=cfg.r_px),
            calib_by_idx={c.frame_idx: c for c in stream},
            spec=spec,
            cfg=cfg,
        )

    @property
    def last_prediction(self) -> np.ndarray | None:
        """The most recent Kalman prediction, or the last re-acquisition point."""
        return self._last_prediction

    def observe(
        self, frame_idx: int, frame: np.ndarray, detections: Detections
    ) -> list[BallCandidate]:
        """Generate this frame's ball candidates and advance the ROI filter."""
        calib = self._calib_by_idx.get(frame_idx)
        candidates = self._full_candidates(frame_idx, detections)

        if self._kalman.initialized:
            if calib is not None:
                self._kalman.apply_affine(calib.gmc)
            prediction = self._kalman.predict()
            self._last_prediction = prediction
            if (
                self._roi_detector is not None
                and self._kalman.time_since_update <= self._cfg.roi_max_coast
            ):
                candidates += self._roi_candidates(frame_idx, frame, prediction)

        # Gate BEFORE merging. Merging is greedy by confidence, so the other order
        # lets an over-large false box swallow a valid smaller one and then be
        # deleted by the size prior — the frame yields no candidate at all.
        candidates = gate_candidates(
            candidates,
            calib,
            self._spec,
            min_box_px=self._cfg.min_box_px,
            max_box_px=self._cfg.max_box_px,
            mask_margin_m=self._cfg.mask_margin_m,
        )
        candidates = merge_candidates(candidates, self._cfg.merge_dist_px)
        self._advance_kalman(candidates)
        return candidates

    @staticmethod
    def _to_candidates(
        frame_idx: int, balls: Detections, origin: np.ndarray, source: str
    ) -> list[BallCandidate]:
        """Ball boxes -> candidates, with ``origin`` added to map crop pixels to frame pixels."""
        return [
            BallCandidate(
                frame_idx=frame_idx,
                xy=np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32) + origin,
                wh=np.array([x2 - x1, y2 - y1], dtype=np.float32),
                confidence=float(conf),
                source=source,
            )
            for (x1, y1, x2, y2), conf in zip(balls.xyxy, balls.confidence, strict=True)
        ]

    def _full_candidates(self, frame_idx: int, detections: Detections) -> list[BallCandidate]:
        balls = detections.filter_by_class(ObjectClass.BALL)
        return self._to_candidates(frame_idx, balls, np.zeros(2, dtype=np.float32), "full")

    def _roi_candidates(
        self, frame_idx: int, frame: np.ndarray, prediction: np.ndarray
    ) -> list[BallCandidate]:
        """Native-resolution re-detection in a crop centered on the prediction.

        The crop is clamped to the frame by *shifting* (never shrinking), so the
        ROI pass always runs at a constant input size — the ball is seen at its
        true ~10-15 px instead of the 4-7 px the downscaled full-frame pass gets.
        """
        assert self._roi_detector is not None
        size = self._cfg.roi_size
        h, w = frame.shape[:2]
        x0 = int(round(np.clip(prediction[0] - size / 2.0, 0, max(w - size, 0))))
        y0 = int(round(np.clip(prediction[1] - size / 2.0, 0, max(h - size, 0))))
        crop = frame[y0 : y0 + size, x0 : x0 + size]
        balls = self._roi_detector.detect(crop).filter_by_class(ObjectClass.BALL)
        origin = np.array([x0, y0], dtype=np.float32)
        return self._to_candidates(frame_idx, balls, origin, "roi")

    def _advance_kalman(self, candidates: list[BallCandidate]) -> None:
        """Update / (re-)initiate the ROI filter from this frame's gated candidates."""
        if not candidates:
            return
        stale = self._kalman.time_since_update > self._cfg.roi_max_coast
        if not self._kalman.initialized or stale:
            # Past roi_max_coast the ROI has stopped following the prediction, so
            # re-acquire on the strongest evidence instead of associating. Checked
            # before the chi2 gate, whose t^2 velocity variance would otherwise never
            # reject after a long coast.
            best = max(candidates, key=lambda c: c.confidence)
            self._kalman.initiate(best.xy.astype(np.float64))
            self._last_prediction = best.xy.astype(np.float64)
            return

        # Associate only within *physical* reach as well as the chi2 gate: the ball
        # cannot have moved further than max_speed_px per elapsed frame (the same
        # bound the trajectory solver uses), no matter how uncertain the filter is.
        reach = self._cfg.max_speed_px * (self._kalman.time_since_update + 1)
        origin = self._last_prediction
        gated = [
            (d, c)
            for c in candidates
            if (d := self._kalman.gating_distance(c.xy)) < self._cfg.gate_chi2
            and (origin is None or float(np.linalg.norm(c.xy - origin)) <= reach)
        ]
        if gated:
            _, nearest = min(gated, key=lambda dc: dc[0])
            self._kalman.update(nearest.xy.astype(np.float64))


class BallKalman:
    """Constant-velocity Kalman filter over image coordinates, dt = 1 frame.

    Pure numpy (not cv2.KalmanFilter) because the per-frame camera motion must
    warp the *state and covariance* (:meth:`apply_affine`), which cv2's API makes
    awkward — and this stays trivially unit-testable.

    State ``x = [px, py, vx, vy]``.
    """

    def __init__(self, q_pos: float, q_vel: float, r_px: float) -> None:
        self._q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float64)
        self._r = np.eye(2, dtype=np.float64) * (r_px * r_px)
        self._r_px = float(r_px)
        self._f = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
        )
        self._h = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self._x: np.ndarray | None = None  # (4,) state
        self._p: np.ndarray | None = None  # (4,4) covariance
        self._time_since_update = 0

    @property
    def initialized(self) -> bool:
        return self._x is not None

    @property
    def time_since_update(self) -> int:
        """Predict steps since the last measurement (0 right after an update)."""
        return self._time_since_update

    def initiate(self, xy: np.ndarray) -> None:
        """Start the filter at a first measurement (zero velocity, generous vel var)."""
        xy = np.asarray(xy, dtype=np.float64).reshape(2)
        self._x = np.array([xy[0], xy[1], 0.0, 0.0])
        vel_var = (_INIT_VEL_STD_FACTOR * self._r_px) ** 2
        self._p = np.diag([self._r_px**2, self._r_px**2, vel_var, vel_var])
        self._time_since_update = 0

    def apply_affine(self, affine: np.ndarray) -> None:
        """Warp state + covariance with a 2x3 prev→curr affine (the cached GMC).

        Position gets the full affine; velocity only the linear part (a direction
        has no translation); covariance gets the linear part on both blocks.
        """
        if self._x is None or self._p is None:
            return
        a = np.asarray(affine, dtype=np.float64).reshape(2, 3)
        lin, t = a[:, :2], a[:, 2]
        self._x[:2] = lin @ self._x[:2] + t
        self._x[2:] = lin @ self._x[2:]
        jac = np.zeros((4, 4))
        jac[:2, :2] = lin
        jac[2:, 2:] = lin
        self._p = jac @ self._p @ jac.T

    def predict(self) -> np.ndarray:
        """Advance one frame; returns the predicted (2,) position."""
        if self._x is None or self._p is None:
            raise RuntimeError("BallKalman.predict called before initiate.")
        self._x = self._f @ self._x
        self._p = self._f @ self._p @ self._f.T + self._q
        self._time_since_update += 1
        return self._x[:2].copy()

    def update(self, xy: np.ndarray) -> None:
        """Correct with a measurement."""
        if self._x is None or self._p is None:
            raise RuntimeError("BallKalman.update called before initiate.")
        z = np.asarray(xy, dtype=np.float64).reshape(2)
        innovation = z - self._h @ self._x
        s = self._h @ self._p @ self._h.T + self._r
        gain = self._p @ self._h.T @ np.linalg.inv(s)
        self._x = self._x + gain @ innovation
        self._p = (np.eye(4) - gain @ self._h) @ self._p
        # Keep the covariance symmetric against floating-point drift.
        self._p = 0.5 * (self._p + self._p.T)
        self._time_since_update = 0

    def gating_distance(self, xy: np.ndarray) -> float:
        """Squared Mahalanobis distance of a measurement (chi², 2 dof)."""
        if self._x is None or self._p is None:
            raise RuntimeError("BallKalman.gating_distance called before initiate.")
        z = np.asarray(xy, dtype=np.float64).reshape(2)
        innovation = z - self._h @ self._x
        s = self._h @ self._p @ self._h.T + self._r
        return float(innovation @ np.linalg.solve(s, innovation))
