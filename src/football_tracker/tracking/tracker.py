"""Player tracker: BoT-SORT via BoxMOT, motion-only.

Detections are filtered to on-pitch players, goalkeepers and referees, then associated
with the calibration stage's camera motion injected through :class:`PrecomputedCMC`.

BoxMOT (22.0) contract this wraps:
- ``update(dets, img)`` takes ``(N, 6)`` ``[x1,y1,x2,y2,conf,cls]`` + the BGR frame and
  returns ``(M, 8)`` ``[x1,y1,x2,y2,id,conf,cls,det_ind]``.
- Camera motion enters through the public ``.cmc`` attribute (``self.cmc.apply(img, boxes)``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from football_tracker.config import TrackingConfig
from football_tracker.detection.interface import Detections, ObjectClass
from football_tracker.homography.pitch import PitchSpec
from football_tracker.homography.record import FrameCalibration
from football_tracker.tracking.gmc import PrecomputedCMC, off_field_mask

# Classes that enter the player tracker; the ball is tracked separately.
_TRACKED_CLASSES = (ObjectClass.PLAYER, ObjectClass.GOALKEEPER, ObjectClass.REFEREE)


@dataclass(frozen=True)
class TrackedObjects:
    """One frame's tracks: parallel arrays of boxes, IDs, scores, and classes."""

    xyxy: np.ndarray  # (M, 4)
    id: np.ndarray  # (M,) int track IDs
    confidence: np.ndarray  # (M,)
    class_id: np.ndarray  # (M,) ObjectClass ints

    def __len__(self) -> int:
        return len(self.id)

    @classmethod
    def empty(cls) -> TrackedObjects:
        return cls(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            id=np.zeros((0,), dtype=np.int64),
            confidence=np.zeros((0,), dtype=np.float32),
            class_id=np.zeros((0,), dtype=np.int64),
        )


class PlayerTracker:
    """BoT-SORT with the cached camera motion injected and off-field detections rejected."""

    def __init__(
        self,
        tracker: object,
        cmc: PrecomputedCMC | None,
        calib_by_idx: dict[int, FrameCalibration],
        spec: PitchSpec,
        mask_margin_m: float,
        min_conf: float,
        class_margins: dict[int, float] | None = None,
    ) -> None:
        self._tracker = tracker
        self._cmc = cmc
        self._calib_by_idx = calib_by_idx
        self._spec = spec
        self._mask_margin_m = mask_margin_m
        self._min_conf = min_conf
        self._class_margins = class_margins or {}

    @classmethod
    def from_config(
        cls,
        cfg: TrackingConfig,
        stream: list[FrameCalibration],
        spec: PitchSpec,
        fps: float,
    ) -> PlayerTracker:
        """Build BoT-SORT with the calibration stream's camera motion injected."""
        from boxmot.trackers.bbox.botsort import BotSort

        tracker = BotSort(
            reid_model=None,
            with_reid=False,  # motion-only
            use_cmc=True,  # keeps BoxMOT's compensation path live for the injected warp
            cmc_method="ecc",
            frame_rate=max(round(fps), 1),
            track_high_thresh=cfg.track_high_thresh,
            track_low_thresh=cfg.track_low_thresh,
            new_track_thresh=cfg.new_track_thresh,
            match_thresh=cfg.match_thresh,
            track_buffer=cfg.track_buffer,
        )
        cmc = PrecomputedCMC.from_stream(stream)
        tracker.cmc = cmc  # replace BoxMOT's ECC estimator with the cached GMC

        return cls(
            tracker,
            cmc,
            {c.frame_idx: c for c in stream},
            spec,
            cfg.mask_margin_m,
            cfg.min_conf,
            class_margins={int(ObjectClass.REFEREE): cfg.mask_margin_referee_m},
        )

    def update(self, frame_idx: int, frame: np.ndarray, detections: Detections) -> TrackedObjects:
        """Associate one frame's detections into tracks.

        Drops the ball + off-field detections, points the injected GMC at this frame,
        then runs BoT-SORT. ``frame`` must be the same crop the calibration used, so the
        injected affine and the boxes share one pixel space.
        """
        dets = detections.filter_by_class(*_TRACKED_CLASSES).with_min_confidence(self._min_conf)
        keep = off_field_mask(
            dets,
            self._calib_by_idx.get(frame_idx),
            self._spec,
            self._mask_margin_m,
            class_margins=self._class_margins,
        )
        dets = Detections(
            xyxy=dets.xyxy[keep], confidence=dets.confidence[keep], class_id=dets.class_id[keep]
        )

        if self._cmc is not None:
            self._cmc.set_frame(frame_idx)

        arr = (
            np.column_stack([dets.xyxy, dets.confidence, dets.class_id]).astype(np.float64)
            if len(dets)
            else np.zeros((0, 6), dtype=np.float64)
        )
        out = np.asarray(self._tracker.update(arr, frame))
        if out.ndim < 2 or len(out) == 0:
            return TrackedObjects.empty()

        return TrackedObjects(
            xyxy=out[:, :4].astype(np.float32),
            id=out[:, 4].astype(np.int64),
            confidence=out[:, 5].astype(np.float32),
            class_id=out[:, 6].astype(np.int64),
        )
