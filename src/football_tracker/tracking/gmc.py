"""Camera-motion + off-field glue between calibration and the player tracker.

- :class:`PrecomputedCMC` serves the calibration stage's per-frame affines through
  BoxMOT's ``cmc.apply`` contract, so the homography and the tracker share one
  source of camera motion instead of estimating it twice.
- :func:`off_field_mask` keeps only detections whose foot point lands on the pitch,
  and keeps everything when the frame's calibration cannot be trusted.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from football_tracker.detection.interface import Detections
from football_tracker.homography.mask import on_pitch
from football_tracker.homography.pitch import PitchSpec
from football_tracker.homography.record import FrameCalibration

_IDENTITY_2X3 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


class PrecomputedCMC:
    """Serves cached per-frame 2x3 affines through BoxMOT's ``cmc.apply`` contract.

    Assign onto a built tracker (``tracker.cmc = PrecomputedCMC(...)``) and call
    :meth:`set_frame` before each ``tracker.update``. Frames with no cached affine
    get identity.
    """

    def __init__(self, affines: dict[int, np.ndarray]) -> None:
        self._affines = {
            int(k): np.asarray(v, dtype=np.float64).reshape(2, 3) for k, v in affines.items()
        }
        self._frame_idx: int | None = None

    @classmethod
    def from_stream(cls, stream: Iterable[FrameCalibration]) -> PrecomputedCMC:
        """Build from a calibration stream, keyed by ``frame_idx``."""
        return cls({c.frame_idx: c.gmc for c in stream})

    def set_frame(self, frame_idx: int) -> None:
        """Select which frame's affine :meth:`apply` returns next."""
        self._frame_idx = int(frame_idx)

    def apply(self, img: np.ndarray | None = None, dets: np.ndarray | None = None) -> np.ndarray:
        """Return the current frame's 2x3 affine (identity if unknown/missing).

        ``img`` and ``dets`` are ignored; they exist to match BoxMOT's estimator signature.
        """
        if self._frame_idx is None:
            return _IDENTITY_2X3.copy()
        return self._affines.get(self._frame_idx, _IDENTITY_2X3.copy()).copy()


def off_field_mask(
    detections: Detections,
    calib: FrameCalibration | None,
    spec: PitchSpec,
    margin_m: float = 1.0,
    class_margins: Mapping[int, float] | None = None,
) -> np.ndarray:
    """Boolean keep-mask: which detections' foot points land on the (padded) pitch.

    ``class_margins`` overrides ``margin_m`` per :class:`ObjectClass`. Assistant
    referees stand outside the touchline. The margin is widened per class rather
    than globally because the same band holds warming-up substitutes, which the detector
    already labels ``PLAYER``.

    Keeps everything when ``calib`` is missing, has no ``H``, or is low-confidence,
    so an unreliable homography never deletes every player.
    """
    if len(detections) == 0:
        return np.zeros(0, dtype=bool)
    if calib is None or calib.H is None or calib.low_confidence:
        return np.ones(len(detections), dtype=bool)
    if not class_margins:
        return on_pitch(detections.foot_points, calib.H, spec, margin_m)

    foot_points = detections.foot_points
    class_ids = np.asarray(detections.class_id).reshape(-1)
    keep = np.zeros(len(detections), dtype=bool)
    # One on_pitch call per distinct margin, then select per detection.
    per_det = np.array(
        [float(class_margins.get(int(c), margin_m)) for c in class_ids], dtype=np.float64
    )
    for margin in np.unique(per_det):
        rows = per_det == margin
        keep[rows] = on_pitch(foot_points[rows], calib.H, spec, float(margin))
    return keep
