"""The per-frame calibration record, and the schema of ``calibration.npz``.

Tracking and projection read ``H``, ``gmc`` and ``low_confidence`` — the homography, the
camera motion measured beside it, and whether to trust them. ``source``, ``inliers`` and
``reproj_err_px`` are diagnostics, read only by the calibration video and its metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

CalibrationSource = Literal["none", "solved", "propagated"]


@dataclass(frozen=True)
class FrameCalibration:
    """Calibration output for one frame."""

    frame_idx: int
    H: np.ndarray | None  # image px → pitch m; None if no anchor solved at all
    source: CalibrationSource
    low_confidence: bool
    gmc: np.ndarray  # (2, 3) prev→curr affine, identity on the first frame
    inliers: int = 0  # of the anchor solve at this frame; 0 where none was solved
    reproj_err_px: float = float("nan")
