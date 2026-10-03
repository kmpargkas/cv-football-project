"""Frame-to-frame global camera motion via sparse LK flow + similarity RANSAC.

A 4-DOF similarity (translation + rotation + scale) absorbs the camera's pan,
tilt, roll and zoom. Independently moving players are a minority of tracked
features and are rejected by RANSAC as outliers.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameMotion:
    """Global similarity motion from the previous frame to the current one."""

    affine: np.ndarray  # (2, 3) prev-frame px → curr-frame px (the GMC matrix)

    @property
    def as_homography(self) -> np.ndarray:
        """(3, 3) version of :attr:`affine`."""
        return np.vstack([self.affine, (0.0, 0.0, 1.0)])


IDENTITY_MOTION = FrameMotion(affine=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
IDENTITY_MOTION.affine.flags.writeable = False


class MotionEstimator:
    """Streaming global-motion estimator; call :meth:`step` once per frame."""

    def __init__(
        self,
        downscale: float = 0.5,
        max_corners: int = 800,
        quality_level: float = 0.01,
        min_distance: int = 15,
        ransac_thresh_px: float = 2.0,
        min_inliers: int = 30,
    ) -> None:
        self.downscale = downscale
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.ransac_thresh_px = ransac_thresh_px
        self.min_inliers = min_inliers
        self._prev_gray: np.ndarray | None = None

    def step(self, frame: np.ndarray) -> FrameMotion | None:
        """Estimate motion from the previous frame to ``frame`` (BGR or gray).

        Returns ``None`` on the first frame and whenever a reliable estimate
        can't be made (too few features/inliers).
        """
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.downscale != 1.0:
            gray = cv2.resize(gray, None, fx=self.downscale, fy=self.downscale)
        prev, self._prev_gray = self._prev_gray, gray
        if prev is None:
            return None
        pts = cv2.goodFeaturesToTrack(
            prev,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=7,
        )
        if pts is None or len(pts) < self.min_inliers:
            return None
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            prev, gray, pts, None, winSize=(21, 21), maxLevel=3
        )
        ok = status.ravel() == 1
        good_prev = pts[ok].reshape(-1, 2)
        good_next = nxt[ok].reshape(-1, 2)
        if len(good_prev) < self.min_inliers:
            return None
        affine, inlier_mask = cv2.estimateAffinePartial2D(
            good_prev,
            good_next,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh_px,
        )
        if affine is None or not np.isfinite(affine).all():
            return None
        inliers = int(inlier_mask.ravel().sum())
        if inliers < self.min_inliers:
            return None
        # Rescale the downscaled estimate back to full-resolution pixels: the
        # linear part is scale-invariant, only the translation changes.
        affine = affine.copy()
        affine[:, 2] /= self.downscale
        return FrameMotion(affine=affine)


def propagate_h(H_prev: np.ndarray, motion: FrameMotion) -> np.ndarray:
    """Carry an image→pitch homography forward through one frame of motion.

    ``H_prev`` maps prev-frame px → pitch; ``motion`` maps prev → curr px, so
    the current mapping goes curr px → prev px → pitch.
    """
    return H_prev @ np.linalg.inv(motion.as_homography)
