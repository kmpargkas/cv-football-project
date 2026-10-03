"""Two questions about a calibration: is each anchor sound, and does the stream jump?

:func:`anchor_quality` asks how well a single hand-labelled anchor agrees with itself.
Accuracy tracks **per-anchor inlier ratio** more closely than anchor count or spacing:

    inlier ratio 71-85% -> self-fit ~0.28 m
    inlier ratio ~50% -> self-fit ~0.71 m

and a fused stream lands at its anchors' own quality, so it can be no better than they
are. That makes inlier ratio the number to watch while labelling, and ``outlier_indices``
the actionable part: it names the clicks that disagree with every other point in the
frame. This is agreement within a frame, not correctness: an anchor whose every click is
wrong the same way scores perfectly. The held-out gate is what measures accuracy.

:func:`stream_jitter` asks the other question -- not whether the map is right, but
whether it holds still. A discontinuity averages out of a position but not out of a
speed, so a stream can be accurate at every anchor and still ruin distance covered.

``scripts/calibrate_video.py`` reports both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from football_tracker.homography.pitch import PitchSpec, default_control_points
from football_tracker.homography.record import FrameCalibration
from football_tracker.homography.solve import project, solve_homography

# How near a goal line a point must be to count as reaching that goal area. The
# penalty-box line (16.5 m in) deliberately does NOT count: stopping there is exactly
# the habit that left the fit extrapolating over the outer 16 m.
END_ZONE_M = 11.0

# Unexplained movement below this is indistinguishable from noise in the motion
# estimate it is measured against -- it matches `propagate.MotionEstimator`'s own
# RANSAC threshold, so anything under it is inside the GMC's margin of error.
JITTER_NOISE_PX = 2.0


@dataclass(frozen=True)
class AnchorQuality:
    """One anchor's self-consistency and pitch coverage."""

    n_points: int
    n_inliers: int
    inlier_ratio: float
    median_residual_m: float
    x_min: float
    x_max: float
    reaches_left: bool
    reaches_right: bool
    outlier_indices: list[int]  # worst first

    @property
    def end_reach(self) -> str:
        if self.reaches_left and self.reaches_right:
            return "BOTH"
        if self.reaches_left:
            return "LEFT"
        if self.reaches_right:
            return "RIGHT"
        return "neither"


def anchor_quality(
    image_pts: np.ndarray,
    pitch_pts: np.ndarray,
    spec: PitchSpec,
    ransac_thresh_px: float = 10.0,
) -> AnchorQuality:
    """Fit this anchor to its own points and report how well they agree.

    ``outlier_indices`` are the points RANSAC rejected, ordered worst first --
    those are the clicks to revisit. An anchor that cannot be solved at all comes
    back with zero inliers and a NaN residual rather than raising, so a caller can
    report it alongside the others.
    """
    image_pts = np.asarray(image_pts, dtype=float)
    pitch_pts = np.asarray(pitch_pts, dtype=float)
    n = len(image_pts)
    xs = pitch_pts[:, 0] if n else np.array([np.nan])

    base = {
        "n_points": n,
        "x_min": float(xs.min()) if n else float("nan"),
        "x_max": float(xs.max()) if n else float("nan"),
        "reaches_left": bool(n and xs.min() <= END_ZONE_M),
        "reaches_right": bool(n and xs.max() >= spec.length - END_ZONE_M),
    }

    sol = solve_homography(image_pts, pitch_pts, ransac_thresh_px) if n >= 4 else None
    if sol is None:
        return AnchorQuality(
            **base,
            n_inliers=0,
            inlier_ratio=0.0,
            median_residual_m=float("nan"),
            outlier_indices=[],
        )

    residual_m = np.linalg.norm(project(sol.H, image_pts) - pitch_pts, axis=1)
    # Re-derive membership in image space, matching the threshold's units.
    residual_px = np.linalg.norm(project(np.linalg.inv(sol.H), pitch_pts) - image_pts, axis=1)
    is_outlier = residual_px > ransac_thresh_px
    order = np.argsort(-residual_px)
    outliers = [int(i) for i in order if is_outlier[i]]

    return AnchorQuality(
        **base,
        n_inliers=int((~is_outlier).sum()),
        inlier_ratio=float((~is_outlier).mean()),
        median_residual_m=float(np.median(residual_m)),
        outlier_indices=outliers,
    )


@dataclass(frozen=True)
class StreamJitter:
    """How still a calibration stream holds, frame to frame.

    ``residual_px`` is unexplained movement: how far the projected pitch moves between
    two consecutive calibrated frames *beyond* what the measured camera motion accounts
    for. ``frame_idx`` is parallel to it, naming the later frame of each pair.
    """

    frame_idx: np.ndarray
    residual_px: np.ndarray
    median_px: float
    p90_px: float
    max_px: float
    worst_frames: list[int]  # worst first, and only those above the noise floor

    @property
    def n_pairs(self) -> int:
        return int(self.residual_px.size)


def stream_jitter(
    stream: list[FrameCalibration],
    spec: PitchSpec,
    noise_floor_px: float = JITTER_NOISE_PX,
) -> StreamJitter:
    """Movement of the projected pitch that the measured camera motion does not explain.

    Control points are projected through one frame's homography, carried through that
    frame's measured motion, and compared against where the next homography puts them.
    Carrying a homography through motion reproduces it exactly, so propagation scores
    zero and only *forced* frames register: anchors, and the blend around them. Blind to
    drift by design -- seams are what this exists to find. An uncalibrated frame breaks
    the chain rather than bridging it.

    ``worst_frames`` is empty for a stream that never jumps.
    """
    control = default_control_points(spec)
    frames: list[int] = []
    residuals: list[float] = []
    prev_points: np.ndarray | None = None
    for calib in stream:
        if calib.H is None:
            prev_points = None
            continue
        points = project(np.linalg.inv(calib.H), control)
        if prev_points is not None:
            # The affine carries a (0, 0, 1) bottom row, so w is identically 1 and the
            # projected points need no perspective divide.
            carried = (
                np.hstack([prev_points, np.ones((len(prev_points), 1))])
                @ np.vstack([calib.gmc, (0.0, 0.0, 1.0)]).T
            )
            residuals.append(float(np.median(np.linalg.norm(points - carried[:, :2], axis=1))))
            frames.append(calib.frame_idx)
        prev_points = points

    frame_idx = np.asarray(frames, dtype=np.int64)
    residual_px = np.asarray(residuals, dtype=np.float64)
    if not residual_px.size:
        return StreamJitter(
            frame_idx=frame_idx,
            residual_px=residual_px,
            median_px=float("nan"),
            p90_px=float("nan"),
            max_px=float("nan"),
            worst_frames=[],
        )
    order = np.argsort(-residual_px)
    return StreamJitter(
        frame_idx=frame_idx,
        residual_px=residual_px,
        median_px=float(np.median(residual_px)),
        p90_px=float(np.percentile(residual_px, 90)),
        max_px=float(residual_px.max()),
        worst_frames=[int(frame_idx[i]) for i in order if residual_px[i] > noise_floor_px],
    )
