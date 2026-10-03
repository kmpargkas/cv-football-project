"""What counts as on the pitch

Filters out everything outside the pitch through the `on_pitch` function.
The `pitch_mask` function serves as a visual check: the line overlay shows wether H
is right, the mask shows what the filter will keep.
"""

from __future__ import annotations

import cv2
import numpy as np

from football_tracker.homography.pitch import PitchSpec
from football_tracker.homography.solve import project


def foot_points(boxes_xyxy: np.ndarray) -> np.ndarray:
    """Bottom-centre image points of (N, 4) xyxy boxes → (N, 2)."""
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]], axis=1)


def on_pitch(
    image_points: np.ndarray,
    H: np.ndarray,
    spec: PitchSpec,
    margin_m: float = 1.0,
) -> np.ndarray:
    """Boolean mask: which (N, 2) image points land on the (padded) pitch."""
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    pitch_pts = project(H, image_points)
    finite = np.isfinite(pitch_pts).all(axis=1)
    inside = (
        (pitch_pts[:, 0] >= -margin_m)
        & (pitch_pts[:, 0] <= spec.length + margin_m)
        & (pitch_pts[:, 1] >= -margin_m)
        & (pitch_pts[:, 1] <= spec.width + margin_m)
    )
    return finite & inside


def pitch_mask(
    H: np.ndarray,
    frame_shape: tuple[int, int],
    spec: PitchSpec,
    margin_m: float = 1.0,
    samples_per_edge: int = 50,
) -> np.ndarray:
    """Render the (padded) pitch as a uint8 mask (255 = pitch) in image space.

    The boundary is sampled densely and projected point-wise so that a pitch
    partially outside the frame (or with far corners near the horizon) still
    produces a sane polygon: non-finite / behind-camera samples are dropped.
    """
    h, w = frame_shape
    boundary = _padded_boundary(spec, margin_m)
    # Densify each edge so the projected polygon follows perspective curvature.
    dense = []
    for i in range(len(boundary)):
        a, b = boundary[i], boundary[(i + 1) % len(boundary)]
        t = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False)[:, None]
        dense.append(a + t * (b - a))
    dense = np.vstack(dense)
    # Project pitch→image in homogeneous coordinates so w survives the trip: samples on
    # the horizon (w ~ 0) go first, then everything on the far side of it, because points
    # from behind the camera fold the polygon through infinity. The count is checked after
    # both filters -- cv2.fillPoly raises on an empty array.
    H_inv = np.linalg.inv(H)
    pts_h = np.hstack([dense, np.ones((len(dense), 1))]) @ H_inv.T
    valid = np.isfinite(pts_h).all(axis=1) & (np.abs(pts_h[:, 2]) > 1e-9)
    pts_h = pts_h[valid]
    if len(pts_h):
        w_sign = np.sign(np.median(pts_h[:, 2]))
        pts_h = pts_h[np.sign(pts_h[:, 2]) == w_sign]
    if len(pts_h) < 3:
        return np.zeros((h, w), dtype=np.uint8)
    polygon = (pts_h[:, :2] / pts_h[:, 2:3]).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def _padded_boundary(spec: PitchSpec, margin_m: float) -> np.ndarray:
    """Pitch rectangle grown outward by ``margin_m`` on every side."""
    return np.array(
        [
            (-margin_m, -margin_m),
            (spec.length + margin_m, -margin_m),
            (spec.length + margin_m, spec.width + margin_m),
            (-margin_m, spec.width + margin_m),
        ],
        dtype=np.float64,
    )
