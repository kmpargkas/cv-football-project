"""Debug rendering: pitch-line overlays on video frames and a top-down radar."""

from __future__ import annotations

import cv2
import numpy as np

from football_tracker.homography.pitch import PitchSpec


def draw_pitch_overlay(
    frame: np.ndarray,
    H: np.ndarray,
    spec: PitchSpec,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Project the pitch lines through ``H_inv`` and draw them on a copy of ``frame``."""
    out = frame.copy()
    h, w = frame.shape[:2]
    H_inv = np.linalg.inv(H)
    for segment in spec.line_segments():
        pts_h = np.hstack([segment, np.ones((len(segment), 1))]) @ H_inv.T
        valid = np.isfinite(pts_h).all(axis=1) & (pts_h[:, 2] > 1e-9)
        pts_h = pts_h[valid]
        if len(pts_h) < 2:
            continue
        pts = (pts_h[:, :2] / pts_h[:, 2:3]).astype(np.int32)
        # Skip polylines that are entirely far outside the frame.
        margin = 4 * max(h, w)
        if (np.abs(pts) > margin).any():
            continue
        cv2.polylines(out, [pts], isClosed=False, color=color, thickness=thickness)
    return out


def render_radar(
    spec: PitchSpec,
    H: np.ndarray | None,
    frame_shape: tuple[int, int] | None = None,
    scale: float = 8.0,
    pad_px: int = 24,
) -> np.ndarray:
    """Top-down pitch with (optionally) the camera's visible-footprint quad.

    ``scale`` is px per meter. The footprint is the video frame's border
    projected onto the pitch — its stability is the radar eyeball check.
    """
    width = int(spec.length * scale) + 2 * pad_px
    height = int(spec.width * scale) + 2 * pad_px
    canvas = np.full((height, width, 3), (40, 90, 40), dtype=np.uint8)

    def to_px(points_m: np.ndarray) -> np.ndarray:
        return (points_m * scale + pad_px).astype(np.int32)

    for segment in spec.line_segments():
        cv2.polylines(canvas, [to_px(segment)], False, (255, 255, 255), 2)

    if H is not None and frame_shape is not None:
        h, w = frame_shape
        border = np.array([(0, 0), (w, 0), (w, h), (0, h)], dtype=np.float64)
        pts_h = np.hstack([border, np.ones((4, 1))]) @ H.T
        if np.isfinite(pts_h).all() and (np.abs(pts_h[:, 2]) > 1e-9).all():
            quad = pts_h[:, :2] / pts_h[:, 2:3]
            quad = np.clip(quad, (-10.0, -10.0), (spec.length + 10.0, spec.width + 10.0))
            cv2.polylines(canvas, [to_px(quad)], True, (0, 220, 255), 2)
    return canvas
