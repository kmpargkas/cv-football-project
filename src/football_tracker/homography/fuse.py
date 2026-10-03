"""Turn a few hand-clicked anchors and a camera-motion stream into one H per frame.

An anchor is correct only on the frame it was clicked, so between anchors the homography
is carried by optical flow. Carrying it one way and replacing it outright at the next
anchor dumps a whole gap's disagreement — accumulated drift plus labelling noise — onto a
single frame: mid-gap a pitch point moves ~2 mm between frames, but 0.3-1 m at each anchor
and up to 9 m near the far touchline. Averaged positions survive that; speed and distance
covered do not.

The clip is offline, so the future is available: H is propagated forward from the anchor
before and backward from the anchor after, then cross-faded across the gap. The
disagreement spreads evenly, and mid-gap accuracy improves as a side effect — one-way
drift is one-sided, while two estimates bracket the truth. The cross-fade runs over
projected control points, never matrix entries, since the average of two valid
homographies need not be valid.

:func:`fused_calibration_stream` feeds ``calibration.npz`` and everything downstream;
:func:`fuse_anchors` returns bare H and is what the held-out gate scores.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from football_tracker.homography.anchors import Anchor
from football_tracker.homography.pitch import PitchSpec, default_control_points
from football_tracker.homography.propagate import IDENTITY_MOTION, FrameMotion, propagate_h
from football_tracker.homography.record import FrameCalibration
from football_tracker.homography.solve import (
    HomographySolution,
    project,
    solve_homography,
    validate,
)

# The accept/reject gate applied to an anchor's own fit. The two pixel thresholds are
# deliberately looser than solve's own defaults (5.0 and 8.0 px); MIN_INLIERS matches.
RANSAC_THRESH_PX = 10.0
MIN_INLIERS = 6
MAX_REPROJ_ERR_PX = 12.0


def blend_homographies(
    H_a: np.ndarray,
    H_b: np.ndarray,
    weight: float,
    control_points: np.ndarray,
) -> np.ndarray:
    """Cross-fade two image→pitch homographies; ``weight=0`` is ``H_a``, ``1`` is ``H_b``.

    Control points are projected pitch→image through each H, linearly mixed there, and an
    exact homography refitted through the result. Falls back to whichever input the weight
    is nearer if either projection is non-finite or the refit degenerates.
    """
    img_a = project(np.linalg.inv(H_a), control_points)
    img_b = project(np.linalg.inv(H_b), control_points)
    if not (np.isfinite(img_a).all() and np.isfinite(img_b).all()):
        return H_b if weight >= 0.5 else H_a
    mixed = (1.0 - weight) * img_a + weight * img_b
    H_pitch2img, _ = cv2.findHomography(control_points, mixed, 0)
    if H_pitch2img is None or abs(np.linalg.det(H_pitch2img)) < 1e-12:
        return H_b if weight >= 0.5 else H_a
    return np.linalg.inv(H_pitch2img)


def _propagate_back(H_curr: np.ndarray, motion: FrameMotion) -> np.ndarray:
    """Inverse of :func:`propagate_h`: carry H from a frame to the one before it.

    ``propagate_h`` gives ``H_curr = H_prev @ inv(M)``, so ``H_prev = H_curr @ M``.
    """
    return H_curr @ motion.as_homography


def _solved_anchors(
    anchors: Mapping[int, Anchor],
    spec: PitchSpec,
    frame_shape: tuple[int, int] | None,
    ransac_thresh_px: float,
    min_inliers: int,
    max_reproj_err_px: float,
) -> dict[int, HomographySolution]:
    """Fit each anchor to its own clicked points, keeping the fits that pass the gate.

    Returns the whole :class:`HomographySolution` rather than just H: the inlier count and
    reprojection error are what the calibration video and its metrics report at anchor
    frames.
    """
    solved: dict[int, HomographySolution] = {}
    for frame, (image_pts, pitch_pts) in anchors.items():
        solution = solve_homography(image_pts, pitch_pts, ransac_thresh_px)
        if solution is None:
            continue
        if frame_shape is not None and not validate(
            solution,
            spec.boundary,
            frame_shape,
            min_inliers=min_inliers,
            max_reproj_err_px=max_reproj_err_px,
        ):
            continue
        solved[int(frame)] = solution
    return solved


def _fuse(
    solved: Mapping[int, HomographySolution],
    motions: Mapping[int, FrameMotion],
    frames: Sequence[int],
    spec: PitchSpec,
) -> dict[int, np.ndarray | None]:
    """Forward sweep, backward sweep, cross-fade — one H per frame from solved anchors.

    Each anchor's H is carried forward through the motion stream and, separately, backward
    from the anchor after; every frame then blends the two by how far it sits between them.
    Frames outside the span of solved anchors have only one estimate and take it unblended.
    Takes no RANSAC thresholds — fusing needs only the accepted fits.
    """
    control_points = default_control_points(spec)
    frames = list(frames)
    if not solved:
        return dict.fromkeys(frames)

    forward: dict[int, np.ndarray | None] = {}
    H: np.ndarray | None = None
    for f in frames:
        if H is not None and f in motions:
            H = propagate_h(H, motions[f])
        if f in solved:
            H = solved[f].H
        forward[f] = H

    backward: dict[int, np.ndarray | None] = {}
    H = None
    for i in range(len(frames) - 1, -1, -1):
        f = frames[i]
        # Step back across the motion that led *into* the frame we just left.
        if H is not None and i + 1 < len(frames) and (nxt := frames[i + 1]) in motions:
            H = _propagate_back(H, motions[nxt])
        if f in solved:
            H = solved[f].H
        backward[f] = H

    # Anchor positions along the stream, so the cross-fade is by distance
    # travelled rather than by frame number — the two differ if frames are sparse.
    anchor_positions = [i for i, f in enumerate(frames) if f in solved]
    fused: dict[int, np.ndarray | None] = {}
    for i, f in enumerate(frames):
        H_fwd, H_bwd = forward[f], backward[f]
        if H_fwd is None or H_bwd is None:
            fused[f] = H_fwd if H_fwd is not None else H_bwd
            continue
        prev_pos = max(p for p in anchor_positions if p <= i)
        next_pos = min(p for p in anchor_positions if p >= i)
        span = next_pos - prev_pos
        weight = 0.0 if span == 0 else (i - prev_pos) / span
        fused[f] = blend_homographies(H_fwd, H_bwd, weight, control_points)
    return fused


def fuse_anchors(
    anchors: Mapping[int, Anchor],
    motions: Mapping[int, FrameMotion],
    frames: Sequence[int],
    spec: PitchSpec | None = None,
    frame_shape: tuple[int, int] | None = None,
    ransac_thresh_px: float = RANSAC_THRESH_PX,
    min_inliers: int = MIN_INLIERS,
    max_reproj_err_px: float = MAX_REPROJ_ERR_PX,
) -> dict[int, np.ndarray | None]:
    """Calibrate a whole clip by cross-fading forward and backward propagation.

    ``motions[f]`` is the global motion from frame ``f-1`` to frame ``f`` (the GMC
    stream); a missing entry is treated as no motion. Returns ``{frame: H}``, and ``None``
    at every frame if no anchor was usable at all.
    """
    spec = spec or PitchSpec()
    solved = _solved_anchors(
        anchors, spec, frame_shape, ransac_thresh_px, min_inliers, max_reproj_err_px
    )
    return _fuse(solved, motions, frames, spec)


def fused_calibration_stream(
    anchors: Mapping[int, Anchor],
    motions: Mapping[int, FrameMotion],
    frames: Sequence[int],
    spec: PitchSpec | None = None,
    frame_shape: tuple[int, int] | None = None,
    ransac_thresh_px: float = RANSAC_THRESH_PX,
    min_inliers: int = MIN_INLIERS,
    max_reproj_err_px: float = MAX_REPROJ_ERR_PX,
    max_propagated_frames: int = 240,
) -> list[FrameCalibration]:
    """The fused clip in the shape the rest of the pipeline consumes.

    ``source`` records how each frame got its H: ``"solved"`` at an accepted anchor,
    ``"propagated"`` where it was carried from one, ``"none"`` where nothing was
    available. ``low_confidence`` marks frames further than ``max_propagated_frames``
    from the nearest anchor — the nearer of the two that bracket it, since a fused frame
    is constrained from both sides.
    """
    spec = spec or PitchSpec()
    frames = list(frames)
    solved = _solved_anchors(
        anchors, spec, frame_shape, ransac_thresh_px, min_inliers, max_reproj_err_px
    )
    fused = _fuse(solved, motions, frames, spec)
    stream: list[FrameCalibration] = []
    for f in frames:
        H = fused[f]
        nearest = min((abs(f - a) for a in solved), default=float("inf"))
        at_anchor = f in solved
        stream.append(
            FrameCalibration(
                frame_idx=f,
                H=H,
                source=("none" if H is None else "solved" if at_anchor else "propagated"),
                low_confidence=(H is None or nearest > max_propagated_frames),
                gmc=(motions.get(f) or IDENTITY_MOTION).affine,
                inliers=solved[f].inliers if at_anchor else 0,
                reproj_err_px=solved[f].reproj_err_px if at_anchor else float("nan"),
            )
        )
    return stream
