"""Fit a homography from correspondences, and decide whether to trust it.

``H`` always maps **cropped-image pixels → pitch meters**; ``H_inv`` maps back. A candidate
is refused in three separate ways: :func:`solve_homography` rejects what is numerically
broken, :func:`plausible` rejects geometry no camera could produce, and :func:`validate`
applies the accept thresholds on top.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MIN_POINTS = 4  # a homography needs 4 correspondences


@dataclass(frozen=True)
class HomographySolution:
    """A fitted homography with the quality figures :func:`validate` judges it by."""

    H: np.ndarray  # (3, 3) image px → pitch meters
    inliers: int
    total: int
    reproj_err_px: float

    @property
    def H_inv(self) -> np.ndarray:
        return np.linalg.inv(self.H)


def project(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a homography to (N, 2) points. Returns (N, 2)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def solve_homography(
    image_points: np.ndarray,
    pitch_points: np.ndarray,
    ransac_thresh_px: float = 5.0,
) -> HomographySolution | None:
    """Fit image→pitch H from correspondences via RANSAC.

    ``ransac_thresh_px`` is applied in the *image* domain: we fit pitch→image
    (where the threshold has a stable pixel meaning regardless of zoom) and
    invert. Returns ``None`` when under-determined or degenerate.
    """
    image_points = np.asarray(image_points, dtype=np.float64)
    pitch_points = np.asarray(pitch_points, dtype=np.float64)
    if len(image_points) < MIN_POINTS:
        return None
    H_pitch2img, inlier_mask = cv2.findHomography(
        pitch_points, image_points, cv2.RANSAC, ransac_thresh_px
    )
    if H_pitch2img is None or not np.isfinite(H_pitch2img).all():
        return None
    det = np.linalg.det(H_pitch2img)
    if abs(det) < 1e-12:
        return None
    inliers = inlier_mask.ravel().astype(bool)
    if inliers.sum() < MIN_POINTS:
        return None
    err = np.linalg.norm(
        project(H_pitch2img, pitch_points[inliers]) - image_points[inliers], axis=1
    )
    return HomographySolution(
        H=np.linalg.inv(H_pitch2img),
        inliers=int(inliers.sum()),
        total=len(image_points),
        reproj_err_px=float(np.median(err)),
    )


def _project_homogeneous(M: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    """Project (N, 2) points through a 3x3 matrix, or ``None`` if the result is unusable."""
    pts_h = np.hstack([points, np.ones((len(points), 1))]) @ M.T
    w = pts_h[:, 2]
    if not np.isfinite(pts_h).all() or np.any(np.abs(w) < 1e-9):
        return None
    if not (np.all(w > 0) or np.all(w < 0)):
        return None
    return pts_h[:, :2] / w[:, None]


def plausible(
    solution: HomographySolution,
    pitch_boundary: np.ndarray,
    frame_shape: tuple[int, int],
) -> bool:
    """Geometric sanity: the projected pitch must look like a pitch.

    Checks that the pitch boundary projects to a convex quad entirely in front of the
    camera, that the frame maps back onto the pitch the same way, and that the visible
    area isn't absurdly small or large relative to the pitch itself.
    """
    quad = _project_homogeneous(solution.H_inv, pitch_boundary)
    if quad is None or not _convex(quad):
        return False
    height, width = frame_shape
    frame_corners = np.array([(0, 0), (width, 0), (width, height), (0, height)], dtype=np.float64)
    frame_on_pitch = _project_homogeneous(solution.H, frame_corners)
    if frame_on_pitch is None:
        return False
    # Frame area in pitch m² must be a sane fraction of the pitch itself:
    # too small ⇒ collapsed H, too large ⇒ exploded H. Generous bounds — the
    # camera never frames less than ~5% or more than ~20× the pitch area.
    pitch_area = _polygon_area(pitch_boundary)
    frame_area = _polygon_area(frame_on_pitch)
    return 0.05 * pitch_area <= frame_area <= 20.0 * pitch_area


def validate(
    solution: HomographySolution | None,
    pitch_boundary: np.ndarray,
    frame_shape: tuple[int, int],
    min_inliers: int = 6,
    max_reproj_err_px: float = 8.0,
) -> bool:
    """Accept/reject gate for a candidate solve."""
    return (
        solution is not None
        and solution.inliers >= min_inliers
        and solution.reproj_err_px <= max_reproj_err_px
        and plausible(solution, pitch_boundary, frame_shape)
    )


def _convex(quad: np.ndarray) -> bool:
    """True if a 4-point polygon is convex (consistent cross-product signs)."""
    crosses = []
    for i in range(len(quad)):
        a = quad[(i + 1) % len(quad)] - quad[i]
        b = quad[(i + 2) % len(quad)] - quad[(i + 1) % len(quad)]
        crosses.append(a[0] * b[1] - a[1] * b[0])  # 2D cross product (z component)
    crosses = np.array(crosses)
    return bool(np.all(crosses > 0) or np.all(crosses < 0))


def _polygon_area(points: np.ndarray) -> float:
    """Shoelace area of an (N, 2) polygon."""
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)
