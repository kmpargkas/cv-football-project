"""Read the hand-labelled anchors that pitch calibration is built on.

A handful of frames are clicked against known pitch features, and ``fuse`` turns those
correspondences into a homography for every frame of the clip. They are the calibration
path's only ground truth, so a click that is wrong rather than merely hard propagates —
which is why reading them involves filtering at all.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from football_tracker.homography.pitch import PHANTOM_VERTICES
from football_tracker.homography.solve import MIN_POINTS

# One anchor frame's ground truth: (image_pts (N, 2), pitch_pts (N, 2)).
Anchor = tuple[np.ndarray, np.ndarray]


def _coordinates(points: list[dict], field: str, path: Path, frame: str) -> np.ndarray:
    """(N, 2) float array of one coordinate field across a frame's points."""
    try:
        return np.asarray([p[field] for p in points], dtype=np.float64).reshape(-1, 2)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}: anchor frame {frame} has a point whose {field!r} is missing "
            "or is not an [x, y] pair"
        ) from exc


def load_anchors(
    path: str | Path, *, drop_phantom: bool = True, drop_excluded: bool = True
) -> dict[int, Anchor]:
    """Parse a ``pitch_points.json`` (from ``scripts/annotate_pitch_points.py``)
    into ``{frame_idx: (image_pts, pitch_pts)}``.

    Image coords are in the *cropped* frame, matching the H convention — the same
    file also serves as the eval ground truth, so anchors and eval share a format.

    ``drop_phantom`` removes clicks on :data:`PHANTOM_VERTICES` (unlocalisable by a
    human — see there); ``drop_excluded`` honours a per-point ``"exclude": "<reason>"``
    marking one demonstrably bad click, such as a corner clicked at the frame edge
    because the real one is off-screen. Both filter *evaluation* as well as fitting,
    because scoring against a click that is not on the feature it names measures
    agreement with the mistake; both are switchable so a caller can report what its own
    filtering removed. A point carrying no ``vertex`` is always kept — the field is
    optional, and its absence must not silently discard a label.

    A frame filtered below the four points a homography needs keeps all of them, and warns.

    Raises:
        ValueError: a non-integer frame key, or a point whose ``image`` or ``pitch``
            is missing or is not an ``[x, y]`` pair.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    phantom = set(PHANTOM_VERTICES) if drop_phantom else set()

    def keep(point: dict) -> bool:
        if drop_excluded and point.get("exclude"):
            return False
        vertex = point.get("vertex")
        return vertex is None or (int(vertex) - 1) not in phantom

    anchors: dict[int, Anchor] = {}
    for key, points in data["frames"].items():
        # Caught before the filter below can warn: a bad file should be reported as a
        # bad file, not as a complaint about how many points a filter left behind.
        try:
            frame_idx = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: anchor frame key {key!r} is not an integer") from exc
        image = _coordinates(points, "image", path, key)
        pitch = _coordinates(points, "pitch", path, key)
        kept = [i for i, point in enumerate(points) if keep(point)]
        if len(kept) < MIN_POINTS:
            # Warn rather than reinstate silently: clicks the filters just rejected
            # becoming trusted data again is the regression this filtering prevents.
            warnings.warn(
                f"anchor frame {key}: filtering left {len(kept)} of {len(points)} "
                f"points, below the {MIN_POINTS} a homography needs -- keeping the frame "
                "intact with phantom/excluded points reinstated rather than cripple it",
                stacklevel=2,
            )
            kept = list(range(len(points)))
        anchors[frame_idx] = (image[kept], pitch[kept])
    return anchors
