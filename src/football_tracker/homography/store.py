"""Persist / load a clip's calibration stream (per-frame H + GMC + flags).

Calibration is its own pass: ``scripts/calibrate_video.py`` computes the stream once and
saves it here, and every later stage reads the ``.npz`` instead of recomputing it. It is
stored as parallel arrays, one row per frame, with an absent homography written as
all-NaN.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from football_tracker.homography.record import FrameCalibration

_SOURCES = ("none", "solved", "propagated")


def save_calibration(
    path: str | Path,
    stream: list[FrameCalibration],
    frame_shape: tuple[int, int] | None = None,
) -> None:
    """Write a calibration stream to a compressed ``.npz``.

    ``frame_shape`` is the ``(height, width)`` of the cropped frames the stream was
    fitted in. It is recorded because ``H`` maps *those* pixels to pitch metres and
    means nothing without them: a reader holding only this file cannot otherwise tell
    what image its coordinates refer to. Stream-level, so it is stored once rather
    than per frame, and omitted entirely when the caller does not know it.
    """
    n = len(stream)
    H = np.full((n, 3, 3), np.nan)
    gmc = np.zeros((n, 2, 3))
    frame_idx = np.zeros(n, dtype=np.int64)
    source = np.zeros(n, dtype=np.int8)
    low_conf = np.zeros(n, dtype=bool)
    inliers = np.zeros(n, dtype=np.int32)
    reproj = np.full(n, np.nan)
    for i, c in enumerate(stream):
        frame_idx[i] = c.frame_idx
        if c.H is not None:
            H[i] = c.H
        gmc[i] = c.gmc
        source[i] = _SOURCES.index(c.source)
        low_conf[i] = c.low_confidence
        inliers[i] = c.inliers
        reproj[i] = c.reproj_err_px
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "frame_idx": frame_idx,
        "H": H,
        "gmc": gmc,
        "source": source,
        "low_confidence": low_conf,
        "inliers": inliers,
        "reproj_err_px": reproj,
    }
    if frame_shape is not None:
        arrays["frame_shape"] = np.asarray(frame_shape, dtype=np.int64)
    np.savez_compressed(path, **arrays)


def load_calibration(path: str | Path) -> list[FrameCalibration]:
    """Load a stream saved by :func:`save_calibration`."""
    path = Path(path)
    with np.load(path) as data:
        # Read each array once. NpzFile decompresses on every lookup, so indexing it
        # inside the loop leaves every frame holding a view into its own copy (120*n**2 B).
        frame_idx = data["frame_idx"]
        H = data["H"]
        gmc = data["gmc"]
        source = data["source"]
        low_confidence = data["low_confidence"]
        inliers = data["inliers"]
        reproj_err_px = data["reproj_err_px"]

    stream: list[FrameCalibration] = []
    for i in range(len(frame_idx)):
        h = H[i]
        stream.append(
            FrameCalibration(
                frame_idx=int(frame_idx[i]),
                H=None if np.isnan(h).any() else h,
                source=_SOURCES[int(source[i])],
                low_confidence=bool(low_confidence[i]),
                gmc=gmc[i],
                inliers=int(inliers[i]),
                reproj_err_px=float(reproj_err_px[i]),
            )
        )
    return stream


def load_frame_shape(path: str | Path) -> tuple[int, int] | None:
    """The ``(height, width)`` a stream's homographies were fitted in, if recorded.

    ``None`` for a stream saved without one. A caller must decide what to do in that
    case rather than assume a resolution -- guessing wrong silently changes any
    geometry checked against the frame bounds.
    """
    with np.load(path) as data:
        if "frame_shape" not in data:
            return None
        height, width = data["frame_shape"]
    return int(height), int(width)
