"""Checks a stage script runs before doing any work; each exits cleanly instead of tracing back."""

from __future__ import annotations

import sys
from pathlib import Path

from football_tracker.config import Config
from football_tracker.io.video import probe_video

FPS_TOLERANCE = 0.5


def require_file(path: str | Path, arg: str) -> Path:
    """Return ``path`` if it exists, else exit naming the ``--arg`` it came from."""
    p = Path(path)
    if not p.exists():
        sys.exit(f"--{arg} not found: {p} (check the path is correct and the file exists).")
    return p


def require_contiguous_frames(cfg: Config) -> None:
    """Exit unless ``io.frame_stride == 1``; projection detects gaps by ``frame_idx + 1``."""
    if cfg.io.frame_stride != 1:
        sys.exit(
            f"io.frame_stride={cfg.io.frame_stride} is not supported here: projection and "
            "possession assume consecutive decoded frame indices. Re-run with io.frame_stride: 1."
        )


def check_projection_fps(cfg: Config) -> None:
    """Exit if ``projection.fps`` disagrees with the clip; warn if the clip cannot be probed."""
    video = cfg.io.input_video
    if video is None:
        return
    try:
        meta = probe_video(video)
    except (FileNotFoundError, OSError) as exc:
        print(
            f"warning: could not probe {video} for fps ({exc}); "
            f"trusting projection.fps={cfg.projection.fps} unchecked.",
            file=sys.stderr,
        )
        return
    if abs(meta.fps - cfg.projection.fps) > FPS_TOLERANCE:
        sys.exit(
            f"projection.fps={cfg.projection.fps} does not match the probed fps of {video} "
            f"({meta.fps:.3f}); ball speed, the possession dropout window and the radar's "
            "playback rate would all be silently wrong. Fix projection.fps in the config."
        )
