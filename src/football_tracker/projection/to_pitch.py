"""Foot-point × H → pitch metres, plus world-space smoothing.

Smoothing happens here, before any consumer differentiates: ~2 px of
calibration jitter is ~13 cm, which a naive frame delta at 60 fps turns
into phantom m/s.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TypeVar

import numpy as np

from football_tracker.homography.mask import foot_points
from football_tracker.homography.record import FrameCalibration
from football_tracker.homography.solve import project
from football_tracker.tracking.ball_trajectory import BallState
from football_tracker.tracking.tracker import TrackedObjects

_Row = TypeVar("_Row", "PitchPosition", "BallPitchState")


@dataclass(frozen=True)
class PitchPosition:
    """One track's pitch-plane position on one frame."""

    frame_idx: int
    track_id: int
    xy_m: np.ndarray  # (2,) pitch metres
    reliable: bool  # calibration did not flag this frame low-confidence


@dataclass(frozen=True)
class BallPitchState:
    """The ball's pitch-plane position on one frame."""

    frame_idx: int
    xy_m: np.ndarray | None  # None when the ball was missing or the frame had no H
    source: str  # "detected" | "interpolated" | "missing"
    airborne: bool  # upstream flag with near-zero recall; gate on speed instead
    reliable: bool


def project_player_tracks(
    tracks: dict[int, TrackedObjects], calibration: Sequence[FrameCalibration]
) -> list[PitchPosition]:
    """Foot points → pitch metres; frames without H are dropped, low-confidence ones kept."""
    calib = {c.frame_idx: c for c in calibration}
    out: list[PitchPosition] = []
    for frame_idx in sorted(tracks):
        c = calib.get(frame_idx)
        if c is None or c.H is None:
            continue
        objs = tracks[frame_idx]
        pitch = project(c.H, foot_points(objs.xyxy))
        out.extend(
            PitchPosition(frame_idx, int(tid), xy, not c.low_confidence)
            for tid, xy in zip(objs.id, pitch, strict=True)
        )
    return out


def project_ball_states(
    states: dict[int, BallState], calibration: Sequence[FrameCalibration]
) -> list[BallPitchState]:
    """Project the ball trajectory into pitch metres, preserving missing frames."""
    calib = {c.frame_idx: c for c in calibration}
    out: list[BallPitchState] = []
    for frame_idx in sorted(states):
        s = states[frame_idx]
        c = calib.get(frame_idx)
        if s.xy is None or c is None or c.H is None:
            out.append(BallPitchState(frame_idx, None, s.source, s.airborne, False))
            continue
        xy_m = project(c.H, s.xy.reshape(1, 2))[0]
        out.append(BallPitchState(frame_idx, xy_m, s.source, s.airborne, not c.low_confidence))
    return out


def _smooth_run(xy: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average over one contiguous run; edges use edge-padding."""
    w = min(window, len(xy))
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return xy
    pad = w // 2
    padded = np.pad(xy, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(w) / w
    return np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in (0, 1)], axis=1)


def _contiguous_runs(rows: list[_Row]) -> list[list[_Row]]:
    """Split frame-sorted rows into maximal runs of consecutive frame indices."""
    runs: list[list[_Row]] = []
    for r in rows:
        if runs and r.frame_idx == runs[-1][-1].frame_idx + 1:
            runs[-1].append(r)
        else:
            runs.append([r])
    return runs


def _smooth_rows(run: list[_Row], window: int) -> list[_Row]:
    xy = _smooth_run(np.array([r.xy_m for r in run]), window)
    return [replace(r, xy_m=xy[i]) for i, r in enumerate(run)]


def smooth_positions(positions: list[PitchPosition], window: int) -> list[PitchPosition]:
    """Smooth each track's positions in world space. Runs break at frame gaps."""
    by_track: dict[int, list[PitchPosition]] = {}
    for p in positions:
        by_track.setdefault(p.track_id, []).append(p)
    out: list[PitchPosition] = []
    for rows in by_track.values():
        rows.sort(key=lambda p: p.frame_idx)
        for run in _contiguous_runs(rows):
            out.extend(_smooth_rows(run, window))
    out.sort(key=lambda p: (p.frame_idx, p.track_id))
    return out


def smooth_ball(ball: list[BallPitchState], window: int) -> list[BallPitchState]:
    """Smooth the ball path in world space; missing frames break the runs."""
    present = sorted((s for s in ball if s.xy_m is not None), key=lambda s: s.frame_idx)
    out = [s for s in ball if s.xy_m is None]
    for run in _contiguous_runs(present):
        out.extend(_smooth_rows(run, window))
    out.sort(key=lambda s: s.frame_idx)
    return out


def ball_speeds(ball: Sequence[BallPitchState], fps: float) -> dict[int, float]:
    """Ball speed (m/s) per frame from its predecessor; a frame after a gap has no entry."""
    speeds: dict[int, float] = {}
    prev: BallPitchState | None = None
    for b in sorted(ball, key=lambda s: s.frame_idx):
        if (
            b.xy_m is not None
            and prev is not None
            and prev.xy_m is not None
            and b.frame_idx == prev.frame_idx + 1
        ):
            speeds[b.frame_idx] = float(np.linalg.norm(b.xy_m - prev.xy_m)) * fps
        prev = b
    return speeds
